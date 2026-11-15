#!/usr/bin/env python3
"""
feeds_ingest.py - Ingest RSS/Atom feed articles into Qdrant

Reads feed URLs from config/feeds.yaml, fetches new entries, optionally
retrieves full article content via Firecrawl or trafilatura, chunks and
embeds the text, then upserts to Qdrant with deduplication by entry GUID.

The GUIDs already seen are kept in config/.feeds_sync_state.json as digests
rather than verbatim -- see _guid_key() for why.

A feed's optional 'kind' is carried through to the payload as 'feed_kind', so
generated feeds (a scraper standing in for a site that publishes none) stay
distinguishable from feeds the site publishes itself. Defaults to 'rss'.
"""

import sys
import os
import argparse
import logging
import hashlib
import json
import re
import socket
import warnings
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
import yaml
import trafilatura
from dotenv import load_dotenv
from openai import OpenAI

from alert import send_alert, run_guarded
from sync_state import claim_state, write_state
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range, PayloadSchemaType
)

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'

# Minimum character count to consider feed-provided content "full"
CONTENT_THRESHOLD = 500


def main():
    parser = argparse.ArgumentParser(description='RSS/Atom Feeds to Qdrant Ingestion')
    parser.add_argument("-v", action="store_true", default=False, help="Print extra info")
    parser.add_argument("-vv", action="store_true", default=False, help="Print (more) extra info")
    sync_mode = parser.add_mutually_exclusive_group()
    sync_mode.add_argument("--full", action="store_true", help="Full re-sync (ignore state)")
    sync_mode.add_argument("--feeds", nargs='+', metavar='URL',
                           help="Reprocess specific feed URLs only")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Qdrant")
    parser.add_argument("--config", metavar='PATH', help="Path to feeds YAML config file")
    parser.add_argument("--self-test", action="store_true",
                        help="Check the sync-state rules against known cases and exit")
    args = parser.parse_args()

    if args.vv:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.DEBUG)
    elif args.v:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.INFO)
    else:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.WARNING)

    if args.self_test:
        return self_test()

    # Load configuration
    config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
    load_dotenv(config_dir / '.env')

    # Validate required env vars
    required_vars = ['QDRANT_URL', 'OPENAI_API_KEY']
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        log_fatal(f"Missing required environment variables: {', '.join(missing)}")

    # Load feeds config
    feeds_config_path = Path(args.config) if args.config else config_dir / 'feeds.yaml'
    if not feeds_config_path.exists():
        log_fatal(f"Feeds config not found: {feeds_config_path}\n"
                  f"Create it from the example: cp config/feeds.yaml.example config/feeds.yaml")

    feeds_config = load_feeds_config(feeds_config_path)

    max_age_days = int(feeds_config.get('max_age_days', 0))
    min_retention_count = int(feeds_config.get('min_retention_count', 20))

    raw_feeds = feeds_config.get('feeds', [])
    if not raw_feeds:
        log_fatal("No feeds defined in feeds config")

    all_feeds, feed_problems = validate_feeds(raw_feeds)
    for problem in feed_problems:
        logging.error(f"Skipping feed entry: {problem}")
    if not all_feeds:
        log_fatal(f"No usable feed entries in {feeds_config_path}:\n  "
                  + "\n  ".join(feed_problems))

    # Firecrawl settings
    firecrawl_url = feeds_config.get('firecrawl_url', '').rstrip('/')
    firecrawl_timeout = int(feeds_config.get('firecrawl_timeout', 30))
    firecrawl_api_key = os.environ.get('FIRECRAWL_API_KEY', '')

    if firecrawl_url:
        logging.info(f"Firecrawl configured at {firecrawl_url}")
    else:
        logging.info("Firecrawl not configured; will use trafilatura for full-page fetches")

    # Processing settings
    chunk_size = int(os.environ.get('FEEDS_CHUNK_SIZE', os.environ.get('CHUNK_SIZE', 1000)))
    chunk_overlap = int(os.environ.get('FEEDS_CHUNK_OVERLAP', os.environ.get('CHUNK_OVERLAP', 200)))
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
    collection_name = os.environ.get('FEEDS_COLLECTION', 'news_feeds')

    state_file = config_dir / '.feeds_sync_state.json'

    # Another run may already be working on this file: cron fires on a timer
    # whether or not the last run finished, and a run started by hand knows
    # nothing of either. Racing would lose one of the two runs' state.
    if not args.dry_run and not claim_state(state_file):
        return 0

    # Initialize clients
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Api key is used with an insecure connection')
        qdrant = QdrantClient(
            url=os.environ['QDRANT_URL'],
            api_key=os.environ.get('QDRANT_API_KEY')
        )

    openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    if not args.dry_run:
        ensure_collection(qdrant, collection_name)

    # Load state
    state = {}
    if not args.full and state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
    # Bring the ledger into its current shape before anything is compared
    # against it. A targeted run knows only the feeds it was handed, so it must
    # not be allowed to prune -- hence None rather than a partial set.
    processed_map, hashed, dropped = migrate_state(
        state.get('processed', {}),
        None if args.feeds else {f['url'] for f in all_feeds},
    )
    if hashed:
        logging.warning(f"Converted {hashed} stored identifier(s) to digests")
    if dropped:
        logging.warning(f"Dropped {len(dropped)} feed(s) no longer in "
                        f"{feeds_config_path.name} from the sync state:\n  "
                        + "\n  ".join(dropped))

    # Determine which feeds to process
    if args.feeds:
        feed_urls = set(args.feeds)
        target_feeds = [f for f in all_feeds if f['url'] in feed_urls]
        unknown = feed_urls - {f['url'] for f in target_feeds}
        if unknown:
            logging.warning(f"Feed URLs not found in config: {unknown}")
    else:
        target_feeds = all_feeds

    # Process each feed
    total_entries = 0
    total_chunks = 0
    feed_failures = 0

    for feed_def in target_feeds:
        feed_url = feed_def['url']
        feed_name = feed_def.get('name', feed_url)
        do_fetch = feed_def.get('fetch_content', True)
        feed_kind = feed_def.get('kind', 'rss')

        logging.info(f"Fetching feed: {feed_name} ({feed_url})")

        try:
            parsed = feedparser.parse(feed_url)
        except Exception as e:
            logging.error(f"Failed to parse feed {feed_url}: {e}")
            feed_failures += 1
            continue

        if parsed.bozo and parsed.bozo_exception:
            logging.warning(f"Feed parse warning for {feed_url}: {parsed.bozo_exception}")
            # feedparser reports a dead host or unparseable body by setting bozo
            # rather than raising. Bozo alone is common and harmless (a stray
            # entity in otherwise fine XML); bozo with nothing to show for it is
            # the shape of a feed that did not actually arrive.
            if not parsed.entries:
                feed_failures += 1

        entries = parsed.entries
        logging.debug(f"Feed has {len(entries)} entries")

        # Digests of the entries already seen from this feed
        known_keys = set(processed_map.get(feed_url, []))
        new_keys = []

        for entry in entries:
            entry_id = _entry_id(entry)
            guid_key = _guid_key(entry_id)
            if guid_key in known_keys:
                logging.debug(f"Skipping known entry: {entry_id}")
                continue

            try:
                chunks = process_entry(
                    entry, feed_url, feed_name, do_fetch, feed_kind,
                    firecrawl_url, firecrawl_api_key, firecrawl_timeout,
                    openai_client, qdrant, collection_name,
                    chunk_size, chunk_overlap, embedding_model, args.dry_run
                )
                total_chunks += chunks
                total_entries += 1
                new_keys.append(guid_key)
                title = getattr(entry, 'title', entry_id)[:60]
                logging.info(f"  Indexed: {title} ({chunks} chunks)")
            except Exception as e:
                logging.error(f"Failed to process entry {entry_id}: {e}")

        # Update state for this feed
        if not args.dry_run:
            processed_map[feed_url] = list(known_keys | set(new_keys))

        logging.info(f"Feed done: {feed_name} — {len(new_keys)} new entries")

    # Save state
    if not args.dry_run and not args.feeds:
        # last_sync marks a sync that ran to completion, which is the signal
        # health_check.py watches for staleness. The --feeds branch below
        # deliberately leaves it alone: a targeted rerun of one feed must not
        # reset the clock and hide the fact that scheduled runs have stopped.
        write_state(state_file, {
            'processed': processed_map,
            'last_sync': datetime.now(timezone.utc).isoformat(),
        })
    elif not args.dry_run and args.feeds:
        # Merge targeted feed updates into existing state, leaving last_sync as
        # it was: a one-feed rerun must not make a stalled pipeline look healthy.
        if state_file.exists():
            with open(state_file) as f:
                full_state = json.load(f)
        else:
            full_state = {}
        merged = full_state.setdefault('processed', {})
        merged.update(processed_map)
        # The feeds this run was not asked about may still hold raw identifiers
        # from an older state file; convert them too rather than writing a file
        # in two shapes. Pruning stays off: this run cannot tell a retired feed
        # from one it simply was not given.
        full_state['processed'], hashed, _ = migrate_state(merged, None)
        if hashed:
            logging.warning(f"Converted {hashed} stored identifier(s) to digests")
        write_state(state_file, full_state)

    logging.warning(f"Completed: {total_entries} new entries, {total_chunks} chunks indexed")

    if max_age_days > 0:
        feed_urls = [f['url'] for f in all_feeds]
        cull_old_entries(qdrant, collection_name, max_age_days, min_retention_count, feed_urls, args.dry_run)

    # Problems worth mailing about that are not fatal: entries skipped as
    # unusable, and the case where every feed failed. The latter matters because
    # the run still "completes", so it stamps last_sync and leaves health_check
    # reporting green while nothing at all is being indexed.
    problems = list(feed_problems)
    if target_feeds and feed_failures == len(target_feeds):
        problems.append(f"every one of the {len(target_feeds)} feeds failed to fetch or "
                        f"parse, so nothing was indexed - network, DNS or Firecrawl "
                        f"may be down")

    if problems:
        summary = "\n".join(f"  - {p}" for p in problems)
        logging.warning(f"{len(problems)} feed problem(s) this run:\n{summary}")
        send_alert(
            subject=f"[WARN] feeds_ingest: {len(problems)} "
                    f"problem{'' if len(problems) == 1 else 's'} on {socket.gethostname()}",
            body=f"The run finished and indexed what it could, but these problems "
                 f"need attention (config: {feeds_config_path}):\n\n{summary}\n",
        )

    return 0


def _entry_id(entry) -> str:
    """Get a stable identifier for a feed entry"""
    return getattr(entry, 'id', None) or getattr(entry, 'link', None) or ''


# Shape of a ledger key: 16 bytes of blake2b, hex. Anything in a loaded state
# file that does not look like this is an identifier from before the ledger
# stored digests, and gets converted on the way in.
GUID_KEY_RE = re.compile(r'\A[0-9a-f]{32}\Z')


def _guid_key(entry_id: str) -> str:
    """Ledger key for an entry identifier.

    The sync state is only ever asked one question -- "have I seen this id?" --
    so it has no use for the identifier itself, and some feeds mint opaque
    tokens well over a kilobyte long. Storing a 32-character digest instead
    keeps the answer identical (the same id always hashes the same way, so
    nothing is re-fetched or re-embedded) and takes the state file from tens of
    megabytes of strings nothing reads to a fraction of that.

    Deliberately NOT the same derivation as the Qdrant point id in
    process_entry(), which is md5 over the raw entry_id and must stay that way
    -- change it and every article already indexed reappears under a new id.

    surrogatepass keeps a lone surrogate -- which a JSON state file can hold and
    feedparser can produce -- from raising mid-conversion and taking the run
    with it.
    """
    return hashlib.blake2b(entry_id.encode('utf-8', 'surrogatepass'),
                           digest_size=16).hexdigest()


def migrate_state(processed_map: dict,
                  live_feed_urls: set[str] | None) -> tuple[dict, int, list[str]]:
    """Normalise a loaded ledger. Returns (map, identifiers converted, feeds dropped).

    Two jobs, both idempotent:

    * Convert any stored value that is not already a digest, which is how a
      state file written before _guid_key() existed is upgraded in place. The
      conversion is one-way but safe, because both sides of the only comparison
      that matters get hashed. An identifier that already looks like a digest --
      a feed using bare md5 hex as its GUID -- is left alone and so re-indexes
      once; that costs one upsert over the same point ids, not a duplicate,
      since those ids come from the raw entry_id either way.

    * Drop feeds that are not in live_feed_urls, whose entries can never be
      matched again. Pass None to skip this: a caller that knows about only
      some of the feeds (a --feeds run) would otherwise forget every feed it
      was not asked about, and re-fetch and re-embed the lot on the next run.
    """
    migrated: dict[str, list[str]] = {}
    converted = 0
    dropped: list[str] = []

    for feed_url, guids in processed_map.items():
        if live_feed_urls is not None and feed_url not in live_feed_urls:
            dropped.append(feed_url)
            continue
        keys = []
        for guid in guids:
            if isinstance(guid, str) and GUID_KEY_RE.match(guid):
                keys.append(guid)
            else:
                keys.append(_guid_key(guid if isinstance(guid, str) else str(guid)))
                converted += 1
        migrated[feed_url] = keys

    return migrated, converted, dropped


def process_entry(entry, feed_url: str, feed_name: str, do_fetch: bool,
                  feed_kind: str,
                  firecrawl_url: str, firecrawl_api_key: str, firecrawl_timeout: int,
                  openai_client: OpenAI, qdrant: QdrantClient, collection_name: str,
                  chunk_size: int, chunk_overlap: int, embedding_model: str,
                  dry_run: bool = False) -> int:
    """Process a single feed entry: get content, chunk, embed, upsert"""

    entry_id = _entry_id(entry)
    title = getattr(entry, 'title', 'Untitled')
    link = getattr(entry, 'link', '')

    # Extract content from the feed itself
    content = _extract_feed_content(entry)

    # Fetch full content if needed
    if do_fetch and len(content) < CONTENT_THRESHOLD and link:
        logging.debug(f"Content short ({len(content)} chars), fetching full page: {link}")
        fetched = _fetch_content(link, firecrawl_url, firecrawl_api_key, firecrawl_timeout)
        if fetched and len(fetched) > len(content):
            content = fetched
            logging.debug(f"Fetched content: {len(content)} chars")
        else:
            logging.debug("Fetch returned no improvement, using feed content")

    if not content:
        logging.debug(f"Skipping entry {entry_id}: no content")
        return 0

    # Build metadata
    published = _parse_date(entry)
    published_ts = _parse_date_unix(entry)
    author = getattr(entry, 'author', '')
    tags = [t.term for t in getattr(entry, 'tags', []) if hasattr(t, 'term')]

    # Delete existing chunks for this entry (idempotent updates)
    if not dry_run:
        try:
            qdrant.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key='entry_id', match=MatchValue(value=entry_id))]
                )
            )
        except Exception as e:
            logging.debug(f"Delete failed (may not exist): {e}")

    # Chunk content
    chunks = chunk_text(f"{title}\n\n{content}", chunk_size, chunk_overlap)
    if not chunks:
        return 0

    # Embed
    embeddings = get_embeddings(chunks, openai_client, embedding_model)

    # Build points
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = hashlib.md5(f"{entry_id}_{i}".encode()).hexdigest()
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                'entry_id': entry_id,
                'chunk_index': i,
                'title': title,
                'text': chunk,
                'url': link,
                'feed_url': feed_url,
                'feed_name': feed_name,
                'feed_kind': feed_kind,
                'published': published,
                'published_ts': published_ts,
                'author': author,
                'tags': tags,
                'source': 'feed',
            }
        ))

    if not dry_run:
        qdrant.upsert(collection_name=collection_name, points=points)

    return len(points)


def _extract_feed_content(entry) -> str:
    """Extract and clean text content from a feedparser entry"""
    # Prefer full content over summary
    content_list = getattr(entry, 'content', [])
    if content_list:
        raw = content_list[0].get('value', '')
    else:
        raw = getattr(entry, 'summary', '') or getattr(entry, 'description', '')

    if not raw:
        return ''

    # Strip HTML, but keep paragraph boundaries. chunk_text() prefers to break
    # on a blank line, and flattening every tag to a space -- as this did --
    # left it nothing to find, so that preference never once applied to a feed.
    # Block-level tags become a blank line, <br> a single newline, and every
    # other tag a space, so words either side of an inline tag stay separate.
    text = re.sub(r'(?i)<br\s*/?>', '\n', raw)
    text = re.sub(
        r'(?i)</?(?:h[1-6]|blockquote|figcaption|figure|section|article'
        r'|table|tr|pre|div|li|ul|ol|dl|dd|dt|p)(?:\s[^>]*)?/?>',
        '\n\n', text)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Collapse horizontal runs, strip the spaces a break left at line edges,
    # and cap blank lines at one, so '\n\n' is the only paragraph marker.
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _fetch_content(url: str, firecrawl_url: str, firecrawl_api_key: str,
                   timeout: int) -> str:
    """Fetch full article content via Firecrawl or trafilatura fallback"""
    if firecrawl_url:
        return _fetch_via_firecrawl(url, firecrawl_url, firecrawl_api_key, timeout)
    return _fetch_via_trafilatura(url, timeout)


def _fetch_via_firecrawl(url: str, firecrawl_url: str, api_key: str, timeout: int) -> str:
    """Fetch and extract article text using Firecrawl API"""
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    try:
        resp = requests.post(
            f"{firecrawl_url}/v1/scrape",
            json={"url": url, "formats": ["markdown"]},
            headers=headers,
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        # Firecrawl returns {"success": true, "data": {"markdown": "..."}}
        markdown = data.get('data', {}).get('markdown', '')
        logging.debug(f"Firecrawl returned {len(markdown)} chars for {url}")
        return markdown
    except Exception as e:
        logging.warning(f"Firecrawl fetch failed for {url}: {e}")
        return ''


def _fetch_via_trafilatura(url: str, timeout: int) -> str:
    """Fetch page and extract article text using trafilatura"""
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={'User-Agent': 'Mozilla/5.0 (compatible; qdrant-rag/1.0)'})
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
        logging.debug(f"trafilatura extracted {len(text or '')} chars for {url}")
        return text or ''
    except Exception as e:
        logging.warning(f"trafilatura fetch failed for {url}: {e}")
        return ''


def _parse_date_unix(entry) -> float | None:
    """Extract the entry publication date as a Unix timestamp, or None"""
    for attr in ('published_parsed', 'updated_parsed', 'created_parsed'):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).timestamp()
            except Exception:
                pass
    return None


def _parse_date(entry) -> str:
    """Extract and normalise the entry publication date to ISO format"""
    # feedparser provides parsed date tuples
    for attr in ('published_parsed', 'updated_parsed', 'created_parsed'):
        t = getattr(entry, attr, None)
        if t:
            try:
                dt = datetime(*t[:6], tzinfo=timezone.utc)
                return dt.isoformat()
            except Exception:
                pass
    # Fallback to raw string
    for attr in ('published', 'updated', 'created'):
        v = getattr(entry, attr, None)
        if v:
            return str(v)
    return ''



def load_feeds_config(path: Path) -> dict:
    """Parse the feeds YAML, failing with a message that names the offending line.

    One unquoted colon in a feed name is enough to invalidate the whole file.
    The YAMLError that causes used to escape main() and die as a bare traceback,
    so a bad edit stopped the news index silently -- hence the explicit catch
    and the line-and-caret detail in the alert.
    """
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        log_fatal(f"Invalid YAML in {path}\n\n{_yaml_error_detail(path, e)}")
    except OSError as e:
        log_fatal(f"Could not read feeds config {path}: {e}")

    if config is None:
        log_fatal(f"Feeds config is empty: {path}")
    if not isinstance(config, dict):
        log_fatal(f"Feeds config must be a mapping of settings, "
                  f"got {type(config).__name__}: {path}")
    return config


def _yaml_error_detail(path: Path, e: yaml.YAMLError) -> str:
    """Describe a YAML error.

    str(e) already names the file, line and column; add the source line itself
    with a caret under the column, so the alert shows what to fix.
    """
    detail = str(e)
    mark = getattr(e, 'problem_mark', None)
    if mark is None:
        return detail
    try:
        line = path.read_text().splitlines()[mark.line]
    except (OSError, IndexError, UnicodeDecodeError):
        return detail
    return f"{detail}\n\n    {line}\n    {' ' * mark.column}^"


def validate_feeds(raw_feeds) -> tuple[list[dict], list[str]]:
    """Split feed entries into the usable ones and a description of each bad one.

    A malformed entry is skipped rather than fatal, so one typo cannot stop
    every other feed from being ingested. The caller alerts about whatever it
    skipped -- a feed must not be able to drop out of the index quietly.
    """
    if not isinstance(raw_feeds, list):
        return [], [f"'feeds' must be a list of entries, "
                    f"got {type(raw_feeds).__name__}"]

    good: list[dict] = []
    problems: list[str] = []
    for i, feed_def in enumerate(raw_feeds, start=1):
        if not isinstance(feed_def, dict):
            problems.append(f"entry {i}: expected a mapping, "
                            f"got {type(feed_def).__name__} ({feed_def!r:.60})")
            continue
        url = feed_def.get('url')
        if not isinstance(url, str) or not url.strip():
            problems.append(f"entry {i} ({feed_def.get('name', '<unnamed>')}): "
                            f"no usable 'url'")
            continue
        good.append(feed_def)
    return good, problems


# ---------------------------------------------------------------------------
# Shared utilities (same pattern as wallabag_ingest.py)
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient, collection_name: str, dimensions: int = 1536):
    """Create collection if it doesn't exist"""
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE)
        )
        logging.info(f"Created collection: {collection_name}")
    else:
        logging.debug(f"Collection exists: {collection_name}")
    # Payload indexes for efficient culling and for filtering by feed kind
    for field_name, schema in [('published_ts', PayloadSchemaType.FLOAT),
                               ('feed_url', PayloadSchemaType.KEYWORD),
                               ('feed_kind', PayloadSchemaType.KEYWORD)]:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema
            )
            logging.debug(f"Ensured payload index on {field_name}")
        except Exception as e:
            logging.debug(f"Payload index on {field_name} already exists or failed: {e}")


def cull_old_entries(qdrant: QdrantClient, collection_name: str, max_age_days: int,
                     min_retention_count: int, feed_urls: list[str], dry_run: bool = False):
    """Delete points older than max_age_days per feed, but always keep at least
    min_retention_count of the most-recent articles per feed regardless of age."""
    cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)
    total_culled = 0

    for feed_url in feed_urls:
        feed_filter = Filter(must=[FieldCondition(key='feed_url', match=MatchValue(value=feed_url))])

        total = qdrant.count(collection_name=collection_name, count_filter=feed_filter).count
        if total <= min_retention_count:
            logging.info(f"Cull [{feed_url}]: only {total} articles, below min_retention_count={min_retention_count}, skipping")
            continue

        # Find the published_ts of the Nth most-recent article to establish the protection floor.
        # Scroll min_retention_count points ordered by published_ts DESC; the last one is T_protect.
        protected, _ = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=feed_filter,
            with_payload=['published_ts'],
            with_vectors=False,
            limit=min_retention_count,
            order_by=models.OrderBy(key='published_ts', direction=models.Direction.DESC),
        )
        if not protected:
            continue
        t_protect = protected[-1].payload.get('published_ts', 0)

        # effective_cutoff: only delete articles that are both beyond the protection floor
        # AND older than max_age_days. If the feed is low-volume and the Nth article is already
        # older than max_age_days, effective_cutoff = t_protect (keeps top N, drops older tail).
        effective_cutoff = min(cutoff, t_protect)

        delete_filter = Filter(must=[
            FieldCondition(key='feed_url', match=MatchValue(value=feed_url)),
            FieldCondition(key='published_ts', range=Range(lt=effective_cutoff)),
        ])
        count = qdrant.count(collection_name=collection_name, count_filter=delete_filter).count
        if count == 0:
            logging.info(f"Cull [{feed_url}]: nothing to delete")
            continue

        logging.warning(f"Culling {count} chunks for {feed_url} (effective_cutoff={effective_cutoff:.0f}, protect top {min_retention_count})")
        if not dry_run:
            qdrant.delete(collection_name=collection_name, points_selector=delete_filter)
        total_culled += count

    if total_culled:
        logging.warning(f"Cull complete: {total_culled} total chunks removed")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            search_start = end - int(chunk_size * 0.2)
            # Paragraph break first, then sentence ends: a chunk that stops at
            # a blank line holds whole paragraphs, where one stopping at the
            # nearest '. ' saws through the middle of one. '\n\n' was last
            # here and could never match anyway, since _extract_feed_content()
            # had already collapsed every blank line into a single space.
            for punct in ['\n\n', '. ', '! ', '? ']:
                idx = text.rfind(punct, search_start, end)
                if idx != -1:
                    end = idx + len(punct)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def get_embeddings(texts: list[str], client: OpenAI, model: str) -> list[list[float]]:
    """Get embeddings for a batch of texts"""
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def self_test() -> int:
    """Check the ledger rules against known cases.

    Same shape as tools/publish_scrub.py's --self-test, and for the same
    reason: this is the part where a wrong answer is expensive and silent. A
    ledger that stops matching does not fail -- it quietly re-fetches and
    re-embeds every article it ever saw.
    """
    import tempfile

    feed_a = 'https://news.example.com/atom?x=1'
    feed_b = 'https://blog.example.com/rss'
    retired = 'https://gone.example.com/feed'

    # A token of the shape that made the state file big in the first place
    long_token = 'CBMi' + 'Ab3Z' * 400
    old_shape = {
        feed_a: [long_token, 'https://news.example.com/story/1'],
        feed_b: ['tag:blog.example.com,2026:post-7'],
        retired: ['https://gone.example.com/post/1'],
        'https://odd.example.com/feed': [12345],  # a feed that handed out a number
    }
    live = {feed_a, feed_b, 'https://odd.example.com/feed'}

    full, converted, dropped = migrate_state(old_shape, live)
    again, converted_again, _ = migrate_state(full, live)
    targeted, _, targeted_dropped = migrate_state(old_shape, None)
    every_key = [k for keys in full.values() for k in keys]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / '.feeds_sync_state.json'
        write_state(path, {'processed': full, 'last_sync': '2026-11-11T04:00:00+00:00'})
        with open(path) as f:
            round_tripped = json.load(f)

    cases = [
        # Migration from the old raw-GUID shape
        ("every stored value ends up a digest",
         all(GUID_KEY_RE.match(k) for k in every_key), True),
        ("converted count covers the kept feeds only",
         converted, 4),
        ("a non-string value converts instead of raising",
         len(full['https://odd.example.com/feed']), 1),
        ("migration is idempotent",
         (again, converted_again), (full, 0)),
        ("an opaque-token feed shrinks by an order of magnitude",
         len(json.dumps(full[feed_a])) * 10 < len(json.dumps(old_shape[feed_a])), True),

        # A known GUID still dedups once the ledger holds digests
        ("a migrated raw GUID still matches",
         _guid_key(long_token) in set(full[feed_a]), True),
        ("so does a short one",
         _guid_key('tag:blog.example.com,2026:post-7') in set(full[feed_b]), True),
        ("an unseen GUID does not",
         _guid_key('https://news.example.com/story/2') in set(full[feed_a]), False),
        ("an entry with no id at all still gets a key",
         bool(GUID_KEY_RE.match(_guid_key(''))), True),

        # Pruning retired feeds, and only when the caller knows them all
        ("a retired feed is dropped on a full run",
         (dropped, retired in full), ([retired], False)),
        ("a --feeds run prunes nothing",
         (targeted_dropped, sorted(targeted)), ([], sorted(old_shape))),

        # The key health_check.py reads survives a write
        ("last_sync survives the state write",
         round_tripped.get('last_sync'), '2026-11-11T04:00:00+00:00'),
        ("and so does the ledger beside it",
         round_tripped.get('processed'), full),
    ]

    failures = 0
    for label, got, want in cases:
        if got != want:
            print(f"  FAIL {label}: {got!r} != {want!r}", file=sys.stderr)
            failures += 1
        else:
            print(f"  ok   {label}")

    print("self-test failed" if failures else "self-test passed")
    return 1 if failures else 0


def log_fatal(msg, exit_code=-1):
    logging.critical(f"Fatal Err: {msg}")
    send_alert(
        subject=f"[ALERT] feeds_ingest failed on {socket.gethostname()}",
        body=msg
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    sys.exit(run_guarded('feeds_ingest', main))
