#!/usr/bin/env python3
"""
feeds_ingest.py - Ingest RSS/Atom feed articles into Qdrant

Reads feed URLs from config/feeds.yaml, fetches new entries, optionally
retrieves full article content via Firecrawl or trafilatura, chunks and
embeds the text, then upserts to Qdrant with deduplication by entry GUID.
"""

import sys
import os
import argparse
import logging
import hashlib
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
import yaml
import trafilatura
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
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
    args = parser.parse_args()

    if args.vv:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.DEBUG)
    elif args.v:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.INFO)
    else:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.WARNING)

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

    with open(feeds_config_path) as f:
        feeds_config = yaml.safe_load(f)

    max_age_days = int(feeds_config.get('max_age_days', 0))

    all_feeds = feeds_config.get('feeds', [])
    if not all_feeds:
        log_fatal("No feeds defined in feeds config")

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
    processed_map = state.get('processed', {})

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

    for feed_def in target_feeds:
        feed_url = feed_def['url']
        feed_name = feed_def.get('name', feed_url)
        do_fetch = feed_def.get('fetch_content', True)

        logging.info(f"Fetching feed: {feed_name} ({feed_url})")

        try:
            parsed = feedparser.parse(feed_url)
        except Exception as e:
            logging.error(f"Failed to parse feed {feed_url}: {e}")
            continue

        if parsed.bozo and parsed.bozo_exception:
            logging.warning(f"Feed parse warning for {feed_url}: {parsed.bozo_exception}")

        entries = parsed.entries
        logging.debug(f"Feed has {len(entries)} entries")

        # Known GUIDs for this feed
        known_guids = set(processed_map.get(feed_url, []))
        new_guids = []

        for entry in entries:
            entry_id = _entry_id(entry)
            if entry_id in known_guids:
                logging.debug(f"Skipping known entry: {entry_id}")
                continue

            try:
                chunks = process_entry(
                    entry, feed_url, feed_name, do_fetch,
                    firecrawl_url, firecrawl_api_key, firecrawl_timeout,
                    openai_client, qdrant, collection_name,
                    chunk_size, chunk_overlap, embedding_model, args.dry_run
                )
                total_chunks += chunks
                total_entries += 1
                new_guids.append(entry_id)
                title = getattr(entry, 'title', entry_id)[:60]
                logging.info(f"  Indexed: {title} ({chunks} chunks)")
            except Exception as e:
                logging.error(f"Failed to process entry {entry_id}: {e}")

        # Update state for this feed
        if not args.dry_run:
            all_guids = list(known_guids | set(new_guids))
            processed_map[feed_url] = all_guids

        logging.info(f"Feed done: {feed_name} — {len(new_guids)} new entries")

    # Save state
    if not args.dry_run and not args.feeds:
        with open(state_file, 'w') as f:
            json.dump({'processed': processed_map}, f, indent=2)
    elif not args.dry_run and args.feeds:
        # Merge targeted feed updates into existing state
        if state_file.exists():
            with open(state_file) as f:
                full_state = json.load(f)
        else:
            full_state = {'processed': {}}
        full_state['processed'].update(processed_map)
        with open(state_file, 'w') as f:
            json.dump(full_state, f, indent=2)

    logging.warning(f"Completed: {total_entries} new entries, {total_chunks} chunks indexed")

    if max_age_days > 0:
        cull_old_entries(qdrant, collection_name, max_age_days, args.dry_run)

    return 0


def _entry_id(entry) -> str:
    """Get a stable identifier for a feed entry"""
    return getattr(entry, 'id', None) or getattr(entry, 'link', None) or ''


def process_entry(entry, feed_url: str, feed_name: str, do_fetch: bool,
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

    # Strip HTML
    text = re.sub(r'<[^>]+>', ' ', raw)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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
    # Payload index for efficient age-based culling
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name='published_ts',
            field_schema=PayloadSchemaType.FLOAT
        )
        logging.debug("Ensured payload index on published_ts")
    except Exception as e:
        logging.debug(f"Payload index on published_ts already exists or failed: {e}")


def cull_old_entries(qdrant: QdrantClient, collection_name: str, max_age_days: int, dry_run: bool = False):
    """Delete points older than max_age_days. Only affects points with published_ts set."""
    cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)
    old_filter = Filter(must=[FieldCondition(key='published_ts', range=Range(lt=cutoff))])
    count = qdrant.count(collection_name=collection_name, count_filter=old_filter).count
    if count == 0:
        logging.info(f"Cull: no chunks older than {max_age_days} days")
        return
    logging.warning(f"Culling {count} chunks older than {max_age_days} days")
    if not dry_run:
        qdrant.delete(collection_name=collection_name, points_selector=old_filter)


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
            for punct in ['. ', '! ', '? ', '\n\n']:
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


def log_fatal(msg, exit_code=-1):
    logging.critical(f"Fatal Err: {msg}")
    sys.exit(exit_code)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
