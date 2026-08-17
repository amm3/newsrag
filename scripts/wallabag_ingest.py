#!/usr/bin/env python3
"""
wallabag_ingest.py - Ingest Wallabag articles into Qdrant

Fetches articles from Wallabag API, chunks them, generates embeddings,
and upserts to Qdrant with deduplication based on article ID. Each
article's highlights and notes (annotations) are indexed as their own
searchable points alongside the body chunks.

Annotations are (re)indexed whenever their article is reprocessed. If you
add/edit an annotation without changing the article body itself, it may not
be picked up by an incremental (`since`-based) sync until Wallabag bumps the
entry's updated_at. Use `--entries <id>` to force a refresh for a specific
article, `--annotated-only` to reprocess every article that currently has
annotations (cheap way to catch all of them without a full resync), or
`--full` to force a full re-sync.
"""

import sys
import os
import argparse
import logging
import time
import hashlib
import json
import re
import socket
import warnings
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from alert import send_alert
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, PayloadSchemaType
)

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'


def main():
    parser = argparse.ArgumentParser(description='Wallabag to Qdrant Ingestion')
    parser.add_argument("-v", action="store_true", default=False, help="Print extra info")
    parser.add_argument("-vv", action="store_true", default=False, help="Print (more) extra info")
    sync_mode = parser.add_mutually_exclusive_group()
    sync_mode.add_argument("--full", action="store_true", help="Full re-sync (ignore state)")
    sync_mode.add_argument("--entries", type=int, nargs='+', metavar='ID',
                           help="Reprocess specific Wallabag entry IDs")
    sync_mode.add_argument("--annotated-only", action="store_true",
                           help="Reprocess only entries that currently have annotations "
                                "(requires Wallabag >= 2.6.14; ignores --full/incremental state)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Qdrant")
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

    # Validate required config
    required_vars = ['WALLABAG_URL', 'WALLABAG_CLIENT_ID', 'WALLABAG_CLIENT_SECRET',
                     'WALLABAG_USERNAME', 'WALLABAG_PASSWORD', 'QDRANT_URL', 'OPENAI_API_KEY']
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        log_fatal(f"Missing required environment variables: {', '.join(missing)}")

    # Configuration from env
    chunk_size = int(os.environ.get('CHUNK_SIZE', 1000))
    chunk_overlap = int(os.environ.get('CHUNK_OVERLAP', 200))
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
    collection_name = os.environ.get('WALLABAG_COLLECTION', 'wallabag_articles')

    # State file location
    state_file = config_dir / '.wallabag_sync_state.json'

    # Initialize clients
    wallabag = WallabagClient(
        url=os.environ['WALLABAG_URL'],
        client_id=os.environ['WALLABAG_CLIENT_ID'],
        client_secret=os.environ['WALLABAG_CLIENT_SECRET'],
        username=os.environ['WALLABAG_USERNAME'],
        password=os.environ['WALLABAG_PASSWORD']
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Api key is used with an insecure connection')
        qdrant = QdrantClient(
            url=os.environ['QDRANT_URL'],
            api_key=os.environ.get('QDRANT_API_KEY')
        )

    openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    # Ensure collection exists
    if not args.dry_run:
        ensure_collection(qdrant, collection_name)

    # Fetch articles
    if args.entries:
        logging.info(f"Fetching {len(args.entries)} specific entries from Wallabag...")
        articles = []
        for entry_id in args.entries:
            try:
                article = wallabag.get_entry(entry_id)
                articles.append(article)
            except Exception as e:
                logging.error(f"Failed to fetch entry {entry_id}: {e}")
        logging.info(f"Retrieved {len(articles)} of {len(args.entries)} requested entries")
    elif args.annotated_only:
        _check_annotations_filter_support(wallabag)
        logging.info("Fetching entries with annotations from Wallabag...")
        articles = wallabag.get_entries(annotations=True)
        logging.info(f"Found {len(articles)} annotated articles to process")
    else:
        last_sync = None
        if not args.full and state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                last_sync = state.get('last_sync')
                logging.info(f"Resuming from last sync: {last_sync}")

        logging.info("Fetching articles from Wallabag...")
        articles = wallabag.get_entries(since=last_sync)
        logging.info(f"Found {len(articles)} articles to process")

    if not articles:
        logging.info("No new articles to process")
        return 0

    # Process articles
    total_chunks = 0
    total_annotations = 0
    for i, article in enumerate(articles, 1):
        try:
            chunks, annotation_count = process_article(
                article, openai_client, qdrant, collection_name,
                chunk_size, chunk_overlap, embedding_model, wallabag, args.dry_run
            )
            total_chunks += chunks
            total_annotations += annotation_count
            logging.info(
                f"[{i}/{len(articles)}] Processed: {article['title'][:50]}... "
                f"({chunks} chunks, {annotation_count} annotations)"
            )
        except Exception as e:
            logging.error(f"Failed to process article {article.get('id')}: {e}")

    # Save state (skip for targeted reprocessing)
    if not args.dry_run and not args.entries and not args.annotated_only:
        with open(state_file, 'w') as f:
            json.dump({'last_sync': datetime.now(timezone.utc).isoformat()}, f)

    logging.warning(
        f"Completed: {len(articles)} articles, {total_chunks} chunks, "
        f"{total_annotations} annotations indexed"
    )
    return 0


class WallabagClient:
    """Simple Wallabag API client"""

    def __init__(self, url, client_id, client_secret, username, password):
        self.url = url.rstrip('/')
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.token = None
        self.token_expires = 0

    def _get_token(self):
        """Get or refresh OAuth token"""
        if self.token and time.time() < self.token_expires:
            return self.token

        logging.debug("Requesting new OAuth token from Wallabag")
        resp = requests.post(f"{self.url}/oauth/v2/token", data={
            'grant_type': 'password',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'username': self.username,
            'password': self.password
        })
        resp.raise_for_status()
        data = resp.json()

        self.token = data['access_token']
        self.token_expires = time.time() + data.get('expires_in', 3600) - 60
        return self.token

    def get_entries(self, since=None, per_page=30, annotations=None):
        """Fetch all entries, optionally since a timestamp and/or filtered to
        those that currently have annotations"""
        token = self._get_token()
        headers = {'Authorization': f'Bearer {token}'}

        all_entries = []
        page = 1

        while True:
            params = {'perPage': per_page, 'page': page}
            if since:
                # Wallabag uses 'since' as Unix timestamp
                params['since'] = int(datetime.fromisoformat(since.replace('Z', '+00:00')).timestamp())
            if annotations is not None:
                params['annotations'] = 1 if annotations else 0

            logging.debug(f"Fetching page {page} from Wallabag API")
            resp = requests.get(f"{self.url}/api/entries.json",
                              headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = data.get('_embedded', {}).get('items', [])
            if not items:
                break

            all_entries.extend(items)
            logging.debug(f"Page {page}: retrieved {len(items)} entries")

            if page >= data.get('pages', 1):
                break
            page += 1

        return all_entries

    def get_entry(self, entry_id: int) -> dict:
        """Fetch a single entry by ID"""
        token = self._get_token()
        headers = {'Authorization': f'Bearer {token}'}
        resp = requests.get(f"{self.url}/api/entries/{entry_id}.json",
                            headers=headers)
        resp.raise_for_status()
        return resp.json()

    def get_annotations(self, entry_id: int) -> list[dict]:
        """Fetch annotations (highlights + notes) for a single entry"""
        token = self._get_token()
        headers = {'Authorization': f'Bearer {token}'}
        resp = requests.get(f"{self.url}/api/annotations/{entry_id}.json",
                            headers=headers)
        resp.raise_for_status()
        return resp.json().get('rows', [])

    def get_version(self) -> str:
        """Fetch the Wallabag server version (public endpoint, no auth needed)"""
        resp = requests.get(f"{self.url}/api/version.json")
        resp.raise_for_status()
        return resp.json()


MIN_ANNOTATIONS_FILTER_VERSION = (2, 6, 14)


def _check_annotations_filter_support(wallabag: 'WallabagClient'):
    """Best-effort warning if the server predates Wallabag's `annotations`
    entries filter (added in 2.6.14, wallabag/wallabag#8346). On an older
    server the filter is silently ignored and /api/entries.json?annotations=1
    returns the FULL unfiltered entry list instead of just annotated ones.
    """
    try:
        version = wallabag.get_version()
        match = re.match(r'(\d+)\.(\d+)\.(\d+)', version or '')
        if match and tuple(int(g) for g in match.groups()) < MIN_ANNOTATIONS_FILTER_VERSION:
            logging.warning(
                f"Wallabag server reports version {version}, older than 2.6.14 — the "
                f"'annotations' entries filter may not be supported. --annotated-only "
                f"could silently reprocess your ENTIRE library instead of just annotated "
                f"articles. Verify with: curl {wallabag.url}/api/entries.json?annotations=1"
            )
    except Exception as e:
        logging.debug(f"Could not verify Wallabag version for annotations-filter support: {e}")


def ensure_collection(client: QdrantClient, collection_name: str, dimensions: int = 1536):
    """Create collection if it doesn't exist"""
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=Distance.COSINE
            )
        )
        logging.info(f"Created collection: {collection_name}")
    else:
        logging.debug(f"Collection exists: {collection_name}")
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name='is_starred',
            field_schema=PayloadSchemaType.BOOL
        )
        logging.debug("Ensured payload index on is_starred")
    except Exception as e:
        logging.debug(f"Payload index on is_starred already exists or failed: {e}")
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name='chunk_type',
            field_schema=PayloadSchemaType.KEYWORD
        )
        logging.debug("Ensured payload index on chunk_type")
    except Exception as e:
        logging.debug(f"Payload index on chunk_type already exists or failed: {e}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
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
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]


def _article_payload_base(article: dict) -> dict:
    """Fields shared by every point (body chunk or annotation) for one article."""
    return {
        'article_id': article['id'],
        'title': article.get('title', 'Untitled'),
        'url': article.get('url', ''),
        'domain': article.get('domain_name', ''),
        'reading_time': article.get('reading_time', 0),
        'created_at': article.get('created_at', ''),
        'updated_at': article.get('updated_at', ''),
        'published_at': article.get('published_at', ''),
        'tags': [t['label'] for t in article.get('tags', [])],
        'published_by': article.get('published_by', []),
        'source': 'wallabag',
        'is_starred': bool(article.get('is_starred', 0)),
        'starred_at': article.get('starred_at'),
    }


def _build_annotation_items(article_id: int, annotations: list[dict]) -> list[tuple[str, str, dict]]:
    """Build (point_id, embed_text, extra_payload) tuples for one article's
    annotations, skipping any with neither a usable quote nor note.
    """
    items = []
    for idx, ann in enumerate(annotations):
        quote = (ann.get('quote') or '').strip()
        note = (ann.get('text') or '').strip()

        if not quote and not note:
            continue

        embed_text = f"{quote}\n\nNote: {note}" if quote and note else (quote or note)

        annotation_id = ann.get('id')
        if annotation_id is None:
            annotation_id = idx
        point_id = hashlib.md5(f"{article_id}_annotation_{annotation_id}".encode()).hexdigest()

        items.append((point_id, embed_text, {
            'chunk_type': 'annotation',
            'annotation_id': annotation_id,
            'quote': quote,
            'note': note,
            'text': embed_text,
        }))

    return items


def process_article(article: dict, openai_client: OpenAI,
                   qdrant: QdrantClient, collection_name: str,
                   chunk_size: int, chunk_overlap: int, embedding_model: str,
                   wallabag: 'WallabagClient', dry_run: bool = False) -> tuple[int, int]:
    """Process a single article: chunk body + annotations, embed, upsert.

    Returns (body_chunk_count, annotation_count).
    """

    article_id = article['id']
    title = article.get('title', 'Untitled')
    content = article.get('content', '')

    # Strip HTML tags
    clean_content = re.sub(r'<[^>]+>', '', content)
    clean_content = re.sub(r'\s+', ' ', clean_content).strip()

    # Fetch annotations (highlights + notes). Best-effort: a failure here
    # degrades to zero annotations rather than aborting article processing.
    try:
        annotations = wallabag.get_annotations(article_id)
    except Exception as e:
        logging.warning(f"Failed to fetch annotations for article {article_id}: {e}")
        annotations = []
    annotation_items = _build_annotation_items(article_id, annotations)

    if not clean_content and not annotation_items:
        logging.debug(f"Skipping article {article_id}: no content and no annotations")
        return 0, 0

    # Delete existing points for this article (for updates). If the body
    # came back empty this run (e.g. a transient extraction hiccup) but
    # annotations exist, only clear out stale annotation points rather than
    # wiping previously-indexed body chunks that are still good.
    if not dry_run:
        try:
            if clean_content:
                qdrant.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[FieldCondition(key='article_id', match=MatchValue(value=article_id))]
                    )
                )
            else:
                qdrant.delete(
                    collection_name=collection_name,
                    points_selector=Filter(must=[
                        FieldCondition(key='article_id', match=MatchValue(value=article_id)),
                        FieldCondition(key='chunk_type', match=MatchValue(value='annotation')),
                    ])
                )
        except Exception as e:
            logging.debug(f"Delete failed (may not exist): {e}")

    # Chunk the content
    chunks = chunk_text(f"{title}\n\n{clean_content}", chunk_size, chunk_overlap) if clean_content else []

    # Batch body-chunk and annotation texts into a single embeddings call
    annotation_texts = [t for _, t, _ in annotation_items]
    texts_to_embed = chunks + annotation_texts

    if not texts_to_embed:
        return 0, 0

    embeddings = get_embeddings(texts_to_embed, openai_client, embedding_model)
    chunk_embeddings = embeddings[:len(chunks)]
    annotation_embeddings = embeddings[len(chunks):]

    base_payload = _article_payload_base(article)

    # Create body chunk points
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        point_id = hashlib.md5(f"{article_id}_{i}".encode()).hexdigest()
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={**base_payload, 'chunk_type': 'body', 'chunk_index': i, 'text': chunk}
        ))

    # Create annotation points
    for (point_id, _, extra_payload), embedding in zip(annotation_items, annotation_embeddings):
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={**base_payload, **extra_payload}
        ))

    # Upsert to Qdrant
    if not dry_run:
        qdrant.upsert(collection_name=collection_name, points=points)

    return len(chunks), len(annotation_items)


def log_fatal(msg, exit_code=-1):
    logging.critical(f"Fatal Err: {msg}")
    send_alert(
        subject=f"[ALERT] wallabag_ingest failed on {socket.gethostname()}",
        body=msg
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
