#!/usr/bin/env python3
"""
wallabag_ingest.py - Ingest Wallabag articles into Qdrant

Fetches articles from Wallabag API, chunks them, generates embeddings,
and upserts to Qdrant with deduplication based on article ID.
"""

import sys
import os
import argparse
import logging
import time
import hashlib
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
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
    for i, article in enumerate(articles, 1):
        try:
            chunks = process_article(
                article, openai_client, qdrant, collection_name,
                chunk_size, chunk_overlap, embedding_model, args.dry_run
            )
            total_chunks += chunks
            logging.info(f"[{i}/{len(articles)}] Processed: {article['title'][:50]}... ({chunks} chunks)")
        except Exception as e:
            logging.error(f"Failed to process article {article.get('id')}: {e}")

    # Save state (skip for targeted reprocessing)
    if not args.dry_run and not args.entries:
        with open(state_file, 'w') as f:
            json.dump({'last_sync': datetime.now(timezone.utc).isoformat()}, f)

    logging.warning(f"Completed: {len(articles)} articles, {total_chunks} chunks indexed")
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

    def get_entries(self, since=None, per_page=30):
        """Fetch all entries, optionally since a timestamp"""
        token = self._get_token()
        headers = {'Authorization': f'Bearer {token}'}

        all_entries = []
        page = 1

        while True:
            params = {'perPage': per_page, 'page': page}
            if since:
                # Wallabag uses 'since' as Unix timestamp
                params['since'] = int(datetime.fromisoformat(since.replace('Z', '+00:00')).timestamp())

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


def process_article(article: dict, openai_client: OpenAI,
                   qdrant: QdrantClient, collection_name: str,
                   chunk_size: int, chunk_overlap: int, embedding_model: str,
                   dry_run: bool = False) -> int:
    """Process a single article: chunk, embed, upsert"""

    article_id = article['id']
    title = article.get('title', 'Untitled')
    content = article.get('content', '')

    # Strip HTML tags
    clean_content = re.sub(r'<[^>]+>', '', content)
    clean_content = re.sub(r'\s+', ' ', clean_content).strip()

    if not clean_content:
        logging.debug(f"Skipping article {article_id}: no content")
        return 0

    # Delete existing chunks for this article (for updates)
    if not dry_run:
        try:
            qdrant.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key='article_id', match=MatchValue(value=article_id))]
                )
            )
        except Exception as e:
            logging.debug(f"Delete failed (may not exist): {e}")

    # Chunk the content
    chunks = chunk_text(f"{title}\n\n{clean_content}", chunk_size, chunk_overlap)

    if not chunks:
        return 0

    # Generate embeddings
    embeddings = get_embeddings(chunks, openai_client, embedding_model)

    # Create points
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = hashlib.md5(f"{article_id}_{i}".encode()).hexdigest()

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                'article_id': article_id,
                'chunk_index': i,
                'title': title,
                'text': chunk,
                'url': article.get('url', ''),
                'domain': article.get('domain_name', ''),
                'reading_time': article.get('reading_time', 0),
                'created_at': article.get('created_at', ''),
                'tags': [t['label'] for t in article.get('tags', [])],
                'source': 'wallabag'
            }
        ))

    # Upsert to Qdrant
    if not dry_run:
        qdrant.upsert(collection_name=collection_name, points=points)

    return len(points)


def log_fatal(msg, exit_code=-1):
    logging.critical(f"Fatal Err: {msg}")
    sys.exit(exit_code)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
