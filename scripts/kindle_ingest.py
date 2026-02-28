#!/usr/bin/env python3
"""
kindle_ingest.py - Ingest Kindle highlights from Bookcision JSON exports into Qdrant

Scans a directory for Bookcision JSON files, embeds each highlight,
and upserts to Qdrant. Each highlight is stored as its own point (no chunking).
Point IDs are deterministic (hash of ASIN + location), so re-runs are idempotent.
"""

import sys
import os
import argparse
import logging
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'


def main():
    # Load .env early so env vars are available for argument defaults
    config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
    load_dotenv(config_dir / '.env')

    parser = argparse.ArgumentParser(description='Kindle Highlights to Qdrant Ingestion')
    parser.add_argument("--kindle-dir", default=os.environ.get('KINDLE_JSON_DIR'),
                        required=not os.environ.get('KINDLE_JSON_DIR'),
                        help="Directory containing Bookcision JSON export files")
    parser.add_argument("-v", action="store_true", default=False, help="Print extra info")
    parser.add_argument("-vv", action="store_true", default=False, help="Print (more) extra info")
    sync_mode = parser.add_mutually_exclusive_group()
    sync_mode.add_argument("--full", action="store_true", help="Full re-sync (ignore state)")
    sync_mode.add_argument("--files", nargs='+', metavar='PATH',
                           help="Reprocess specific JSON file paths")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Qdrant")
    args = parser.parse_args()

    if args.vv:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.DEBUG)
    elif args.v:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.INFO)
    else:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.WARNING)

    # Validate required config
    required_vars = ['QDRANT_URL', 'OPENAI_API_KEY']
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        log_fatal(f"Missing required environment variables: {', '.join(missing)}")

    # Configuration from env
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
    collection_name = os.environ.get('KINDLE_COLLECTION', 'kindle_highlights')

    # State file location
    state_file = config_dir / '.kindle_sync_state.json'

    # Initialize clients
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

    # Load state
    file_mtimes = {}
    if not args.full and state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            file_mtimes = state.get('file_mtimes', {})
            logging.info(f"Loaded state: {len(file_mtimes)} files tracked")

    kindle_dir = Path(args.kindle_dir)
    if not kindle_dir.exists():
        log_fatal(f"Kindle directory does not exist: {kindle_dir}")

    if args.files:
        # Reprocess specific files by path
        new_files = []
        for fp in args.files:
            p = Path(fp).resolve()
            if not p.exists():
                logging.error(f"File not found: {fp}")
                continue
            if p.suffix.lower() != '.json' or p.name.startswith("._"):
                logging.error(f"Not a JSON file: {fp}")
                continue
            new_files.append(p)
        logging.info(f"Reprocessing {len(new_files)} of {len(args.files)} requested files")
    else:
        # Discover all JSON files in the kindle directory
        all_files = sorted(f for f in kindle_dir.glob("*.json") if not f.name.startswith("._"))

        if args.full:
            new_files = all_files
        else:
            new_files = [f for f in all_files
                         if int(f.stat().st_mtime) != file_mtimes.get(str(f))]

        logging.info(f"Found {len(all_files)} JSON files, {len(new_files)} to process")

    if not new_files:
        logging.info("No new Kindle export files to process")
        return 0

    # Process files
    total_highlights = 0
    newly_processed = []

    for i, json_path in enumerate(new_files, 1):
        try:
            count = process_book_file(
                json_path, openai_client, qdrant, collection_name, embedding_model, args.dry_run
            )
            total_highlights += count
            newly_processed.append(str(json_path))
            logging.info(f"[{i}/{len(new_files)}] Processed: {json_path.name} ({count} highlights)")
        except Exception as e:
            logging.error(f"Failed to process {json_path}: {e}")

    # Save state (skip for targeted reprocessing or dry-run)
    if not args.dry_run and not args.files:
        file_mtimes.update({p: int(Path(p).stat().st_mtime) for p in newly_processed})
        with open(state_file, 'w') as f:
            json.dump({
                'file_mtimes': file_mtimes,
                'last_sync': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

    logging.warning(f"Completed: {len(newly_processed)} books, {total_highlights} highlights indexed")
    return 0


def process_book_file(json_path: Path, openai_client: OpenAI, qdrant: QdrantClient,
                      collection_name: str, embedding_model: str, dry_run: bool = False) -> int:
    """Parse a Bookcision JSON export and upsert each highlight as a Qdrant point."""

    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read/parse {json_path}: {e}")
        return 0

    # Validate required top-level fields
    for key in ('asin', 'title', 'highlights'):
        if key not in data:
            logging.error(f"Skipping {json_path.name}: missing required field '{key}'")
            return 0

    asin = data['asin']
    title = data['title']
    authors = data.get('authors', '')
    highlights = data['highlights']

    if not highlights:
        logging.warning(f"Skipping {json_path.name}: no highlights")
        return 0

    logging.info(f"Processing '{title}' ({len(highlights)} highlights)")

    # Build (embed_text, payload) pairs, skipping empties
    items = []
    for h in highlights:
        text = (h.get('text') or '').strip()
        note = (h.get('note') or '').strip() or None
        is_note_only = h.get('isNoteOnly', False)
        location = h.get('location', {})
        location_value = location.get('value')
        location_url = location.get('url', '')

        if location_value is None:
            logging.debug(f"Skipping highlight with no location in '{title}'")
            continue

        # Determine text to embed
        if is_note_only or not text:
            if not note:
                logging.debug(f"Skipping empty highlight at location {location_value} in '{title}'")
                continue
            embed_text = note
        elif note:
            embed_text = f"{text}\n\nNote: {note}"
        else:
            embed_text = text

        # Deterministic point ID from ASIN + location
        point_id = hashlib.md5(f"{asin}:{location_value}".encode()).hexdigest()

        payload = {
            'source': 'kindle',
            'book_title': title,
            'authors': authors,
            'asin': asin,
            'highlight_text': text,
            'note': note,
            'location_value': location_value,
            'location_url': location_url,
            'file_name': json_path.name,
            'text': embed_text,
        }
        items.append((point_id, embed_text, payload))

    if not items:
        logging.warning(f"No valid highlights found in {json_path.name}")
        return 0

    # Embed and upsert in batches of 100
    batch_size = 100
    points = []

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        texts = [embed_text for _, embed_text, _ in batch]
        embeddings = get_embeddings(texts, openai_client, embedding_model)

        for (point_id, _, payload), embedding in zip(batch, embeddings):
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    if not dry_run:
        qdrant.upsert(collection_name=collection_name, points=points)

    return len(points)


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


def get_embeddings(texts: list[str], client: OpenAI, model: str) -> list[list[float]]:
    """Get embeddings for a batch of texts"""
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]


def log_fatal(msg, exit_code=-1):
    logging.critical(f"Fatal Err: {msg}")
    sys.exit(exit_code)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
