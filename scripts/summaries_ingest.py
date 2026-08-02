#!/usr/bin/env python3
"""
summaries_ingest.py - Load AI-generated summaries into Qdrant (Phase 2)

Scans a podcast directory and/or a papers directory for *.ai-summary.md
files (written by summarize.py, Phase 1 — or hand-edited afterwards),
chunks them, generates embeddings, and upserts them into a single
'summaries' collection tagged with source_type ('podcast' or 'paper').

Re-running after a summary file has been hand-edited on disk picks up the
change automatically via mtime-based state tracking, same as every other
loader in this project.
"""

import sys
import os
import re
import time
import argparse
import logging
import hashlib
import json
import socket
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from alert import send_alert
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'


def main():
    # Load .env early so env vars are available for argument defaults
    config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
    load_dotenv(config_dir / '.env')

    parser = argparse.ArgumentParser(description='AI Summaries to Qdrant Ingestion')
    parser.add_argument("--podcast-dir", default=os.environ.get('PODCAST_DIR'), help="Root directory containing podcasts (scanned for .ai-summary.md files)")
    parser.add_argument("--papers-dir", default=os.environ.get('PAPERS_DIR'), help="Root directory containing papers/documents (scanned for .ai-summary.md files)")
    parser.add_argument("--collection", default=None, help="Qdrant collection name (overrides SUMMARIES_COLLECTION env var, default: 'summaries')")
    parser.add_argument("-v", action="store_true", default=False, help="Print extra info")
    parser.add_argument("-vv", action="store_true", default=False, help="Print (more) extra info")
    parser.add_argument("--full", action="store_true", help="Full re-sync (ignore state)")
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
    chunk_size = int(os.environ.get('SUMMARY_CHUNK_SIZE', 2000))
    chunk_overlap = int(os.environ.get('SUMMARY_CHUNK_OVERLAP', 400))
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
    collection_name = args.collection or os.environ.get('SUMMARIES_COLLECTION', 'summaries')

    # State file (single file — this loader always targets one 'summaries' collection)
    state_file = config_dir / '.summaries_sync_state.json'

    # Initialize clients
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Api key is used with an insecure connection')
        qdrant = QdrantClient(
            url=os.environ['QDRANT_URL'],
            api_key=os.environ.get('QDRANT_API_KEY')
        )

    openai_client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        max_retries=int(os.environ.get('OPENAI_MAX_RETRIES', 5)),
    )

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

    # Resolve the root directories to scan
    roots: list[tuple[Path, str]] = []
    if args.podcast_dir:
        p = Path(args.podcast_dir)
        if p.exists():
            roots.append((p, 'podcast'))
        else:
            logging.warning(f"Podcast directory does not exist, skipping: {p}")
    if args.papers_dir:
        p = Path(args.papers_dir)
        if p.exists():
            roots.append((p, 'paper'))
        else:
            logging.warning(f"Papers directory does not exist, skipping: {p}")

    if not roots:
        log_fatal("No valid --podcast-dir or --papers-dir provided (or PODCAST_DIR/PAPERS_DIR env vars)")

    # Find summary files across all roots
    summary_files: list[tuple[Path, Path, str]] = []
    for root_dir, source_type in roots:
        for f in find_summaries(root_dir):
            summary_files.append((f, root_dir, source_type))

    # Filter to new/modified files only (or all files if --full)
    if args.full:
        new_files = summary_files
    else:
        new_files = [
            (f, root, source_type) for (f, root, source_type) in summary_files
            if int(f.stat().st_mtime) != file_mtimes.get(str(f))
        ]

    logging.info(f"Found {len(summary_files)} summaries, {len(new_files)} to process")

    if not new_files:
        logging.info("No new summaries to process")
        return 0

    # Process files
    total_chunks = 0
    newly_processed = []

    for i, (summary_path, root_dir, source_type) in enumerate(new_files, 1):
        try:
            chunks = process_summary(
                summary_path, root_dir, source_type, openai_client,
                qdrant, collection_name,
                chunk_size, chunk_overlap, embedding_model, args.dry_run
            )
            total_chunks += chunks
            newly_processed.append(str(summary_path))
            logging.info(f"[{i}/{len(new_files)}] Processed: {summary_path.name} ({chunks} chunks)")
        except Exception as e:
            logging.error(f"Failed to process {summary_path}: {e}")

    # Save state
    if not args.dry_run:
        file_mtimes.update({p: int(Path(p).stat().st_mtime) for p in newly_processed})
        with open(state_file, 'w') as f:
            json.dump({
                'file_mtimes': file_mtimes,
                'last_sync': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

    logging.warning(f"Completed: {len(newly_processed)} files, {total_chunks} chunks indexed")
    return 0


def find_summaries(root_dir: Path) -> list[Path]:
    """Find all .ai-summary.md files under root_dir."""
    return sorted(root_dir.rglob("*.ai-summary.md"))


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


def parse_header(content: str) -> tuple[dict, str]:
    """
    Parse optional key: value metadata from the top of a file.
    Parsing stops at the first blank line or non-matching line.
    'tags' values are split by comma, lowercased, and whitespace-stripped.
    Returns (metadata dict, remaining content with header stripped).
    """
    metadata = {}
    lines = content.split('\n')
    end = 0
    for line in lines:
        if not line.strip():
            end += 1  # consume the blank separator line
            break
        m = re.match(r'^(\w[\w\s]*?)\s*:\s*(.+)$', line)
        if not m:
            break
        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        metadata[key] = [t.strip().lower() for t in value.split(',')] if key == 'tags' else value
        end += 1
    return metadata, '\n'.join(lines[end:])


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
            for punct in ['. ', '! ', '? ', '\n\n', '\n']:
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


def process_summary(summary_path: Path, root_dir: Path, source_type: str,
                     openai_client: OpenAI, qdrant: QdrantClient,
                     collection_name: str,
                     chunk_size: int, chunk_overlap: int, embedding_model: str,
                     dry_run: bool = False) -> int:
    """Process a single .ai-summary.md file"""

    # Generate stable ID from source_type + path relative to root
    # (prefixed with source_type since podcast-dir and papers-dir are
    # scanned into the same collection and could otherwise collide)
    relative_path = summary_path.relative_to(root_dir)
    file_id = hashlib.md5(f"{source_type}:{relative_path}".encode()).hexdigest()

    # Recover the original episode/document name: <name>.ai-summary.md ->
    # stem strips .md, stem again strips .ai-summary
    base_name = Path(summary_path.stem).stem

    if source_type == 'podcast':
        identity = {'show_name': summary_path.parent.name, 'episode_name': base_name}
    else:
        identity = {'document_name': base_name}

    # Read content
    try:
        content = summary_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        logging.error(f"Failed to read {summary_path}: {e}")
        return 0

    if not content.strip():
        logging.debug(f"Skipping {summary_path}: empty content")
        return 0

    # Parse and strip metadata header
    header_meta, body = parse_header(content)

    # Delete existing chunks for this file (for re-processing)
    if not dry_run:
        try:
            qdrant.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key='file_id', match=MatchValue(value=file_id))]
                )
            )
        except Exception as e:
            logging.debug(f"Delete failed (may not exist): {e}")

    # Chunk the content
    chunks = chunk_text(body, chunk_size, chunk_overlap)

    if not chunks:
        return 0

    # Get file modification time
    mtime = datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat()

    # Generate embeddings in batches
    batch_size = 100
    embedding_batch_delay = float(os.environ.get('EMBEDDING_BATCH_DELAY', 0))
    points = []

    for batch_start in range(0, len(chunks), batch_size):
        if batch_start > 0 and embedding_batch_delay > 0:
            time.sleep(embedding_batch_delay)
        batch_chunks = chunks[batch_start:batch_start + batch_size]
        batch_embeddings = get_embeddings(batch_chunks, openai_client, embedding_model)

        for i, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
            chunk_idx = batch_start + i
            point_id = hashlib.md5(f"{file_id}_{chunk_idx}".encode()).hexdigest()

            payload = {
                'file_id': file_id,
                'chunk_index': chunk_idx,
                'source_type': source_type,
                'summary_file': str(relative_path),
                'text': chunk,
                'modified_at': mtime,
                'source': 'summary',
            }
            payload.update(identity)
            payload.update({k: v for k, v in header_meta.items() if k != 'tags'})
            if 'tags' in header_meta:
                payload['tags'] = header_meta['tags']
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    # Upsert to Qdrant in batches to stay under payload size limit
    qdrant_batch_size = int(os.environ.get('QDRANT_UPSERT_BATCH_SIZE', 500))
    if not dry_run:
        for b in range(0, len(points), qdrant_batch_size):
            qdrant.upsert(collection_name=collection_name, points=points[b:b + qdrant_batch_size])

    return len(points)


def log_fatal(msg, exit_code=-1):
    logging.critical(f"Fatal Err: {msg}")
    send_alert(
        subject=f"[ALERT] summaries_ingest failed on {socket.gethostname()}",
        body=msg
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
