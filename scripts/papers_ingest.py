#!/usr/bin/env python3
"""
papers_ingest.py - Ingest papers/documents from a folder into Qdrant

Scans a directory for .md and .txt files (preferring converted text over
originals), chunks them, generates embeddings, and upserts to Qdrant.

File selection logic:
  - For each unique (directory, stem), prefer .md > .txt
  - If an original (.pdf, .docx, etc.) exists alongside .md or .txt,
    only the text version is processed
  - Originals without a .md/.txt conversion are skipped
"""

import sys
import os
import argparse
import logging
import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
TEXT_EXTENSIONS = {'.md', '.txt'}


def main():
    # Load .env early so env vars are available for argument defaults
    config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
    load_dotenv(config_dir / '.env')

    parser = argparse.ArgumentParser(description='Papers/Documents to Qdrant Ingestion')
    parser.add_argument("--papers-dir", default=os.environ.get('PAPERS_DIR'), required=not os.environ.get('PAPERS_DIR'), help="Root directory containing papers/documents")
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
    chunk_size = int(os.environ.get('PAPERS_CHUNK_SIZE', os.environ.get('CHUNK_SIZE', 2000)))
    chunk_overlap = int(os.environ.get('PAPERS_CHUNK_OVERLAP', os.environ.get('CHUNK_OVERLAP', 400)))
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
    collection_name = os.environ.get('PAPERS_COLLECTION', 'papers')

    # State file location
    state_file = config_dir / '.papers_sync_state.json'

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
    processed_files = set()
    if not args.full and state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            processed_files = set(state.get('processed_files', []))
            logging.info(f"Loaded state: {len(processed_files)} files already processed")

    # Find document files
    papers_dir = Path(args.papers_dir)
    if not papers_dir.exists():
        log_fatal(f"Papers directory does not exist: {papers_dir}")

    document_files = find_documents(papers_dir)

    # Filter to new files only (or all files if --full)
    if args.full:
        new_files = document_files
    else:
        new_files = [f for f in document_files if str(f) not in processed_files]

    logging.info(f"Found {len(document_files)} documents, {len(new_files)} to process")

    if not new_files:
        logging.info("No new documents to process")
        return 0

    # Process files
    total_chunks = 0
    newly_processed = []

    for i, doc_path in enumerate(new_files, 1):
        try:
            chunks = process_document(
                doc_path, papers_dir, openai_client,
                qdrant, collection_name,
                chunk_size, chunk_overlap, embedding_model, args.dry_run
            )
            total_chunks += chunks
            newly_processed.append(str(doc_path))
            logging.info(f"[{i}/{len(new_files)}] Processed: {doc_path.name} ({chunks} chunks)")
        except Exception as e:
            logging.error(f"Failed to process {doc_path}: {e}")

    # Save state
    if not args.dry_run:
        all_processed = processed_files | set(newly_processed)
        with open(state_file, 'w') as f:
            json.dump({
                'processed_files': list(all_processed),
                'last_sync': datetime.utcnow().isoformat()
            }, f, indent=2)

    logging.warning(f"Completed: {len(newly_processed)} files, {total_chunks} chunks indexed")
    return 0


def find_documents(root_dir: Path) -> list[Path]:
    """
    Find processable documents, preferring .md over .txt over originals.

    For each unique (directory, stem) combination:
      - If .md exists, use it
      - Else if .txt exists, use it
      - Else skip (unconverted original)
    """
    # Build map: (parent_dir, stem) -> {suffix_lower: actual_path}
    stem_map: dict[tuple[Path, str], dict[str, Path]] = {}

    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith('.'):
            continue
        key = (file_path.parent, file_path.stem)
        stem_map.setdefault(key, {})[file_path.suffix.lower()] = file_path

    # Select best text representation for each stem
    selected = []

    for (parent, stem), ext_paths in stem_map.items():
        if '.md' in ext_paths:
            selected.append(ext_paths['.md'])
        elif '.txt' in ext_paths:
            selected.append(ext_paths['.txt'])
        else:
            logging.debug(f"Skipping {parent / stem}: no .md or .txt version")

    return sorted(selected)


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


def process_document(document_path: Path, root_dir: Path,
                     openai_client: OpenAI, qdrant: QdrantClient,
                     collection_name: str,
                     chunk_size: int, chunk_overlap: int, embedding_model: str,
                     dry_run: bool = False) -> int:
    """Process a single document file"""

    # Generate stable ID from file path relative to root
    relative_path = document_path.relative_to(root_dir)
    file_id = hashlib.md5(str(relative_path).encode()).hexdigest()

    # Extract metadata
    document_name = document_path.stem

    # Check if an original (non-.md/.txt) file exists with the same stem
    original_file = None
    for sibling in document_path.parent.iterdir():
        if (sibling.stem == document_path.stem
                and sibling.suffix.lower() not in TEXT_EXTENSIONS
                and sibling.is_file()):
            original_file = str(sibling.relative_to(root_dir))
            break

    # Read content
    try:
        content = document_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        logging.error(f"Failed to read {document_path}: {e}")
        return 0

    if not content.strip():
        logging.debug(f"Skipping {document_path}: empty content")
        return 0

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
    chunks = chunk_text(content, chunk_size, chunk_overlap)

    if not chunks:
        return 0

    # Get file modification time
    mtime = datetime.fromtimestamp(document_path.stat().st_mtime).isoformat()

    # Generate embeddings in batches
    batch_size = 100
    points = []

    for batch_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_start:batch_start + batch_size]
        batch_embeddings = get_embeddings(batch_chunks, openai_client, embedding_model)

        for i, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
            chunk_idx = batch_start + i
            point_id = hashlib.md5(f"{file_id}_{chunk_idx}".encode()).hexdigest()

            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'file_id': file_id,
                    'chunk_index': chunk_idx,
                    'document_name': document_name,
                    'file_path': str(relative_path),
                    'original_file': original_file,
                    'text': chunk,
                    'modified_at': mtime,
                    'source': 'paper'
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
