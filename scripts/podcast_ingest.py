#!/usr/bin/env python3
"""
podcast_ingest.py - Ingest podcast transcripts from NAS into Qdrant

Scans directories for .txt files alongside .mp3 files, chunks them,
generates embeddings, and upserts to Qdrant.
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
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'


def main():
    # Load .env early so env vars are available for argument defaults
    config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
    load_dotenv(config_dir / '.env')

    parser = argparse.ArgumentParser(description='Podcast Transcript to Qdrant Ingestion')
    parser.add_argument("--podcast-dir", default=os.environ.get('PODCAST_DIR'), required=not os.environ.get('PODCAST_DIR'), help="Root directory containing podcasts")
    parser.add_argument("-v", action="store_true", default=False, help="Print extra info")
    parser.add_argument("-vv", action="store_true", default=False, help="Print (more) extra info")
    sync_mode = parser.add_mutually_exclusive_group()
    sync_mode.add_argument("--full", action="store_true", help="Full re-sync (ignore state)")
    sync_mode.add_argument("--files", nargs='+', metavar='PATH',
                           help="Reprocess specific transcript file paths")
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
    chunk_size = int(os.environ.get('PODCAST_CHUNK_SIZE', os.environ.get('CHUNK_SIZE', 1500)))
    chunk_overlap = int(os.environ.get('PODCAST_CHUNK_OVERLAP', os.environ.get('CHUNK_OVERLAP', 300)))
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
    collection_name = os.environ.get('PODCAST_COLLECTION', 'podcast_transcripts')

    # State file location
    state_file = config_dir / '.podcast_sync_state.json'

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

    podcast_dir = Path(args.podcast_dir)
    if not podcast_dir.exists():
        log_fatal(f"Podcast directory does not exist: {podcast_dir}")

    if args.files:
        # Reprocess specific files by path
        new_files = []
        for fp in args.files:
            tp = Path(fp).resolve()
            if not tp.exists():
                logging.error(f"File not found: {fp}")
                continue
            audio = find_audio_for(tp)
            if audio is None:
                logging.error(f"No audio counterpart for: {fp}")
                continue
            new_files.append((tp, audio))
        logging.info(f"Reprocessing {len(new_files)} of {len(args.files)} requested files")
    else:
        # Discover transcript files
        transcript_files = find_transcripts(podcast_dir)

        # Filter to new files only (or all files if --full)
        if args.full:
            new_files = transcript_files
        else:
            new_files = [(t, a) for t, a in transcript_files if str(t) not in processed_files]

        logging.info(f"Found {len(transcript_files)} transcripts, {len(new_files)} to process")

    if not new_files:
        logging.info("No new transcripts to process")
        return 0

    # Process files
    total_chunks = 0
    newly_processed = []

    for i, (transcript_path, audio_path) in enumerate(new_files, 1):
        try:
            chunks = process_transcript(
                transcript_path, audio_path, podcast_dir, openai_client,
                qdrant, collection_name,
                chunk_size, chunk_overlap, embedding_model, args.dry_run
            )
            total_chunks += chunks
            newly_processed.append(str(transcript_path))
            logging.info(f"[{i}/{len(new_files)}] Processed: {transcript_path.name} ({chunks} chunks)")
        except Exception as e:
            logging.error(f"Failed to process {transcript_path}: {e}")

    # Save state (skip for targeted reprocessing)
    if not args.dry_run and not args.files:
        all_processed = processed_files | set(newly_processed)
        with open(state_file, 'w') as f:
            json.dump({
                'processed_files': list(all_processed),
                'last_sync': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

    logging.warning(f"Completed: {len(newly_processed)} files, {total_chunks} chunks indexed")
    return 0


AUDIO_EXTENSIONS = ['.mp3', '.m4a', '.wav', '.ogg', '.flac']


def find_audio_for(txt_path: Path) -> Path | None:
    """Find the audio file corresponding to a transcript .txt file."""
    for ext in AUDIO_EXTENSIONS:
        candidate = txt_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def find_transcripts(root_dir: Path) -> list[tuple[Path, Path]]:
    """Find all .txt files that have corresponding audio files.

    Returns list of (transcript_path, audio_path) tuples.
    """
    transcripts = []

    for txt_file in root_dir.rglob("*.txt"):
        audio_file = find_audio_for(txt_file)
        if audio_file is not None:
            transcripts.append((txt_file, audio_file))
        else:
            logging.debug(f"Skipping {txt_file}: no audio counterpart")

    return sorted(transcripts, key=lambda pair: pair[0])


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


def process_transcript(transcript_path: Path, audio_path: Path, root_dir: Path,
                       openai_client: OpenAI, qdrant: QdrantClient,
                       collection_name: str,
                       chunk_size: int, chunk_overlap: int, embedding_model: str,
                       dry_run: bool = False) -> int:
    """Process a single transcript file"""

    # Generate stable ID from file path relative to root
    relative_path = transcript_path.relative_to(root_dir)
    file_id = hashlib.md5(str(relative_path).encode()).hexdigest()

    # Extract metadata from path
    show_name = transcript_path.parent.name
    episode_name = transcript_path.stem

    # Read content
    try:
        content = transcript_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        logging.error(f"Failed to read {transcript_path}: {e}")
        return 0

    if not content.strip():
        logging.debug(f"Skipping {transcript_path}: empty content")
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
    mtime = datetime.fromtimestamp(transcript_path.stat().st_mtime).isoformat()

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
                    'show_name': show_name,
                    'episode_name': episode_name,
                    'file_path': str(relative_path),
                    'audio_file': str(audio_path.relative_to(root_dir)),
                    'text': chunk,
                    'modified_at': mtime,
                    'source': 'podcast_transcript'
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
