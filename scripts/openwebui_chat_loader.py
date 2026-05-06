#!/usr/bin/env python3
"""
openwebui_chat_loader.py - Ingest OpenWebUI chat history into Qdrant

Fetches chat history from the OpenWebUI REST API, indexes each message as a
separate Qdrant point with the chat UUID preserved in the payload. The chat_id
allows a retrieval tool to fetch full chat content from /api/v1/chats/{chat_id}
at query time.

Deduplication: point IDs are deterministic (hash of chat_id + message_index),
so re-runs are idempotent. Re-processing a chat deletes its existing points
before upserting the new set.

Incremental sync: state file tracks last_sync Unix timestamp; only chats
updated since then are re-indexed on subsequent runs.
"""

import sys
import os
import argparse
import logging
import hashlib
import json
import socket
import warnings
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, PayloadSchemaType
)

from alert import send_alert

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'

# Chunk size for message content — mirrors the pattern in all other loaders.
# 1000 chars ÷ 2 chars/token (worst-case pure code) = 500 tokens, well under
# text-embedding-3-small's 8192-token per-input limit.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def main():
    parser = argparse.ArgumentParser(description='OpenWebUI Chat History to Qdrant Ingestion')
    parser.add_argument("-v", action="store_true", default=False, help="Print extra info")
    parser.add_argument("-vv", action="store_true", default=False, help="Print (more) extra info")
    sync_mode = parser.add_mutually_exclusive_group()
    sync_mode.add_argument("--full", action="store_true", help="Full re-sync (ignore state)")
    sync_mode.add_argument("--chats", type=str, nargs='+', metavar='UUID',
                           help="Reprocess specific chat UUIDs only")
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
    required_vars = ['OPENWEBUI_URL', 'OPENWEBUI_API_KEY', 'QDRANT_URL', 'OPENAI_API_KEY']
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        log_fatal(f"Missing required environment variables: {', '.join(missing)}")

    # Configuration from env
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
    collection_name = os.environ.get('OPENWEBUI_COLLECTION', 'openwebui_chats')
    chunk_size = int(os.environ.get('OPENWEBUI_CHUNK_SIZE', DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(os.environ.get('OPENWEBUI_CHUNK_OVERLAP', DEFAULT_CHUNK_OVERLAP))

    # State file location
    state_file = config_dir / '.openwebui_sync_state.json'

    # Initialize clients
    owui = OpenWebUIClient(
        url=os.environ['OPENWEBUI_URL'],
        api_key=os.environ['OPENWEBUI_API_KEY']
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

    # Fetch chats to process
    if args.chats:
        logging.info(f"Fetching {len(args.chats)} specific chats from OpenWebUI...")
        chats = []
        for chat_id in args.chats:
            try:
                chat = owui.get_chat(chat_id)
                chats.append(chat)
            except requests.HTTPError as e:
                if e.response.status_code in (401, 403):
                    log_fatal(f"Authentication failed fetching chat {chat_id} — check OPENWEBUI_API_KEY: {e}")
                logging.error(f"Failed to fetch chat {chat_id}: {e}")
        logging.info(f"Retrieved {len(chats)} of {len(args.chats)} requested chats")
    else:
        last_sync_ts = None
        if not args.full and state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                last_sync_iso = state.get('last_sync')
                if last_sync_iso:
                    last_sync_ts = datetime.fromisoformat(last_sync_iso).timestamp()
                    logging.info(f"Resuming from last sync: {last_sync_iso}")

        logging.info("Fetching chat list from OpenWebUI...")
        try:
            all_chats_meta = owui.list_chats(since_ts=last_sync_ts)
        except requests.HTTPError as e:
            if e.response.status_code in (401, 403):
                log_fatal(f"Authentication failed listing chats — check OPENWEBUI_API_KEY: {e}")
            log_fatal(f"Failed to fetch chat list from OpenWebUI: {e}")

        logging.info(f"Found {len(all_chats_meta)} chats to process")

        if not all_chats_meta:
            logging.info("No new or updated chats to process")
            return 0

        # Fetch full chat details
        chats = []
        for meta in all_chats_meta:
            try:
                chat = owui.get_chat(meta['id'])
                chats.append(chat)
            except requests.HTTPError as e:
                if e.response.status_code in (401, 403):
                    log_fatal(f"Authentication failed fetching chat {meta['id']} — check OPENWEBUI_API_KEY: {e}")
                logging.error(f"Failed to fetch chat {meta['id']}: {e}")

    if not chats:
        logging.info("No chats to index")
        return 0

    # Process chats
    total_points = 0
    for i, chat in enumerate(chats, 1):
        chat_id = chat.get('id', 'unknown')
        title = chat.get('title', 'Untitled')
        try:
            count = process_chat(
                chat, openai_client, qdrant, collection_name,
                embedding_model, chunk_size, chunk_overlap, args.dry_run
            )
            total_points += count
            logging.info(f"[{i}/{len(chats)}] {title[:60]} ({count} messages indexed)")
        except Exception as e:
            logging.error(f"Failed to process chat {chat_id}: {e}")

    # Save state (skip for targeted reprocessing or dry-run)
    if not args.dry_run and not args.chats:
        with open(state_file, 'w') as f:
            json.dump({'last_sync': datetime.now(timezone.utc).isoformat()}, f)

    logging.warning(f"Completed: {len(chats)} chats, {total_points} message points indexed")
    return 0


class OpenWebUIClient:
    """Minimal OpenWebUI REST API client"""

    def __init__(self, url: str, api_key: str):
        self.url = url.rstrip('/')
        self.api_key = api_key

    def _headers(self) -> dict:
        return {'Authorization': f'Bearer {self.api_key}'}

    def list_chats(self, since_ts: float | None = None) -> list[dict]:
        """Fetch all chat metadata, optionally filtered to those updated since since_ts.

        OpenWebUI returns chats sorted newest-first by updated_at. We stop
        paginating early once we see a chat older than since_ts (if provided).
        """
        all_chats = []
        page = 1

        while True:
            logging.debug(f"Fetching chat list page {page}")
            resp = requests.get(
                f"{self.url}/api/v1/chats/",
                headers=self._headers(),
                params={'page': page}
            )
            resp.raise_for_status()

            data = resp.json()

            # API may return a list directly or wrap in a dict
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get('items', data.get('chats', []))
            else:
                items = []

            if not items:
                break

            for item in items:
                updated_at = item.get('updated_at', 0)
                # updated_at is a Unix timestamp (int or float)
                if since_ts is not None and updated_at <= since_ts:
                    # Items are newest-first; anything older can be skipped
                    logging.debug(f"Stopping pagination: found chat older than last_sync")
                    return all_chats
                all_chats.append(item)

            logging.debug(f"Page {page}: retrieved {len(items)} chats")
            page += 1

            # If the API returned fewer than a full page, we're done
            if len(items) < 20:
                break

        return all_chats

    def get_chat(self, chat_id: str) -> dict:
        """Fetch full chat content by UUID"""
        logging.debug(f"Fetching chat {chat_id}")
        resp = requests.get(
            f"{self.url}/api/v1/chats/{chat_id}",
            headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()


def extract_messages(chat: dict) -> list[dict]:
    """Extract an ordered list of messages from an OpenWebUI chat response.

    OpenWebUI stores messages as a dict keyed by UUID in chat.chat.history.messages.
    We reconstruct the ordered list by traversing the parentId chain from currentId.
    Falls back to timestamp sort if chain traversal fails.

    Returns a list of message dicts with at least: role, content, timestamp.
    """
    chat_data = chat.get('chat', {})

    # Handle both formats: messages may be nested under history or at top level
    history = chat_data.get('history', chat_data)
    messages_dict = history.get('messages', {})
    current_id = history.get('currentId')

    if not messages_dict:
        return []

    # Traverse parentId chain from currentId to reconstruct order
    ordered = []
    if current_id and current_id in messages_dict:
        node_id = current_id
        while node_id:
            msg = messages_dict.get(node_id)
            if not msg:
                break
            ordered.append(msg)
            node_id = msg.get('parentId')
        ordered.reverse()  # chain goes newest→root; reverse to get root→newest
    else:
        # Fallback: sort by timestamp
        logging.debug("Could not traverse message chain; falling back to timestamp sort")
        ordered = sorted(messages_dict.values(), key=lambda m: m.get('timestamp', 0))

    return ordered


def should_skip_message(msg: dict) -> bool:
    """Return True for messages that should not be indexed."""
    content = (msg.get('content') or '').strip()
    if not content:
        return True
    role = msg.get('role', '')
    # Skip tool calls, tool results, and system prompts — noisy, low retrieval value
    if role in ('tool', 'system'):
        return True
    # Skip messages that look like raw tool-call JSON blobs
    if content.startswith('{') and '"name"' in content and '"parameters"' in content:
        return True
    return False


def process_chat(chat: dict, openai_client: OpenAI,
                 qdrant: QdrantClient, collection_name: str,
                 embedding_model: str, chunk_size: int, chunk_overlap: int,
                 dry_run: bool = False) -> int:
    """Process a single chat: chunk messages, embed, delete old points, upsert new.

    Each message is split into overlapping chunks (same pattern as all other loaders).
    Short messages produce a single chunk; long messages (code dumps, pasted documents)
    are split so no individual chunk exceeds the embedding model's token limit.

    Ordering: all embeddings are generated FIRST. Existing Qdrant points are only
    deleted after embeddings succeed, so a failed embedding call never leaves the
    collection in a partially-updated state.
    """

    chat_id = chat.get('id', '')
    title = chat.get('title', 'Untitled')

    messages = extract_messages(chat)
    if not messages:
        logging.debug(f"Chat {chat_id}: no messages found")
        return 0

    # Filter messages
    indexable = [(idx, msg) for idx, msg in enumerate(messages) if not should_skip_message(msg)]
    if not indexable:
        logging.debug(f"Chat {chat_id}: no indexable messages after filtering")
        return 0

    # Build the flat list of (chunk_text, message_index, msg) across all messages.
    # chunk_index is the position of each chunk within its parent message.
    all_chunks: list[str] = []
    all_meta: list[tuple[int, dict, int]] = []  # (message_index, msg, chunk_index)

    for message_index, msg in indexable:
        content = msg.get('content', '').strip()
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append((message_index, msg, chunk_index))

    if not all_chunks:
        return 0

    # Step 1: Generate embeddings — if this raises, existing Qdrant points are preserved
    embeddings = []
    batch_size = 100
    for batch_start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[batch_start:batch_start + batch_size]
        embeddings.extend(get_embeddings(batch, openai_client, embedding_model))

    # Step 2: Embeddings succeeded — now safe to delete stale points
    if not dry_run:
        try:
            qdrant.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key='chat_id', match=MatchValue(value=chat_id))]
                )
            )
        except Exception as e:
            logging.debug(f"Delete failed for chat {chat_id} (may not exist): {e}")

    # Step 3: Build and upsert points
    points = []
    for (message_index, msg, chunk_index), chunk, embedding in zip(all_meta, all_chunks, embeddings):
        point_id = hashlib.md5(f"{chat_id}_{message_index}_{chunk_index}".encode()).hexdigest()

        # Convert Unix timestamp to ISO string
        ts_raw = msg.get('timestamp', 0) or 0
        try:
            ts_float = float(ts_raw)
            ts_iso = datetime.fromtimestamp(ts_float, tz=timezone.utc).isoformat()
        except (ValueError, OSError, OverflowError):
            ts_float = 0.0
            ts_iso = ''

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                'chat_id': chat_id,
                'chat_title': title,
                'role': msg.get('role', ''),
                'message_index': message_index,
                'chunk_index': chunk_index,
                'text': chunk,
                'content': chunk,
                'timestamp': ts_iso,
                'timestamp_ts': ts_float,
                'source': 'openwebui_chat',
            }
        ))

    if not dry_run and points:
        qdrant.upsert(collection_name=collection_name, points=points)

    return len(points)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, breaking at sentence boundaries.

    Identical to the chunk_text() used in wallabag_ingest.py and all other loaders.
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary (last 20% of chunk)
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

        # Create payload index on timestamp_ts for future range filtering
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name='timestamp_ts',
                field_schema=PayloadSchemaType.FLOAT
            )
        except Exception as e:
            logging.debug(f"Payload index creation skipped: {e}")
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
    send_alert(
        subject=f"[ALERT] openwebui_chat_loader failed on {socket.gethostname()}",
        body=msg
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
