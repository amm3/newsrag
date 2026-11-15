#!/usr/bin/env python3
"""
summaries_cull.py - Remove Qdrant chunks of .ai-summary.md files that are gone

summaries_ingest.py has the same blind spot as papers_ingest.py: it embeds new
and edited summaries but never notices one that was deleted, so a summary of a
document that no longer exists stays searchable. This removes those chunks and
their .summaries_sync_state.json entries. See file_cull.py for the rule and the
guards.

Run it with the same --podcast-dir/--papers-dir as the summaries run. Only the
roots it is given are in scope: points with a source_type it was not given a
directory for are left alone, so a papers-only cull never touches podcast
summaries.
"""

import sys
import os
import argparse
import logging
import socket
import warnings
from pathlib import Path

from dotenv import load_dotenv

from alert import send_alert, run_guarded
from sync_state import claim_state
from file_cull import cull, summary_scope, self_test
from qdrant_client import QdrantClient

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'


def main():
    config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
    load_dotenv(config_dir / '.env')

    parser = argparse.ArgumentParser(
        description='Remove Qdrant chunks of .ai-summary.md files that no longer exist'
    )
    parser.add_argument("--podcast-dir", default=os.environ.get('PODCAST_DIR'), help="Root directory containing podcasts (same as for 'summaries')")
    parser.add_argument("--papers-dir", default=os.environ.get('PAPERS_DIR'), help="Root directory containing papers/documents (same as for 'summaries')")
    parser.add_argument("--collection", default=None, help="Qdrant collection name (overrides SUMMARIES_COLLECTION env var, default: 'summaries')")
    parser.add_argument("--force", action="store_true", help="Cull even when the guards think a directory is missing or mostly gone")
    parser.add_argument("--dry-run", action="store_true", help="List what would be removed; change nothing")
    parser.add_argument("--self-test", action="store_true", help="Check the cull rules against an in-memory Qdrant and exit")
    parser.add_argument("-v", action="store_true", default=False, help="Print extra info")
    parser.add_argument("-vv", action="store_true", default=False, help="Print (more) extra info")
    args = parser.parse_args()

    if args.vv:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.DEBUG)
    elif args.v:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.INFO)
    else:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.WARNING)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if args.self_test:
        return self_test(verbose=args.v or args.vv)

    if not os.environ.get('QDRANT_URL'):
        log_fatal("Missing required environment variable: QDRANT_URL")

    collection_name = args.collection or os.environ.get('SUMMARIES_COLLECTION', 'summaries')
    # One file whatever the collection, as summaries_ingest.py keeps it
    state_file = config_dir / '.summaries_sync_state.json'

    # A root that does not exist is skipped, as the loader skips it. Its
    # points are then out of scope, which is the safe way round.
    roots: list[tuple[Path, str]] = []
    for flag, value, source_type in (('--podcast-dir', args.podcast_dir, 'podcast'),
                                     ('--papers-dir', args.papers_dir, 'paper')):
        if not value:
            continue
        p = Path(value)
        if p.is_dir():
            roots.append((p, source_type))
        else:
            logging.warning(f"{flag} does not exist, skipping its summaries: {p}")
    if not roots:
        log_fatal("No valid --podcast-dir or --papers-dir provided (or PODCAST_DIR/PAPERS_DIR env vars)")

    # Before reading the disk: a summaries run that read its state already
    # would write back the entries this is about to prune.
    if not args.dry_run and not claim_state(state_file):
        return 0

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Api key is used with an insecure connection')
        qdrant = QdrantClient(
            url=os.environ['QDRANT_URL'],
            api_key=os.environ.get('QDRANT_API_KEY')
        )

    scopes = [summary_scope(root, source_type) for root, source_type in roots]
    return cull(qdrant, collection_name, scopes, state_file,
                name='summaries-cull', dry_run=args.dry_run, force=args.force,
                alert=lambda subject, body: send_alert(
                    f"[ALERT] {subject} on {socket.gethostname()}", body))


def log_fatal(msg, exit_code=-1):
    logging.critical(f"Fatal Err: {msg}")
    send_alert(
        subject=f"[ALERT] summaries_cull failed on {socket.gethostname()}",
        body=msg
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    sys.exit(run_guarded('summaries_cull', main))
