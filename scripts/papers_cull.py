#!/usr/bin/env python3
"""
papers_cull.py - Remove Qdrant chunks of documents gone from a papers dir

papers_ingest.py adds and re-embeds, but never notices a file that has gone. A
deleted or renamed document keeps its chunks in the collection, and its entry
in .papers_sync_state_<collection>.json, until something removes them. This is
that something. See file_cull.py for the rule and the guards.

Run it with the same --papers-dir and --collection as the papers run it tidies
up after. Chunk ids are paths relative to that directory, so a different
spelling of it makes every chunk look orphaned (the guards refuse that). It
assumes one directory feeds the collection, which is how papers_ingest.py is
run.
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
from file_cull import cull, papers_scope, self_test
from qdrant_client import QdrantClient

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'


def main():
    config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
    load_dotenv(config_dir / '.env')

    parser = argparse.ArgumentParser(
        description='Remove Qdrant chunks of documents no longer in a papers dir'
    )
    parser.add_argument("--papers-dir", default=os.environ.get('PAPERS_DIR'), help="Root directory the collection is ingested from (same as for 'papers')")
    parser.add_argument("--collection", default=None, help="Qdrant collection name (overrides PAPERS_COLLECTION env var, default: 'papers')")
    parser.add_argument("--force", action="store_true", help="Cull even when the guards think the directory is missing or mostly gone")
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

    if not args.papers_dir:
        parser.error("--papers-dir is required (or set PAPERS_DIR)")
    if not os.environ.get('QDRANT_URL'):
        log_fatal("Missing required environment variable: QDRANT_URL")

    collection_name = args.collection or os.environ.get('PAPERS_COLLECTION', 'papers')
    state_file = config_dir / f'.papers_sync_state_{collection_name}.json'

    papers_dir = Path(args.papers_dir)
    if not papers_dir.is_dir():
        log_fatal(f"Papers directory does not exist: {papers_dir}")

    # Before reading the disk: a papers run for this collection that read its
    # state already would write back the entries this is about to prune.
    if not args.dry_run and not claim_state(state_file):
        return 0

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Api key is used with an insecure connection')
        qdrant = QdrantClient(
            url=os.environ['QDRANT_URL'],
            api_key=os.environ.get('QDRANT_API_KEY')
        )

    return cull(qdrant, collection_name, [papers_scope(papers_dir)], state_file,
                name='papers-cull', dry_run=args.dry_run, force=args.force,
                alert=lambda subject, body: send_alert(
                    f"[ALERT] {subject} on {socket.gethostname()}", body))


def log_fatal(msg, exit_code=-1):
    logging.critical(f"Fatal Err: {msg}")
    send_alert(
        subject=f"[ALERT] papers_cull failed on {socket.gethostname()}",
        body=msg
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    sys.exit(run_guarded('papers_cull', main))
