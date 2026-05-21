#!/usr/bin/env python3
"""
wallabag_cull.py - Remove orphaned Qdrant entries for deleted Wallabag articles

Fetches all current Wallabag entry IDs, finds gaps in the sequential ID range
(which indicate deleted articles), then removes any matching Qdrant points.
"""

import sys
import os
import argparse
import logging
import time
import warnings
from pathlib import Path

import requests
from dotenv import load_dotenv

from alert import send_alert
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'


def log_fatal(msg: str):
    logging.error(msg)
    send_alert("wallabag-cull fatal error", msg)
    sys.exit(1)


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

    def get_entries(self, per_page=30):
        """Fetch all entries (paginated), returning full entry dicts"""
        token = self._get_token()
        headers = {'Authorization': f'Bearer {token}'}

        all_entries = []
        page = 1

        while True:
            params = {'perPage': per_page, 'page': page}
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


def main():
    parser = argparse.ArgumentParser(
        description='Remove orphaned Qdrant entries for deleted Wallabag articles'
    )
    parser.add_argument("-v", action="store_true", default=False, help="Print extra info")
    parser.add_argument("-vv", action="store_true", default=False, help="Print (more) extra info")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Qdrant")
    args = parser.parse_args()

    if args.vv:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.DEBUG)
    elif args.v:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.INFO)
    else:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.WARNING)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
    load_dotenv(config_dir / '.env')

    required_vars = ['WALLABAG_URL', 'WALLABAG_CLIENT_ID', 'WALLABAG_CLIENT_SECRET',
                     'WALLABAG_USERNAME', 'WALLABAG_PASSWORD', 'QDRANT_URL']
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        log_fatal(f"Missing required environment variables: {', '.join(missing)}")

    collection_name = os.environ.get('WALLABAG_COLLECTION', 'wallabag_articles')

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

    if args.dry_run:
        logging.warning("Dry run mode — no changes will be made to Qdrant")

    # 1. Fetch all current Wallabag entry IDs
    logging.info("Fetching all Wallabag entry IDs...")
    try:
        entries = wallabag.get_entries()
    except Exception as e:
        log_fatal(f"Failed to fetch Wallabag entries: {e}")

    wallabag_ids = {e['id'] for e in entries}

    if not wallabag_ids:
        logging.warning("No entries returned from Wallabag, aborting")
        return 1

    logging.info(f"Wallabag has {len(wallabag_ids)} entries")

    # 2. Scroll Qdrant to collect all unique article_ids that are indexed
    logging.info("Scanning Qdrant collection for indexed article IDs...")
    qdrant_ids = set()
    offset = None
    while True:
        try:
            results, offset = qdrant.scroll(
                collection_name=collection_name,
                with_payload=['article_id'],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
        except Exception as e:
            log_fatal(f"Failed to scroll Qdrant collection: {e}")

        for point in results:
            aid = point.payload.get('article_id')
            if aid is not None:
                qdrant_ids.add(aid)

        if offset is None:
            break

    logging.info(f"Qdrant has {len(qdrant_ids)} unique article IDs indexed")

    # 3. Orphaned = indexed in Qdrant but no longer in Wallabag
    orphaned_ids = sorted(qdrant_ids - wallabag_ids, reverse=True)  # most recent first
    logging.info(f"Found {len(orphaned_ids)} orphaned article ID(s)")

    if not orphaned_ids:
        logging.warning("No orphaned articles found — nothing to cull")
        return 0

    # 4. Delete orphaned articles from Qdrant
    deleted = 0
    for article_id in orphaned_ids:
        article_filter = Filter(
            must=[FieldCondition(key='article_id', match=MatchValue(value=article_id))]
        )
        logging.info(f"Orphaned ID {article_id}: removing from Qdrant")
        if not args.dry_run:
            try:
                qdrant.delete(collection_name=collection_name, points_selector=article_filter)
            except Exception as e:
                logging.warning(f"Orphaned ID {article_id}: delete failed: {e}")
                continue
        deleted += 1

    suffix = " (dry run)" if args.dry_run else ""
    logging.warning(f"Cull complete{suffix}: {deleted} orphaned article(s) removed from Qdrant")
    return 0


if __name__ == '__main__':
    sys.exit(main())
