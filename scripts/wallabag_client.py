#!/usr/bin/env python3
"""
wallabag_client.py - Shared Wallabag API client

Thin wrapper over the Wallabag REST API, shared by the scripts that talk to
Wallabag (wallabag_ingest.py, wallabag_cull.py, summarize.py). Handles the
OAuth password grant and caches the access token in memory for the life of
the process.
"""

import logging
import time
from datetime import datetime

import requests


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

    def get_tags(self) -> list[dict]:
        """Fetch the user's full tag list, as [{'id': .., 'label': .., 'slug': ..}]"""
        token = self._get_token()
        headers = {'Authorization': f'Bearer {token}'}
        resp = requests.get(f"{self.url}/api/tags.json", headers=headers)
        resp.raise_for_status()
        return resp.json()

    def get_version(self) -> str:
        """Fetch the Wallabag server version (public endpoint, no auth needed)"""
        resp = requests.get(f"{self.url}/api/version.json")
        resp.raise_for_status()
        return resp.json()
