"""
OpenWebUI Tool: Qdrant Knowledge Search

Search personal knowledge base (Wallabag articles, podcast transcripts, document
collections, RSS/Atom news feeds, and Kindle book highlights) stored in Qdrant.

Installation:
1. In OpenWebUI, go to Workspace → Tools → Create
2. Paste this entire file content
3. Configure the Valves (settings) with your Qdrant URL, API key, and OpenAI key
4. Enable the tool for your models

Usage:
The LLM can call search_knowledge(query, collection, date_from, date_to, date_field) to retrieve relevant
context from your indexed articles, transcripts, document collections, and feeds.
It can call get_full_article(article_id) to fetch full Wallabag article text,
or get_full_document(file_path, source_type) to fetch full document/podcast text
from the static file server.

Document collections beyond wallabag and podcasts are configurable via the
DOCUMENT_COLLECTIONS Valve (comma-separated collection names) and
DOCUMENT_COLLECTIONS_BASE_URL (files live under {base_url}/{collection_name}/).

title: Qdrant Knowledge Search
author: adam
description: Search personal knowledge base (Wallabag articles, podcast transcripts, document collections, and Kindle highlights)
version: 1.5.1
"""

from typing import Callable
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        QDRANT_URL: str = Field(
            default="http://host.docker.internal:6333",
            description="Qdrant server URL"
        )
        QDRANT_API_KEY: str = Field(
            default="",
            description="Qdrant API key (leave empty if not using authentication)"
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="OpenAI API key for generating query embeddings"
        )
        TOP_K: int = Field(
            default=8,
            description="Maximum total number of results to return"
        )
        PER_ARTICLE_MAX: int = Field(
            default=2,
            description="Preferred max results per article/episode (may exceed if total budget remains)"
        )
        WALLABAG_COLLECTION: str = Field(
            default="wallabag_articles",
            description="Qdrant collection name for Wallabag articles"
        )
        PODCAST_COLLECTION: str = Field(
            default="podcast_transcripts",
            description="Qdrant collection name for podcast transcripts"
        )
        FEEDS_COLLECTION: str = Field(
            default="news_feeds",
            description="Qdrant collection name for RSS/Atom feed articles"
        )
        KINDLE_COLLECTION: str = Field(
            default="kindle_highlights",
            description="Qdrant collection name for Kindle book highlights"
        )
        DOCUMENT_COLLECTIONS: str = Field(
            default="papers",
            description="Comma-separated Qdrant collection names for document collections (e.g., 'papers,books,manuals')"
        )
        DOCUMENT_COLLECTIONS_BASE_URL: str = Field(
            default="https://static-lan.maddock.net",
            description="Base URL for document collections; files are at {base_url}/{collection_name}/..."
        )
        WALLABAG_URL: str = Field(
            default="",
            description="Wallabag instance URL (e.g., https://wallabag.example.com)"
        )
        WALLABAG_CLIENT_ID: str = Field(
            default="",
            description="Wallabag API client ID"
        )
        WALLABAG_CLIENT_SECRET: str = Field(
            default="",
            description="Wallabag API client secret"
        )
        WALLABAG_USERNAME: str = Field(
            default="",
            description="Wallabag username"
        )
        WALLABAG_PASSWORD: str = Field(
            default="",
            description="Wallabag password"
        )
        PODCASTS_BASE_URL: str = Field(
            default="https://static-lan.maddock.net/podcasts",
            description="Base URL for podcast files on the static file server"
        )
        COHERE_API_KEY: str = Field(
            default="",
            description="Cohere API key for reranking results (leave empty to skip reranking)"
        )
        COHERE_RERANK_MODEL: str = Field(
            default="rerank-english-v3.0",
            description="Cohere rerank model to use"
        )

    def __init__(self):
        self.valves = self.Valves()
        self._wallabag_token = None
        self._wallabag_token_expires = 0

    def _get_document_collection_names(self) -> list[str]:
        """Parse DOCUMENT_COLLECTIONS into a list of collection names."""
        return [n.strip() for n in self.valves.DOCUMENT_COLLECTIONS.split(",") if n.strip()]

    def _get_base_url_for_source(self, source: str) -> str:
        """Build the base URL for a source type, with singular/plural fuzzy matching.

        Documents live at {DOCUMENT_COLLECTIONS_BASE_URL}/{collection_name}/...
        """
        base = self.valves.DOCUMENT_COLLECTIONS_BASE_URL
        if not base:
            return ""
        for coll in self._get_document_collection_names():
            if source == coll or source == coll.rstrip("s") or source + "s" == coll:
                return f"{base.rstrip('/')}/{coll}"
        return ""

    def _get_wallabag_token(self) -> str:
        """Get or refresh Wallabag OAuth token"""
        import time
        import requests
        
        if self._wallabag_token and time.time() < self._wallabag_token_expires:
            return self._wallabag_token
        
        url = self.valves.WALLABAG_URL.rstrip('/')
        resp = requests.post(f"{url}/oauth/v2/token", data={
            'grant_type': 'password',
            'client_id': self.valves.WALLABAG_CLIENT_ID,
            'client_secret': self.valves.WALLABAG_CLIENT_SECRET,
            'username': self.valves.WALLABAG_USERNAME,
            'password': self.valves.WALLABAG_PASSWORD
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        self._wallabag_token = data['access_token']
        self._wallabag_token_expires = time.time() + data.get('expires_in', 3600) - 60
        return self._wallabag_token

    async def get_full_article(
        self,
        article_id: int,
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        Fetch the full content of a Wallabag article by its ID.
        
        Use this after search_knowledge returns relevant snippets, when you need
        the complete article text for more detailed analysis or summarization.
        
        Args:
            article_id: The Wallabag article ID (from search results payload)
        
        Returns:
            The full article content with title and metadata
        """
        import requests
        import re
        
        if not self.valves.WALLABAG_URL:
            return "Error: Wallabag URL not configured in tool settings"
        
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Fetching full article {article_id} from Wallabag..."}
            })
        
        try:
            token = self._get_wallabag_token()
            url = self.valves.WALLABAG_URL.rstrip('/')
            
            resp = requests.get(
                f"{url}/api/entries/{article_id}.json",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30
            )
            resp.raise_for_status()
            article = resp.json()
            
            title = article.get('title', 'Untitled')
            content = article.get('content', '')
            
            # Strip HTML tags
            clean_content = re.sub(r'<[^>]+>', '', content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            domain = article.get('domain_name', '')
            article_url = article.get('url', '')
            reading_time = article.get('reading_time', 0)
            tags = [t['label'] for t in article.get('tags', [])]
            published_by = article.get('published_by', [])

            result = f"**{title}**\n"
            result += f"Source: {domain}\n"
            result += f"URL: {article_url}\n"
            result += f"Reading time: {reading_time} min\n"
            if tags:
                result += f"Tags: {', '.join(tags)}\n"
            if published_by:
                result += f"Author: {', '.join(published_by)}\n"
            result += f"\n---\n\n{clean_content}"
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Retrieved article: {title[:50]}..."}
                })
            
            return result
            
        except Exception as e:
            error_msg = f"Error fetching article {article_id}: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg}
                })
            return error_msg

    async def list_wallabag_tags(
        self,
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        List all tags in Wallabag with article counts.

        Fetches tags directly from the Wallabag API, sorted alphabetically.
        Article counts are included when the Wallabag instance supports them (v2.5.2+).
        Useful for exploring what topics have been saved to the knowledge base.

        Returns:
            A markdown-formatted list of all tags, with article counts if available.
        """
        import requests

        if not self.valves.WALLABAG_URL:
            return "Error: Wallabag URL not configured in tool settings"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": "Fetching tags from Wallabag..."}
            })

        try:
            token = self._get_wallabag_token()
            url = self.valves.WALLABAG_URL.rstrip('/')

            resp = requests.get(
                f"{url}/api/tags.json",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30
            )
            resp.raise_for_status()
            tags = resp.json()

            tags.sort(key=lambda t: t.get('label', '').lower())

            has_counts = any(t.get('nbEntries') is not None for t in tags)
            lines = [f"## Wallabag Tags ({len(tags)} total)\n"]
            for tag in tags:
                label = tag.get('label', tag.get('slug', ''))
                if has_counts:
                    count = tag.get('nbEntries', 0)
                    lines.append(f"- **{label}** ({count} articles)")
                else:
                    lines.append(f"- **{label}**")

            result = "\n".join(lines)

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Retrieved {len(tags)} tags from Wallabag"}
                })

            return result

        except Exception as e:
            error_msg = f"Error fetching Wallabag tags: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg}
                })
            return error_msg

    async def add_tag_to_article(
        self,
        article_id: int,
        tag: str,
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        Add a single tag to a Wallabag article.

        Only adds the tag — does not remove or replace existing tags. Wallabag
        will create the tag if it doesn't already exist.

        IMPORTANT: Only call this function when the user explicitly instructs
        you to add a tag. Never call it proactively or as part of general
        research and summarization.

        Args:
            article_id: The Wallabag article ID (from search results)
            tag: A single tag label to add, e.g. "ai"

        Returns:
            Confirmation with the article title and resulting tag list.
        """
        import requests

        if not self.valves.WALLABAG_URL:
            return "Error: Wallabag URL not configured in tool settings"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Adding tag '{tag}' to article {article_id}..."}
            })

        try:
            token = self._get_wallabag_token()
            url = self.valves.WALLABAG_URL.rstrip('/')

            resp = requests.post(
                f"{url}/api/entries/{article_id}/tags",
                headers={"Authorization": f"Bearer {token}"},
                data={"tags": tag.strip()},
                timeout=30
            )
            resp.raise_for_status()
            entry = resp.json()

            title = entry.get('title', f'Article {article_id}')
            updated_tags = [t['label'] for t in entry.get('tags', [])]

            result = f"Tag added to **{title}**\n"
            result += f"Added: {tag.strip()}\n"
            result += f"All tags: {', '.join(updated_tags)}"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Tag added to: {title[:50]}"}
                })

            return result

        except Exception as e:
            error_msg = f"Error adding tag to article {article_id}: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg}
                })
            return error_msg

    async def get_full_document(
        self,
        file_path: str,
        source_type: str = "papers",
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        Fetch the full text content of a document from the static file server.

        Use this after search_knowledge returns relevant snippets, when you need
        the complete document text for more detailed analysis or summarization.

        Args:
            file_path: The relative file path from search results payload
                       (e.g., 'paper_name.md' or 'ShowName/Episode.txt').
                       Use the raw path with regular spaces, NOT a URL-encoded
                       path (i.e., 'Show Name/Episode.txt' not 'Show%20Name/Episode.txt').
            source_type: The collection name from search results (the 'Collection'
                         field, e.g. 'papers', 'books'). For podcasts use 'podcasts'.
                         The collection name is used as the folder name in the URL.

        Returns:
            The full document content with metadata header
        """
        from urllib.parse import unquote
        import requests

        # Normalize: decode any URL-encoded characters (e.g., %20 → space)
        # so callers can pass either raw paths or URL-encoded paths.
        file_path = unquote(file_path)

        if source_type in ["podcast", "podcasts", "podcast_transcript"]:
            base_url = self.valves.PODCASTS_BASE_URL
        else:
            base_url = self._get_base_url_for_source(source_type)

        if not base_url:
            valid_types = ["podcasts"] + self._get_document_collection_names()
            return f"Error: Unknown or unconfigured source_type '{source_type}'. Valid types: {', '.join(valid_types)}"

        full_url = self._build_static_url(base_url, file_path)

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Fetching full document: {file_path}..."}
            })

        try:
            resp = requests.get(full_url, timeout=30)
            resp.raise_for_status()

            resp.encoding = resp.apparent_encoding or 'utf-8'
            content = resp.text

            result = f"**Document: {file_path}**\n"
            result += f"Source type: {source_type}\n"
            result += f"URL: {full_url}\n"
            result += f"Length: {len(content)} characters\n"
            result += f"\n---\n\n{content}"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Retrieved document: {file_path}"}
                })

            return result

        except requests.exceptions.HTTPError as e:
            error_msg = f"Error fetching document '{file_path}': HTTP {e.response.status_code}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg}
                })
            return error_msg
        except Exception as e:
            error_msg = f"Error fetching document '{file_path}': {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg}
                })
            return error_msg

    @staticmethod
    def _build_static_url(base_url: str, relative_path: str) -> str:
        """Build a full URL from a base URL and a relative file path."""
        from urllib.parse import quote, unquote
        # Decode first to handle already-encoded paths (e.g., %20 → space),
        # then re-encode properly. This prevents double-encoding (%20 → %2520).
        decoded_path = unquote(relative_path)
        encoded_path = "/".join(quote(segment, safe="") for segment in decoded_path.split("/"))
        return f"{base_url.rstrip('/')}/{encoded_path}"

    @staticmethod
    def _article_key(point) -> str:
        """Return a grouping key for a result point (article ID, episode name, etc.)."""
        payload = point.payload
        source = payload.get("source", "unknown")
        if source == "wallabag":
            return f"wallabag:{payload.get('article_id', payload.get('title', 'unknown'))}"
        elif source == "podcast_transcript":
            return f"podcast:{payload.get('show_name', '')}:{payload.get('episode_name', '')}"
        elif source == "feed":
            return f"feed:{payload.get('entry_id', payload.get('title', 'unknown'))}"
        elif source == "kindle":
            return f"kindle:{payload.get('asin', payload.get('book_title', 'unknown'))}"
        else:
            doc_id = payload.get('document_name', payload.get('file_path', payload.get('title', 'unknown')))
            return f"{source}:{doc_id}"

    def _cohere_rerank(self, query: str, points: list) -> list:
        """Rerank results using Cohere's cross-encoder API. Returns points reordered by relevance."""
        import requests

        documents = [p.payload.get("text", "") for p in points]
        resp = requests.post(
            "https://api.cohere.com/v1/rerank",
            headers={
                "Authorization": f"Bearer {self.valves.COHERE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.valves.COHERE_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        # results is [{index, relevance_score}, ...] ordered by relevance descending
        return [points[r["index"]] for r in results]

    @staticmethod
    def _diversified_top_k(results: list, total_max: int, per_article_max: int) -> list:
        """
        Select up to total_max results, preferring at most per_article_max per article.

        Pass 1: iterate by score, accepting each result until that article hits
                 per_article_max. Deferred results go to a spillover list.
        Pass 2: if the total budget isn't filled, pull from spillover (still
                 sorted by score) regardless of per-article counts.
        """
        selected = []
        spillover = []
        article_counts: dict[str, int] = {}

        for point in results:
            if len(selected) >= total_max:
                break
            key = Tools._article_key(point)
            count = article_counts.get(key, 0)
            if count < per_article_max:
                selected.append(point)
                article_counts[key] = count + 1
            else:
                spillover.append(point)

        # Fill remaining budget from spillover (already in score order)
        for point in spillover:
            if len(selected) >= total_max:
                break
            selected.append(point)

        return selected

    def _build_date_filter(self, coll_name: str, date_from: str, date_to: str, date_mode: str):
        """
        Build a Qdrant Filter for date range, or return None if no filtering needed.

        date_from / date_to are ISO date strings (YYYY-MM-DD), either may be None.
        date_mode is "published" (publication date) or "indexed" (ingest/save date).
        The correct payload key is resolved per collection type internally.
        """
        if not date_from and not date_to:
            return None

        from datetime import date, datetime, timezone
        from qdrant_client.models import DatetimeRange, Filter, FieldCondition, Range

        is_feeds = coll_name == self.valves.FEEDS_COLLECTION
        is_wallabag = coll_name == self.valves.WALLABAG_COLLECTION
        is_podcast = coll_name == self.valves.PODCAST_COLLECTION
        is_doc = coll_name in self._get_document_collection_names()

        if date_mode == "indexed":
            if is_wallabag:
                payload_key = "created_at"
            elif is_feeds:
                payload_key = "published_ts"  # best proxy — feeds have no ingest timestamp
            elif is_podcast or is_doc:
                payload_key = "modified_at"
            else:
                return None  # Kindle — no date fields
        else:  # "published" (default)
            if is_feeds:
                payload_key = "published_ts"
            elif is_wallabag or is_podcast:
                payload_key = "published_at"
            elif is_doc:
                payload_key = "modified_at"
            else:
                return None  # Kindle — no date fields

        range_kwargs = {}
        if payload_key == "published_ts":
            # published_ts is a Unix float — use numeric Range
            def _iso_to_ts(s):
                return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()
            if date_from:
                range_kwargs["gte"] = _iso_to_ts(date_from)
            if date_to:
                range_kwargs["lte"] = _iso_to_ts(date_to)
            range_filter = Range(**range_kwargs)
        else:
            # Use DatetimeRange for proper datetime payload fields
            if date_from:
                range_kwargs["gte"] = date.fromisoformat(date_from)
            if date_to:
                range_kwargs["lte"] = date.fromisoformat(date_to)
            range_filter = DatetimeRange(**range_kwargs)

        return Filter(must=[FieldCondition(key=payload_key, range=range_filter)])

    async def search_knowledge(
        self,
        query: str,
        collection: str = "all",
        date_from: str = None,
        date_to: str = None,
        date_mode: str = "published",
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        Search personal knowledge base for relevant information.

        Use this tool when the user asks questions that might benefit from
        personal context, such as saved articles or podcast content.

        Args:
            query: The search query - what information to look for
            collection: Which collection to search:
                        - 'articles' for Wallabag saved articles only
                        - 'podcasts' for podcast transcripts only
                        - 'feeds' for RSS/Atom news feed articles only
                        - 'kindle' for Kindle book highlights only
                        - 'documents' for all document collections
                        - a specific collection name (e.g. 'papers', 'books')
                        - 'all' for everything (default)
            date_from: Optional start of date range, ISO format (YYYY-MM-DD).
                       If omitted with date_to set, searches from inception to date_to.
            date_to: Optional end of date range, ISO format (YYYY-MM-DD).
                     If omitted with date_from set, searches from date_from to present.
            date_mode: Which date concept to filter on:
                       - 'published' (default) — article/episode publication date
                       - 'indexed' — when the item was added/saved to the knowledgebase

        Returns:
            Relevant context from the knowledge base, formatted with source information
        """
        from qdrant_client import QdrantClient
        from openai import OpenAI

        # Emit status
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Searching knowledge base for: {query[:50]}..."}
            })

        try:
            # Initialize clients
            qdrant = QdrantClient(
                url=self.valves.QDRANT_URL,
                api_key=self.valves.QDRANT_API_KEY or None
            )
            openai_client = OpenAI(api_key=self.valves.OPENAI_API_KEY)

            # Get query embedding
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_vector = response.data[0].embedding

            # Determine collections to search
            collections = []
            if collection in ["articles", "all"]:
                collections.append(self.valves.WALLABAG_COLLECTION)
            if collection in ["podcasts", "all"]:
                collections.append(self.valves.PODCAST_COLLECTION)
            if collection in ["feeds", "all"]:
                collections.append(self.valves.FEEDS_COLLECTION)
            if collection in ["kindle", "all"]:
                collections.append(self.valves.KINDLE_COLLECTION)

            doc_names = self._get_document_collection_names()

            if collection in ["documents", "all"]:
                collections.extend(doc_names)
            elif collection in doc_names:
                collections.append(collection)

            if not collections:
                valid = ["articles", "podcasts", "feeds", "kindle", "documents", "all"] + doc_names
                return f"Unknown collection: {collection}. Valid options: {', '.join(valid)}"

            all_results = []

            for coll_name in collections:
                fetch_limit = self.valves.TOP_K * 3
                date_filter = self._build_date_filter(coll_name, date_from, date_to, date_mode)
                results = None
                try:
                    results = qdrant.query_points(
                        collection_name=coll_name,
                        query=query_vector,
                        query_filter=date_filter,
                        limit=fetch_limit
                    )
                    # If the date filter produced no results, retry without it so the
                    # collection still contributes its best semantic matches
                    if not results.points and date_filter is not None:
                        results = qdrant.query_points(
                            collection_name=coll_name,
                            query=query_vector,
                            limit=fetch_limit
                        )
                except Exception as e:
                    if date_filter is not None:
                        # Filtered query failed — retry without the date filter
                        try:
                            results = qdrant.query_points(
                                collection_name=coll_name,
                                query=query_vector,
                                limit=fetch_limit
                            )
                        except Exception as e2:
                            if __event_emitter__:
                                await __event_emitter__({
                                    "type": "status",
                                    "data": {"description": f"Warning: Could not search {coll_name}: {e2}"}
                                })
                    else:
                        if __event_emitter__:
                            await __event_emitter__({
                                "type": "status",
                                "data": {"description": f"Warning: Could not search {coll_name}: {e}"}
                            })
                if results is not None:
                    for point in results.points:
                        point._collection_name = coll_name
                    all_results.extend(results.points)

            # Diversified top-K: respect per-article limits while filling total budget
            all_results.sort(key=lambda x: x.score, reverse=True)

            if self.valves.COHERE_API_KEY and all_results:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": f"Reranking {len(all_results)} candidates with Cohere..."}
                    })
                try:
                    all_results = self._cohere_rerank(query, all_results)
                except Exception as e:
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": f"Cohere rerank failed, using embedding scores: {e}"}
                        })

            top_results = self._diversified_top_k(
                all_results, self.valves.TOP_K, self.valves.PER_ARTICLE_MAX
            )

            if not top_results:
                return "No relevant information found in the knowledge base."

            # Format results
            context_parts = []
            for r in top_results:
                payload = r.payload
                source = payload.get("source", "unknown")

                if source == "wallabag":
                    header = (
                        f"**Article: {payload.get('title', 'Untitled')}**\n"
                        f"Article ID: {payload.get('article_id')}\n"
                        f"Source: {payload.get('domain', 'unknown')}\n"
                        f"URL: {payload.get('url', 'N/A')}"
                    )
                    if payload.get('tags'):
                        header += f"\nTags: {', '.join(payload['tags'])}"
                    if payload.get('published_by'):
                        header += f"\nAuthor: {', '.join(payload['published_by'])}"
                elif source == "podcast_transcript":
                    header = (
                        f"**Podcast: {payload.get('show_name', 'Unknown Show')}**\n"
                        f"Episode: {payload.get('episode_name', 'Unknown Episode')}"
                    )
                    if self.valves.PODCASTS_BASE_URL:
                        file_path = payload.get('file_path', '')
                        if file_path:
                            header += f"\nTranscript: {self._build_static_url(self.valves.PODCASTS_BASE_URL, file_path)}"
                        audio_file = payload.get('audio_file', '')
                        if audio_file:
                            header += f"\nAudio: {self._build_static_url(self.valves.PODCASTS_BASE_URL, audio_file)}"
                    if payload.get('tags'):
                        header += f"\nTags: {', '.join(payload['tags'])}"
                elif source == "feed":
                    header = (
                        f"**Feed Article: {payload.get('title', 'Untitled')}**\n"
                        f"Feed: {payload.get('feed_name', payload.get('feed_url', 'Unknown Feed'))}\n"
                        f"URL: {payload.get('url', 'N/A')}"
                    )
                    if payload.get('published'):
                        header += f"\nPublished: {payload['published']}"
                    if payload.get('author'):
                        header += f"\nAuthor: {payload['author']}"
                    if payload.get('tags'):
                        header += f"\nTags: {', '.join(payload['tags'])}"
                elif source == "kindle":
                    header = (
                        f"**Kindle: {payload.get('book_title', 'Unknown Book')}**\n"
                        f"Author: {payload.get('authors', 'Unknown')}\n"
                        f"Location: {payload.get('location_value', 'N/A')}"
                    )
                    if payload.get('location_url'):
                        header += f"\nKindle Link: {payload['location_url']}"
                else:
                    collection_name = getattr(r, '_collection_name', source)
                    doc_name = payload.get('document_name', payload.get('title', 'Unknown Document'))
                    header = (
                        f"**Document: {doc_name}**\n"
                        f"Collection: {collection_name}"
                    )
                    file_path = payload.get('file_path', '')
                    if file_path:
                        header += f"\nFile: {file_path}"
                    base_url = self._get_base_url_for_source(collection_name)
                    if base_url:
                        link_path = payload.get('original_file') or file_path
                        if link_path:
                            header += f"\nURL: {self._build_static_url(base_url, link_path)}"
                    for key in ['author', 'date', 'category', 'tags']:
                        if key in payload:
                            val = payload[key]
                            if isinstance(val, list):
                                val = ', '.join(str(v) for v in val)
                            header += f"\n{key.title()}: {val}"

                if payload.get('published_at') and '\nPublished:' not in header:
                    header += f"\nPublished: {payload['published_at']}"

                text = payload.get('text', '')
                context_parts.append(f"{header}\n\n{text}\n\n---")

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Found {len(top_results)} relevant results"}
                })

            return "\n\n".join(context_parts)

        except Exception as e:
            error_msg = f"Error searching knowledge base: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg}
                })
            return error_msg
