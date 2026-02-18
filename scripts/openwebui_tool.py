"""
OpenWebUI Tool: Qdrant Knowledge Search

Search personal knowledge base (Wallabag articles, podcast transcripts, and
papers) stored in Qdrant.

Installation:
1. In OpenWebUI, go to Workspace → Tools → Create
2. Paste this entire file content
3. Configure the Valves (settings) with your Qdrant URL, API key, and OpenAI key
4. Enable the tool for your models

Usage:
The LLM can call search_knowledge(query, collection) to retrieve relevant
context from your indexed articles, transcripts, and papers.
It can call get_full_article(article_id) to fetch full Wallabag article text,
or get_full_document(file_path, source_type) to fetch full paper/podcast text
from the static file server.

title: Qdrant Knowledge Search
author: adam
description: Search personal knowledge base (Wallabag articles, podcast transcripts, and papers)
version: 1.0.0
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
        PAPERS_COLLECTION: str = Field(
            default="papers",
            description="Qdrant collection name for papers/documents"
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
        PAPERS_BASE_URL: str = Field(
            default="https://static-lan.maddock.net/papers",
            description="Base URL for papers on the static file server"
        )
        PODCASTS_BASE_URL: str = Field(
            default="https://static-lan.maddock.net/podcasts",
            description="Base URL for podcast files on the static file server"
        )

    def __init__(self):
        self.valves = self.Valves()
        self._wallabag_token = None
        self._wallabag_token_expires = 0

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
            
            result = f"**{title}**\n"
            result += f"Source: {domain}\n"
            result += f"URL: {article_url}\n"
            result += f"Reading time: {reading_time} min\n"
            if tags:
                result += f"Tags: {', '.join(tags)}\n"
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

    async def get_full_document(
        self,
        file_path: str,
        source_type: str = "paper",
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        Fetch the full text content of a paper or podcast transcript from the
        static file server.

        Use this after search_knowledge returns relevant snippets, when you need
        the complete document text for more detailed analysis or summarization.

        Args:
            file_path: The relative file path from search results payload
                       (e.g., 'paper_name.md' or 'ShowName/Episode.txt')
            source_type: Either 'paper' or 'podcast' to determine the base URL

        Returns:
            The full document content with metadata header
        """
        import requests

        if source_type == "paper":
            base_url = self.valves.PAPERS_BASE_URL
        elif source_type == "podcast":
            base_url = self.valves.PODCASTS_BASE_URL
        else:
            return f"Error: Unknown source_type '{source_type}'. Use 'paper' or 'podcast'."

        if not base_url:
            return f"Error: Base URL not configured for {source_type}"

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
        from urllib.parse import quote
        encoded_path = "/".join(quote(segment, safe="") for segment in relative_path.split("/"))
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
        elif source == "paper":
            return f"paper:{payload.get('document_name', payload.get('file_path', 'unknown'))}"
        return f"other:{payload.get('title', id(point))}"

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

    async def search_knowledge(
        self,
        query: str,
        collection: str = "all",
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
                        - 'papers' for papers/documents only
                        - 'all' for everything (default)

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
            if collection in ["papers", "all"]:
                collections.append(self.valves.PAPERS_COLLECTION)

            if not collections:
                return f"Unknown collection: {collection}. Use 'articles', 'podcasts', 'papers', or 'all'."

            all_results = []

            for coll_name in collections:
                try:
                    # Fetch extra candidates to allow diversification across articles
                    fetch_limit = self.valves.TOP_K * 3
                    results = qdrant.query_points(
                        collection_name=coll_name,
                        query=query_vector,
                        limit=fetch_limit
                    )
                    all_results.extend(results.points)
                except Exception as e:
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": f"Warning: Could not search {coll_name}: {e}"}
                        })

            # Diversified top-K: respect per-article limits while filling total budget
            all_results.sort(key=lambda x: x.score, reverse=True)
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
                elif source == "paper":
                    header = (
                        f"**Paper: {payload.get('document_name', 'Unknown Document')}**\n"
                        f"File: {payload.get('file_path', 'N/A')}"
                    )
                    if self.valves.PAPERS_BASE_URL:
                        link_path = payload.get('original_file') or payload.get('file_path', '')
                        if link_path:
                            header += f"\nURL: {self._build_static_url(self.valves.PAPERS_BASE_URL, link_path)}"
                else:
                    header = f"**Source: {source}**"

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
