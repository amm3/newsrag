"""
OpenWebUI Tool: Qdrant Knowledge Search

Search personal knowledge base (Wallabag articles, podcast transcripts, document
collections, RSS/Atom news feeds, Kindle book highlights, and AI-generated
podcast/paper summaries) stored in Qdrant.

Installation:
1. In OpenWebUI, go to Workspace → Tools → Create
2. Paste this entire file content
3. Configure the Valves (settings) with your Qdrant URL, API key, and OpenAI key
4. Enable the tool for your models

Usage:
The LLM can call search_knowledge(query, collection, date_from, date_to, date_mode, tag) to
retrieve relevant context from your indexed articles, transcripts, document collections,
feeds, and summaries.
It can call get_full_article(article_id) to fetch full Wallabag article text
plus any Wallabag annotations (highlights and notes) on that article,
get_full_document(file_path, source_type) to fetch full document/podcast text
from the static file server, or get_kindle_highlights(file_name) to fetch every
saved highlight and annotation for a specific Kindle book.

It can call list_recent_feed_articles(days, date_from, date_to, feed_name) to
retrieve a complete, deduplicated listing of every RSS/Atom feed article in a
time window — unranked and uncapped, with no categorization or summarization —
for downstream pipelines (e.g. an n8n daily news roundup) rather than
relevance search.

It can call update_article_status(article_ids, read, starred) to mark one or
more Wallabag articles read/unread (archived) and/or starred/unstarred, only
on explicit user instruction.

Document collections beyond wallabag and podcasts are configurable via the
DOCUMENT_COLLECTIONS Valve (comma-separated collection names) and
DOCUMENT_COLLECTIONS_BASE_URL (files live under {base_url}/{collection_name}/).

AI-generated summaries (from a separate summarization pipeline, not Wallabag)
live in the SUMMARIES_COLLECTION collection and exist purely to widen semantic
recall for podcasts/papers. They are never ground truth — always resolve back
to the real transcript/paper via get_full_document (using the summary result's
source_file + source_type) before citing specifics.

title: Qdrant Knowledge Search
author: adam
description: Search personal knowledge base (Wallabag articles, podcast transcripts, document collections, RSS/Atom feeds, Kindle highlights, and AI-generated summaries)
version: 1.12.0
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
        SUMMARIES_COLLECTION: str = Field(
            default="summaries",
            description="Qdrant collection name for AI-generated podcast/paper summaries"
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
        KINDLE_HIGHLIGHTS_BASE_URL: str = Field(
            default="https://static-lan.maddock.net/kindle_highlights",
            description="Base URL for Kindle highlight JSON files on the static file server"
        )
        COHERE_API_KEY: str = Field(
            default="",
            description="Cohere API key for reranking results (leave empty to skip reranking)"
        )
        COHERE_RERANK_MODEL: str = Field(
            default="rerank-english-v3.0",
            description="Cohere rerank model to use"
        )
        ANALYZE_TWITTER_IMAGES: bool = Field(
            default=True,
            description="Automatically describe images attached to x.com/twitter.com articles in get_full_article using vision"
        )
        VISION_MODEL: str = Field(
            default="gpt-4o-mini",
            description="OpenAI vision-capable model used to describe images attached to X/Twitter articles"
        )
        MAX_IMAGES_TO_ANALYZE: int = Field(
            default=3,
            description="Maximum number of attached images to run through vision analysis per get_full_article call"
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

    def _describe_image(self, image_url: str) -> str | None:
        """Describe an image with vision. Returns None on any failure (missing key,
        network error, rate limit, model error) so a vision failure never breaks
        the caller's larger response."""
        if not self.valves.OPENAI_API_KEY:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.valves.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=self.valves.VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This image was attached to a post saved from X (Twitter). "
                                "Describe exactly what it shows. If it is a screenshot of "
                                "another post, tweet, or article, transcribe the visible text "
                                "as accurately as possible, including the author's name or "
                                "handle if visible. If it is a photo, chart, or meme, describe "
                                "it concisely."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                max_tokens=350,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None

    async def get_full_article(
        self,
        article_id: int,
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        Fetch the full content of a Wallabag article by its ID.

        Use this after search_knowledge returns relevant snippets, when you need
        the complete article text for more detailed analysis or summarization.
        Any highlighted passages the user annotated in Wallabag, along with any
        note text attached to them, are appended after the article body.

        Args:
            article_id: The Wallabag article ID (from search results payload)

        Returns:
            The full article content with title, metadata, and the user's
            annotations (highlights and notes), if any
        """
        import requests
        import re
        import html

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

            annotations = []
            try:
                ann_resp = requests.get(
                    f"{url}/api/annotations/{article_id}.json",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                )
                ann_resp.raise_for_status()
                annotations = ann_resp.json().get("rows", [])
            except requests.exceptions.RequestException:
                annotations = []

            title = article.get('title', 'Untitled')
            content = article.get('content', '')
            domain = article.get('domain_name', '')

            # Convert images to markdown image syntax (so screenshots embedded in
            # articles, e.g. saved tweets, remain visible as links), then convert
            # HTML links to markdown links, then strip remaining HTML tags
            def _markdown_image(match):
                tag = match.group(0)
                src_match = re.search(r'src=["\']([^"\']*)["\']', tag, re.IGNORECASE)
                if not src_match:
                    return ''
                src = html.unescape(src_match.group(1))
                if src.startswith('data:'):
                    return ''
                alt_match = re.search(r'alt=["\']([^"\']*)["\']', tag, re.IGNORECASE)
                alt = html.unescape(alt_match.group(1)).strip() if alt_match else ''
                return f"![{alt}]({src})"

            def _markdown_link(match):
                href = html.unescape(match.group(1))
                text = re.sub(r'<[^>]+>', '', match.group(2))
                text = html.unescape(text).strip()
                if not text:
                    return href
                return f"[{text}]({href})"

            clean_content = re.sub(r'<img\b[^>]*>', _markdown_image, content, flags=re.IGNORECASE)
            clean_content = re.sub(
                r'<a\s+[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
                _markdown_link,
                clean_content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            clean_content = re.sub(r'<[^>]+>', '', clean_content)
            clean_content = html.unescape(clean_content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()

            # For X/Twitter articles, describe attached media images with vision so
            # screenshot-based quote-tweets (a screenshot of another post, rather than
            # a native in-platform quote-tweet with real text) aren't opaque to the LLM
            if self.valves.ANALYZE_TWITTER_IMAGES and domain.lower() in ('x.com', 'twitter.com'):
                analyzed_count = 0

                def _annotate_twitter_image(match):
                    nonlocal analyzed_count
                    if analyzed_count >= self.valves.MAX_IMAGES_TO_ANALYZE:
                        return match.group(0)
                    analyzed_count += 1
                    description = self._describe_image(match.group(2))
                    if not description:
                        return match.group(0)
                    return f'{match.group(0)} [Image shows: "{description}"]'

                clean_content = re.sub(
                    r'!\[([^\]]*)\]\((https?://pbs\.twimg\.com/media/[^\)]+)\)',
                    _annotate_twitter_image,
                    clean_content,
                )

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

            if annotations:
                blocks = []
                for a in annotations:
                    quote = (a.get("quote") or "").strip()
                    note = (a.get("text") or "").strip()
                    block = f"> {quote}" if quote else ""
                    if note:
                        block += f"\n\nNote: {note}" if block else f"Note: {note}"
                    if block:
                        blocks.append(block)
                if blocks:
                    result += f"\n\n---\n\n**Your Annotations ({len(blocks)}):**\n\n"
                    result += "\n\n---\n\n".join(blocks)

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

    async def get_articles_by_tag(
        self,
        tag: str,
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        Fetch all Wallabag articles that have a specific tag.

        Uses the Wallabag API to retrieve a complete, exact-match list of every
        article carrying the given tag label. For semantic/fuzzy discovery of
        articles related to a topic, use search_knowledge() with tag= instead.

        Args:
            tag: The tag label to filter by (case-insensitive, exact match)

        Returns:
            A markdown-formatted list of all matching articles with title,
            ID, domain, URL, published date, and full tag list.
        """
        import requests

        MAX_ARTICLES = 500
        PER_PAGE = 30

        if not self.valves.WALLABAG_URL:
            return "Error: Wallabag URL not configured in tool settings"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Fetching articles tagged '{tag.strip()}' from Wallabag..."}
            })

        try:
            token = self._get_wallabag_token()
            url = self.valves.WALLABAG_URL.rstrip('/')
            headers = {"Authorization": f"Bearer {token}"}

            articles = []
            page = 1
            total_pages = 1
            total = 0

            while page <= total_pages and len(articles) < MAX_ARTICLES:
                resp = requests.get(
                    f"{url}/api/entries.json",
                    headers=headers,
                    params={
                        "tags": tag.strip(),
                        "page": page,
                        "perPage": PER_PAGE,
                        "sort": "created",
                        "order": "desc",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                total_pages = data.get("pages", 1)
                total = data.get("total", 0)
                items = data.get("_embedded", {}).get("items", [])
                if not items:
                    break
                articles.extend(items)
                page += 1

            if not articles:
                return f"No articles found with tag **{tag.strip()}**."

            lines = [f"## Articles tagged '{tag.strip()}' ({total} total)\n"]

            for entry in articles[:MAX_ARTICLES]:
                title = entry.get("title") or "Untitled"
                entry_id = entry.get("id")
                domain = entry.get("domain_name") or ""
                entry_url = entry.get("url") or ""
                published = entry.get("published_at") or entry.get("created_at") or ""
                if published and "T" in published:
                    published = published.split("T")[0]
                entry_tags = [t["label"] for t in entry.get("tags", [])]

                line = f"- **{title}** (ID: {entry_id})"
                if domain:
                    line += f" — {domain}"
                if published:
                    line += f" · {published}"
                if entry_url:
                    line += f"\n  {entry_url}"
                if entry_tags:
                    line += f"\n  Tags: {', '.join(entry_tags)}"
                lines.append(line)

            if len(articles) >= MAX_ARTICLES and total > MAX_ARTICLES:
                lines.append(f"\n_(Showing first {MAX_ARTICLES} of {total} articles.)_")

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Retrieved {min(len(articles), MAX_ARTICLES)} articles tagged '{tag.strip()}'"}
                })

            return "\n".join(lines)

        except Exception as e:
            error_msg = f"Error fetching articles by tag '{tag}': {str(e)}"
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

    async def remove_tag_from_article(
        self,
        article_id: int,
        tag: str,
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        Remove a single tag from a Wallabag article.

        Only removes the specified tag — all other tags on the article are left
        unchanged. If the tag is not present on the article, returns an error.

        IMPORTANT: Only call this function when the user explicitly instructs
        you to remove a tag. Never call it proactively or as part of general
        research and summarization.

        Args:
            article_id: The Wallabag article ID (from search results)
            tag: The tag label to remove, e.g. "ai"

        Returns:
            Confirmation with the article title and remaining tag list.
        """
        import requests

        if not self.valves.WALLABAG_URL:
            return "Error: Wallabag URL not configured in tool settings"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Removing tag '{tag}' from article {article_id}..."}
            })

        try:
            token = self._get_wallabag_token()
            url = self.valves.WALLABAG_URL.rstrip('/')
            headers = {"Authorization": f"Bearer {token}"}

            entry_resp = requests.get(
                f"{url}/api/entries/{article_id}.json",
                headers=headers,
                timeout=30
            )
            entry_resp.raise_for_status()
            entry = entry_resp.json()

            tag_label = tag.strip().lower()
            matching_tag = next(
                (t for t in entry.get('tags', []) if t.get('label', '').lower() == tag_label),
                None
            )
            if matching_tag is None:
                title = entry.get('title', f'Article {article_id}')
                current_tags = [t['label'] for t in entry.get('tags', [])]
                return (
                    f"Tag '{tag.strip()}' not found on **{title}**\n"
                    f"Current tags: {', '.join(current_tags) or '(none)'}"
                )

            tag_id = matching_tag['id']
            del_resp = requests.delete(
                f"{url}/api/entries/{article_id}/tags/{tag_id}",
                headers=headers,
                timeout=30
            )
            del_resp.raise_for_status()
            updated_entry = del_resp.json()

            title = updated_entry.get('title', f'Article {article_id}')
            remaining_tags = [t['label'] for t in updated_entry.get('tags', [])]

            result = f"Tag removed from **{title}**\n"
            result += f"Removed: {tag.strip()}\n"
            result += f"Remaining tags: {', '.join(remaining_tags) or '(none)'}"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Tag removed from: {title[:50]}"}
                })

            return result

        except Exception as e:
            error_msg = f"Error removing tag from article {article_id}: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg}
                })
            return error_msg

    @staticmethod
    def _as_bool(value) -> bool:
        """Coerce a model-supplied flag to a real bool.

        Tool calls sometimes deliver booleans as strings ("true", "0", "no"),
        which plain truthiness would read backwards — and reading a read/star
        flag backwards writes the opposite of what the user asked for.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() not in ('false', '0', 'no', 'none', '')

    async def update_article_status(
        self,
        article_ids: str,
        read: bool = None,
        starred: bool = None,
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        Mark Wallabag articles as read/unread and/or starred/unstarred.

        In Wallabag, "read" is the archived state: read=True archives the
        article (clearing it from the unread list), read=False returns it to
        unread. starred=True flags it as a favourite, starred=False clears
        that. At least one of read or starred must be given; both can be set
        in the same call.

        IMPORTANT: Only call this function when the user explicitly instructs
        you to mark articles read/unread or to star/unstar them. Never call it
        proactively, and never as a side-effect of searching, fetching, or
        summarizing articles.

        Args:
            article_ids: One Wallabag article ID, or several separated by
                         commas, e.g. "412" or "412,413,414". IDs come from
                         search_knowledge results (the "Article ID" field) or
                         from get_articles_by_tag.
            read: True marks the article(s) read (archived), False marks them
                  unread. Omit to leave read state untouched.
            starred: True stars the article(s), False unstars them. Omit to
                     leave star state untouched.

        Returns:
            Confirmation with each article's title and resulting state, plus
            any articles that could not be updated.
        """
        import requests

        MAX_ARTICLES = 50

        if not self.valves.WALLABAG_URL:
            return "Error: Wallabag URL not configured in tool settings"

        if read is None and starred is None:
            return (
                "Error: nothing to update — set read (true/false) and/or "
                "starred (true/false)."
            )

        payload = {}
        if read is not None:
            payload['archive'] = 1 if self._as_bool(read) else 0
        if starred is not None:
            payload['starred'] = 1 if self._as_bool(starred) else 0

        ids = []
        for raw in str(article_ids).split(','):
            raw = raw.strip()
            if not raw:
                continue
            try:
                article_id = int(raw)
            except ValueError:
                return (
                    f"Error: '{raw}' is not a valid article ID. Pass one integer ID, "
                    f"or several separated by commas (e.g. \"412,413\")."
                )
            if article_id not in ids:
                ids.append(article_id)

        if not ids:
            return "Error: no article IDs given."
        if len(ids) > MAX_ARTICLES:
            return (
                f"Error: too many article IDs ({len(ids)}). "
                f"Update at most {MAX_ARTICLES} articles per call."
            )

        changes = []
        if 'archive' in payload:
            changes.append('read' if payload['archive'] else 'unread')
        if 'starred' in payload:
            changes.append('starred' if payload['starred'] else 'unstarred')
        change_desc = ' and '.join(changes)

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Marking {len(ids)} article(s) {change_desc}..."}
            })

        try:
            token = self._get_wallabag_token()
        except Exception as e:
            error_msg = f"Error authenticating with Wallabag: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg}
                })
            return error_msg

        url = self.valves.WALLABAG_URL.rstrip('/')
        headers = {"Authorization": f"Bearer {token}"}

        updated = []
        failed = []

        for article_id in ids:
            try:
                resp = requests.patch(
                    f"{url}/api/entries/{article_id}.json",
                    headers=headers,
                    data=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                entry = resp.json()

                state = 'read' if entry.get('is_archived') else 'unread'
                state += ' · starred' if entry.get('is_starred') else ' · not starred'

                updated.append({
                    'id': article_id,
                    'title': entry.get('title') or f'Article {article_id}',
                    'state': state,
                })
            except Exception as e:
                failed.append((article_id, str(e)))

        lines = []
        if len(updated) == 1 and not failed:
            item = updated[0]
            lines.append(f"Updated **{item['title']}** (ID: {item['id']}) — {item['state']}")
        elif updated:
            lines.append(f"Updated {len(updated)} article(s) — {change_desc}:")
            for item in updated:
                lines.append(f"- **{item['title']}** (ID: {item['id']}) — {item['state']}")

        if failed:
            if updated:
                lines.append("")
            lines.append(f"Failed ({len(failed)}):")
            for article_id, err in failed:
                lines.append(f"- ID {article_id}: {err}")

        if __event_emitter__:
            if updated and not failed:
                desc = f"Marked {len(updated)} article(s) {change_desc}"
            elif updated:
                desc = f"Marked {len(updated)} article(s) {change_desc}, {len(failed)} failed"
            else:
                desc = f"Failed to update {len(failed)} article(s)"
            await __event_emitter__({
                "type": "status",
                "data": {"description": desc}
            })

        return "\n".join(lines)

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

    async def get_kindle_highlights(
        self,
        file_name: str,
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        Fetch every saved highlight and annotation for a specific Kindle book.

        Use this after search_knowledge returns Kindle highlight snippets, when you
        need the complete set of highlights from that book rather than just the
        top semantic matches. Note this returns highlighted passages and personal
        annotations only — not the book's full text.

        Args:
            file_name: The Kindle highlights JSON filename from search results
                       (the 'File' field on a search_knowledge Kindle result).

        Returns:
            All highlights from the book with title/author metadata, in reading order
        """
        from urllib.parse import unquote
        import requests

        file_name = unquote(file_name)
        full_url = self._build_static_url(self.valves.KINDLE_HIGHLIGHTS_BASE_URL, file_name)

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Fetching Kindle highlights: {file_name}..."}
            })

        try:
            resp = requests.get(full_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            highlights = data.get("highlights", [])
            result = f"**Kindle Highlights: {data.get('title', file_name)}**\n"
            result += f"Author: {data.get('authors', 'Unknown')}\n"
            result += f"ASIN: {data.get('asin', 'N/A')}\n"
            result += f"Total highlights: {len(highlights)}\n\n---\n\n"

            blocks = []
            for h in highlights:
                location = h.get("location") or {}
                block = f"Location: {location.get('value', 'N/A')}"
                if location.get("url"):
                    block += f"\nKindle Link: {location['url']}"
                text = h.get("text") or ""
                if text:
                    block += f"\n\n{text}"
                if h.get("note"):
                    block += f"\n\nNote: {h['note']}"
                blocks.append(block)

            result += "\n\n---\n\n".join(blocks)

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Retrieved {len(highlights)} highlights"}
                })

            return result

        except requests.exceptions.HTTPError as e:
            error_msg = f"Error fetching Kindle highlights '{file_name}': HTTP {e.response.status_code}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg}
                })
            return error_msg
        except Exception as e:
            error_msg = f"Error fetching Kindle highlights '{file_name}': {str(e)}"
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
        elif source == "summary":
            if payload.get("source_type") == "podcast":
                return f"summary:podcast:{payload.get('show_name', '')}:{payload.get('episode_name', '')}"
            else:
                return f"summary:paper:{payload.get('document_name', payload.get('title', 'unknown'))}"
        else:
            doc_id = payload.get('document_name', payload.get('file_path', payload.get('title', 'unknown')))
            return f"{source}:{doc_id}"

    @staticmethod
    def _dedupe_feed_chunks(points: list) -> list:
        """
        Collapse raw feed chunk points to one point per article.

        Keeps only chunk_index == 0 points (the chunk with the article title
        prepended) — other chunk indices are discarded, never merged. Also
        collapses duplicate chunk_index == 0 points sharing the same article
        key (e.g. from re-ingestion), keeping the first one encountered.
        """
        seen = set()
        deduped = []
        for point in points:
            if point.payload.get("chunk_index", 0) != 0:
                continue
            key = Tools._article_key(point)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(point)
        return deduped

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
        is_summaries = coll_name == self.valves.SUMMARIES_COLLECTION

        if date_mode == "indexed":
            if is_wallabag:
                payload_key = "created_at"
            elif is_feeds:
                payload_key = "published_ts"  # best proxy — feeds have no ingest timestamp
            elif is_podcast or is_doc or is_summaries:
                payload_key = "modified_at"
            else:
                return None  # Kindle — no date fields
        else:  # "published" (default)
            if is_feeds:
                payload_key = "published_ts"
            elif is_wallabag or is_podcast or is_summaries:
                payload_key = "published_at"  # summaries: reliable for podcast-type, sparse for paper-type
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

    def _build_tag_condition(self, coll_name: str, tag: str):
        """Return a FieldCondition matching tag in the tags array, or None if not applicable."""
        from qdrant_client.models import FieldCondition, MatchValue
        has_tags = (
            coll_name == self.valves.WALLABAG_COLLECTION
            or coll_name == self.valves.FEEDS_COLLECTION
            or coll_name == self.valves.PODCAST_COLLECTION
            or coll_name == self.valves.SUMMARIES_COLLECTION
        )
        if not has_tags:
            return None
        return FieldCondition(key="tags", match=MatchValue(value=tag.strip().lower()))

    async def search_knowledge(
        self,
        query: str,
        collection: str = "all",
        date_from: str = None,
        date_to: str = None,
        date_mode: str = "published",
        tag: str = None,
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
                        - 'summaries' for AI-generated podcast/paper summaries only
                          (also included automatically under 'all')
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
            tag: Optional tag label to restrict results to (exact match, case-insensitive).
                 Applies to Wallabag articles, RSS feed articles, podcasts, and AI-generated
                 summaries. Use get_articles_by_tag() instead when you need a complete listing.

        Returns:
            Relevant context from the knowledge base, formatted with source information.
            NOTE: Results labeled as an AI-generated summary are a recall aid, not the
            source itself — never cite their wording as fact; fetch the real content via
            get_full_document() using the result's Source file/Source type fields first.
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
            if collection in ["summaries", "all"]:
                collections.append(self.valves.SUMMARIES_COLLECTION)

            doc_names = self._get_document_collection_names()

            if collection in ["documents", "all"]:
                collections.extend(doc_names)
            elif collection in doc_names:
                collections.append(collection)

            if not collections:
                valid = ["articles", "podcasts", "feeds", "kindle", "summaries", "documents", "all"] + doc_names
                return f"Unknown collection: {collection}. Valid options: {', '.join(valid)}"

            from qdrant_client.models import Filter as QFilter

            all_results = []

            for coll_name in collections:
                fetch_limit = self.valves.TOP_K * 3
                date_filter = self._build_date_filter(coll_name, date_from, date_to, date_mode)
                tag_cond = self._build_tag_condition(coll_name, tag) if tag else None

                if date_filter and tag_cond:
                    combined_filter = QFilter(must=[*date_filter.must, tag_cond])
                elif tag_cond:
                    combined_filter = QFilter(must=[tag_cond])
                else:
                    combined_filter = date_filter  # may be None
                # Fallback omits date but keeps tag (if any)
                fallback_filter = QFilter(must=[tag_cond]) if tag_cond else None

                results = None
                try:
                    results = qdrant.query_points(
                        collection_name=coll_name,
                        query=query_vector,
                        query_filter=combined_filter,
                        limit=fetch_limit
                    )
                    # If the date filter produced no results, retry without it so the
                    # collection still contributes its best semantic matches
                    if not results.points and date_filter is not None:
                        results = qdrant.query_points(
                            collection_name=coll_name,
                            query=query_vector,
                            query_filter=fallback_filter,
                            limit=fetch_limit
                        )
                except Exception as e:
                    if date_filter is not None:
                        # Filtered query failed — retry without the date filter
                        try:
                            results = qdrant.query_points(
                                collection_name=coll_name,
                                query=query_vector,
                                query_filter=fallback_filter,
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
                    if payload.get('file_name'):
                        header += f"\nFile: {payload['file_name']}"
                elif source == "summary":
                    summary_type = payload.get("source_type", "unknown")
                    if summary_type == "podcast":
                        header = (
                            f"**AI Summary — Podcast: {payload.get('show_name', 'Unknown Show')}**\n"
                            f"Episode: {payload.get('episode_name', 'Unknown Episode')}"
                        )
                    elif summary_type == "paper":
                        header = f"**AI Summary — Paper: {payload.get('document_name', 'Unknown Document')}**"
                    else:
                        header = f"**AI Summary: {payload.get('title', 'Untitled')}**"
                    if payload.get('title'):
                        header += f"\nTitle: {payload['title']}"
                    if payload.get('url'):
                        header += f"\nURL: {payload['url']}"
                    if payload.get('tags'):
                        header += f"\nTags: {', '.join(payload['tags'])}"
                    header += f"\nSource type: {summary_type}"
                    if payload.get('source_file'):
                        header += f"\nSource file: {payload['source_file']}"
                    header += (
                        "\nNOTE: This is an AI-generated SUMMARY, not the original source — written "
                        "purely to aid semantic search recall. Do not cite its wording as fact. Before "
                        "presenting specifics, fetch the real content with get_full_document(file_path="
                        "<Source file above>, source_type=<Source type above>), or search the full "
                        "transcript/paper collection instead."
                    )
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

    async def list_recent_feed_articles(
        self,
        days: int = 1,
        date_from: str = None,
        date_to: str = None,
        feed_name: str = None,
        __event_emitter__: Callable[[dict], None] = None,
    ) -> str:
        """
        List every feed article published in a time window — complete and
        deduplicated, with no ranking, categorization, or summarization applied.

        Use this instead of search_knowledge when you need an exhaustive
        inventory of RSS/Atom feed articles for a time range (e.g. to build a
        daily news roundup), rather than a relevance-ranked subset. This is a
        plain Qdrant payload listing (no query embedding, no OpenAI call, no
        semantic ranking) — every matching article is returned exactly once,
        sorted by publication date (newest first). It does not categorize,
        cluster, or summarize results; that is left entirely to the caller.

        Args:
            days: How many days back from today (UTC) to include, e.g. 1 means
                  today and yesterday. Ignored if date_from or date_to is set.
                  Default 1.
            date_from: Optional explicit start of the window, ISO format
                       (YYYY-MM-DD). Overrides `days` when set. If set without
                       date_to, the window runs from date_from through now.
            date_to: Optional explicit end of the window, ISO format
                     (YYYY-MM-DD). Overrides `days` when set. If set without
                     date_from, the window runs from the start of the feed
                     archive through date_to.
            feed_name: Optional feed name to restrict results to (e.g. 'Hacker
                       News'), exact match, case-sensitive. Omit for all feeds.

        Returns:
            A markdown list of every matching feed article — one entry per
            article (chunked articles collapsed to a single entry using the
            chunk with the title prepended), each with title, feed name, URL,
            published date, author, tags, and opening text, sorted newest
            first. Returns a plain "No feed articles found" message if nothing
            matches.
        """
        from datetime import datetime, timedelta, timezone
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter as QFilter, FieldCondition, MatchValue

        SCROLL_PAGE_SIZE = 256
        MAX_POINTS_SCROLLED = 20000  # safety valve, not a feature limit

        if not date_from and not date_to:
            date_from = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Listing feed articles from {date_from or 'inception'} to {date_to or 'now'}..."}
            })

        try:
            qdrant = QdrantClient(
                url=self.valves.QDRANT_URL,
                api_key=self.valves.QDRANT_API_KEY or None
            )

            date_filter = self._build_date_filter(self.valves.FEEDS_COLLECTION, date_from, date_to, "published")

            feed_cond = None
            if feed_name and feed_name.strip():
                feed_cond = FieldCondition(key="feed_name", match=MatchValue(value=feed_name.strip()))

            if date_filter and feed_cond:
                scroll_filter = QFilter(must=[*date_filter.must, feed_cond])
            elif feed_cond:
                scroll_filter = QFilter(must=[feed_cond])
            else:
                scroll_filter = date_filter

            all_points = []
            next_offset = None
            truncated = False
            while True:
                page_points, next_offset = qdrant.scroll(
                    collection_name=self.valves.FEEDS_COLLECTION,
                    scroll_filter=scroll_filter,
                    limit=SCROLL_PAGE_SIZE,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                all_points.extend(page_points)
                if next_offset is None:
                    break
                if len(all_points) >= MAX_POINTS_SCROLLED:
                    truncated = True
                    break

            articles = self._dedupe_feed_chunks(all_points)
            articles.sort(key=lambda p: p.payload.get("published_ts", 0), reverse=True)

            if not articles:
                window_desc = f"{date_from or 'the start of the archive'} to {date_to or 'now'}"
                feed_desc = f" for feed '{feed_name.strip()}'" if feed_name and feed_name.strip() else ""
                return f"No feed articles found{feed_desc} between {window_desc}."

            article_blocks = []
            for point in articles:
                payload = point.payload
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
                text = payload.get('text', '')
                article_blocks.append(f"{header}\n\n{text}\n\n---")

            heading = f"## Feed Articles ({len(articles)} total)"
            if feed_name and feed_name.strip():
                heading += f" — {feed_name.strip()}"

            result = heading + "\n\n" + "\n\n".join(article_blocks)
            if truncated:
                result += (
                    f"\n\n_(Stopped after scanning {MAX_POINTS_SCROLLED} raw points; "
                    f"the window may contain more articles — narrow the date range or "
                    f"feed_name filter to see all of them.)_"
                )

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Found {len(articles)} feed articles"}
                })

            return result

        except Exception as e:
            error_msg = f"Error listing feed articles: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg}
                })
            return error_msg
