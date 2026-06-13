# Qdrant Knowledge Search Tool Usage Guide

## Available Tools

### `search_knowledge`
Search a personal knowledge base of saved Wallabag articles, podcast transcripts, document collections, RSS/Atom news feed articles, and Kindle book highlights stored in Qdrant.

- **When to use**: When the user asks a question that could benefit from their personal saved content — articles they've bookmarked, podcast episodes they've listened to, papers/essays/books they've collected, news articles from their feeds, Kindle highlights from books they've read, or any topic they may have encountered in their reading/listening history.
- **Parameters**:
  - `query` — The search query describing what to look for.
  - `collection` — Which collection to search: `"articles"` for Wallabag only, `"podcasts"` for podcast transcripts only, `"feeds"` for RSS/Atom news feed articles only, `"kindle"` for Kindle book highlights only, `"documents"` for all document collections, a specific collection name (e.g. `"papers"`, `"books"`), or `"all"` for everything (default).
  - `date_from` — Optional start of date range, ISO format (`YYYY-MM-DD`). If omitted with `date_to` set, searches from inception to `date_to`.
  - `date_to` — Optional end of date range, ISO format (`YYYY-MM-DD`). If omitted with `date_from` set, searches from `date_from` to present.
  - `date_mode` — Which date concept to filter on: `"published"` (default) for article/episode publication date; `"indexed"` for when the item was added/saved to the knowledgebase. The tool resolves the correct payload field per collection internally.
  - `tag` — Optional tag label to restrict results to (exact match, case-insensitive). Only filters Wallabag articles, RSS feed articles, and podcasts; Kindle and document collections are unaffected. Use `get_articles_by_tag()` instead when you need a **complete listing** of all articles with a tag rather than a semantic top-K.

### `get_full_article`
Fetch the complete text of a Wallabag article by its ID.

- **When to use**: After `search_knowledge` returns relevant snippets from a Wallabag article and the user wants the full content for deeper reading or summarization.
- **Parameters**: An integer `article_id` (obtained from `search_knowledge` results).

### `list_wallabag_tags`
List all tags used in Wallabag, sorted alphabetically, with article counts per tag (when the Wallabag instance supports it, v2.5.2+).

- **When to use**: When the user asks what tags exist in their Wallabag, wants to browse or explore their saved articles by topic, or asks how many articles they have saved under a particular subject.
- **Parameters**: None.

### `get_articles_by_tag`
Fetch all Wallabag articles that carry a specific tag, using the Wallabag API for a complete, exact-match listing.

- **When to use**: When the user asks for all articles tagged with something (e.g. "show me everything tagged 'ai'", "list my articles tagged 'recipe'"). This returns the full set — not a semantic top-K — so prefer it over `search_knowledge` when completeness matters more than relevance ranking.
- **Parameters**:
  - `tag` — The tag label to filter by (case-insensitive, exact match). One tag per call.
- **Notes**: Paginates automatically through all results (up to 500 articles). Results include article ID, title, domain, URL, published date, and all tags on the article. Requires Wallabag credentials to be configured.

### `add_tag_to_article`
Add a single tag to a Wallabag article.

- **When to use**: **ONLY** when the user explicitly tells you to add a tag to a specific article. Never call this proactively, never as a side-effect of searching or summarizing, and never to "helpfully" organize articles on your own initiative. This function modifies stored data — use it only on direct instruction.
- **Parameters**:
  - `article_id` — The integer Wallabag article ID (from search results).
  - `tag` — A single tag label to add (e.g. `"ai"`). One tag per call. Wallabag will create the tag if it doesn't already exist.
- **Notes**: Only adds — never removes or replaces existing tags. The response confirms the article title and shows the full resulting tag list after the addition.

### `remove_tag_from_article`
Remove a single tag from a Wallabag article.

- **When to use**: **ONLY** when the user explicitly tells you to remove a tag from a specific article. Never call this proactively or as a side-effect of other operations.
- **Parameters**:
  - `article_id` — The integer Wallabag article ID (from search results).
  - `tag` — The tag label to remove (e.g. `"ai"`). One tag per call. Case-insensitive.
- **Notes**: Only removes the named tag — all other tags are left unchanged. Returns an error (with the current tag list) if the tag is not present on the article. The response confirms the article title and shows the remaining tags after removal.

### `get_full_document`
Fetch the full text of a document or podcast transcript from the static file server.

- **When to use**: After `search_knowledge` returns relevant snippets from a document or podcast and the user wants the complete text for deeper analysis or summarization.
- **Parameters**:
  - `file_path` — The relative file path from search results (e.g., `"paper_name.md"` or `"ShowName/Episode.txt"`). Use the raw path with normal spaces — do NOT use URL-encoded paths from transcript/audio URLs.
  - `source_type` — The **collection name** from search results (the `Collection` field, e.g. `"papers"`, `"books"`). For podcasts use `"podcasts"`. The collection name determines the folder in the URL. Default: `"papers"`.

## Configuration (Valves)

Key settings that must be configured for the tool to work:

- **QDRANT_URL** — Qdrant server address (default: `http://host.docker.internal:6333`)
- **QDRANT_API_KEY** — Qdrant API key if authentication is enabled
- **OPENAI_API_KEY** — Required for generating query embeddings via `text-embedding-3-small`
- **TOP_K** (default: 8) — Total number of results to return
- **PER_ARTICLE_MAX** (default: 2) — Preferred max results from any single article/episode, to keep results diverse
- **WALLABAG_COLLECTION** / **PODCAST_COLLECTION** / **FEEDS_COLLECTION** / **KINDLE_COLLECTION** — Qdrant collection names for Wallabag, podcasts, RSS/Atom feeds, and Kindle highlights (defaults: `wallabag_articles`, `podcast_transcripts`, `news_feeds`, `kindle_highlights`)
- **DOCUMENT_COLLECTIONS** — Comma-separated Qdrant collection names for document collections (e.g., `"papers,books,manuals"`)
- **DOCUMENT_COLLECTIONS_BASE_URL** — Base URL for document collections; files are served at `{base_url}/{collection_name}/...` (default: `https://static-lan.maddock.net`)
- **PODCASTS_BASE_URL** — Base URL for podcast files on the static file server (default: `https://static-lan.maddock.net/podcasts`)
- **Wallabag credentials** (URL, client ID, client secret, username, password) — Required for `get_full_article`, `list_wallabag_tags`, and `get_articles_by_tag`

## Usage Patterns

### Answering questions from personal context
When the user asks something that sounds like it could draw on their saved reading or listening:

1. Call `search_knowledge` with a descriptive query.
2. Review the returned snippets — they include source metadata (article title, domain, URL, podcast show/episode name).
3. Synthesize an answer from the relevant snippets, citing sources.

### Narrowing by collection
If the user specifically mentions articles, reading, or bookmarks, use `collection="articles"`. If they mention podcasts or episodes, use `collection="podcasts"`. If they mention news, feeds, or recent articles from a specific publication, use `collection="feeds"`. If they mention Kindle highlights, book highlights, or annotations, use `collection="kindle"`. If they mention papers, essays, books, or reference documents, use `collection="documents"` or a specific collection name like `"papers"`. Default to `"all"` when unsure.

### Linking to Wallabag articles
When the user asks for a link to a Wallabag article, use the article ID from search results to construct the URL:

`https://walla.maddock.net/view/{wallabag_article_id}`

For example, if the search result shows `Article ID: 1234`, the link is `https://walla.maddock.net/view/1234`.

### Getting full content
If a search snippet is promising but incomplete:

- **For Wallabag articles**: Note the `Article ID` from the search results, then call `get_full_article` with that ID.
- **For documents or podcasts**: Note the `File` path and `Collection` name from the search results, then call `get_full_document` with that `file_path` and the `Collection` value as `source_type`. The collection name determines which folder the file is served from.

### Exploring saved content by tag
If the user asks "what tags do I have?" or "what topics have I saved?" or wants to browse their Wallabag by subject:

1. Call `list_wallabag_tags` — no parameters needed.
2. Present the tag list. If article counts are included, mention which tags have the most content.
3. If the user wants to explore a specific tag, follow up with `get_articles_by_tag` or `search_knowledge` depending on intent (see below).

### Retrieving all articles with a tag
When the user asks for every article with a specific tag ("show me all articles tagged X", "list articles tagged X"):

- Call `get_articles_by_tag(tag="X")` — returns a complete paginated list direct from the Wallabag API.

### Semantic search within a tag
When the user wants to find the most relevant content about a topic, scoped to articles that carry a specific tag ("search my 'ai' articles for transformer architecture"):

- Call `search_knowledge(query="transformer architecture", collection="articles", tag="ai")` — runs vector search restricted to articles carrying that tag.
- The `tag` filter applies to Wallabag articles, RSS feed articles, and podcasts; Kindle and document collections are unaffected and contribute their normal semantic results.

### Filtering by date or time range
When the user asks about content from a specific period (e.g. "articles I saved last month", "podcasts from 2023", "news from this week"):

- Use `date_from` and/or `date_to` with ISO dates (`YYYY-MM-DD`). Either bound may be omitted.
- Default `date_mode="published"` filters on the **publication date** of the article/episode — use this when the user refers to when something was *written* or *published*.
- Use `date_mode="indexed"` only when the user explicitly asks about when something was **added** or **saved** to the knowledgebase (e.g. "articles I saved in January", "what did I bookmark recently").
- The tool resolves the correct payload field per collection automatically — you never need to know field names.
- If a collection has no results in the requested window, the tool automatically falls back to unfiltered results for that collection, so you always get something.
- Kindle highlights have no date fields and are never filtered by date.

Examples:
- "What have I saved about AI this year?" → `date_from="2025-01-01"`, `date_mode="published"`
- "What did I bookmark last week?" → `date_from="2025-05-24"`, `date_to="2025-05-31"`, `date_mode="indexed"`
- "Articles published before 2020 about climate" → `date_to="2019-12-31"`

### Broad research across personal knowledge
If the user asks a broad question like "what have I saved about climate policy":

1. Call `search_knowledge` with a relevant query.
2. Results are diversified — the tool limits results per article/episode so you get breadth across different sources rather than multiple chunks from the same piece.
3. If a particular source looks especially relevant, use `get_full_article` (for Wallabag articles) or `get_full_document` (for documents/podcasts) to go deeper.

## Interpreting Results

- Results include a relevance score. Higher-scored results appear first.
- Wallabag results show: article title, article ID, source domain, URL, and tags.
- Podcast results show: show name, episode name, transcript URL, and audio URL.
- Feed results show: article title, feed name, article URL, published date, author, and tags. The article URL links directly to the original source — no `get_full_document` call is needed; direct the user to the URL if they want the full content.
- Kindle results show: book title, author, location value, and an optional Kindle deep link. Each result is a highlight (annotation) from a Kindle book.
- Document results show: document name, collection name, file path, and a URL to the original file (if a base URL is configured for that collection). The `Collection` field indicates which Qdrant collection the result came from and should be used as the `source_type` when calling `get_full_document`.
- Each result contains a text snippet — the most relevant chunk from the original content.
- If no results are found, the topic likely wasn't covered in the user's saved content. Say so clearly rather than speculating.

## Edge Cases

- If `search_knowledge` returns no results, the user may not have saved anything on that topic. Suggest they try broader terms or a different collection.
- `get_full_article` and `list_wallabag_tags` require Wallabag credentials to be configured. If they aren't, the tool will return an error.
- `list_wallabag_tags` may not include article counts on older Wallabag instances (pre-2.5.2); in that case, the list still shows all tag names.
- `get_full_document` requires `DOCUMENT_COLLECTIONS_BASE_URL` to be configured. If not set, it will return an error for document collections.
- Podcast transcript quality depends on the upstream transcription. Some results may contain transcription artifacts.
- Document results depend on the quality of the .md/.txt conversion from the original format. Some converted documents may have formatting artifacts.
- **Important**: When calling `get_full_document`, use the raw `file_path` value from search result metadata — the path with normal spaces and characters (e.g., `Show Name/Episode.txt`). Do NOT copy the path from a Transcript or Audio URL, as those are URL-encoded (e.g., `Show%20Name/Episode.txt`) and will cause a 404 error due to double-encoding.
