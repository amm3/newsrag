# Qdrant Knowledge Search Tool Usage Guide

## Available Tools

### `search_knowledge`
Search a personal knowledge base of saved Wallabag articles, podcast transcripts, document collections, RSS/Atom news feed articles, Kindle book highlights, and AI-generated podcast/paper summaries stored in Qdrant.

- **When to use**: When the user asks a question that could benefit from their personal saved content — articles they've bookmarked, podcast episodes they've listened to, papers/essays/books they've collected, news articles from their feeds, Kindle highlights from books they've read, or any topic they may have encountered in their reading/listening history.
- **Parameters**:
  - `query` — The search query describing what to look for.
  - `collection` — Which collection to search: `"articles"` for Wallabag only, `"podcasts"` for podcast transcripts only, `"feeds"` for RSS/Atom news feed articles only, `"kindle"` for Kindle book highlights only, `"summaries"` for AI-generated podcast/paper summaries only (see [Understanding AI-generated summaries](#understanding-ai-generated-summaries) — already included automatically under `"all"`), `"documents"` for all document collections, a specific collection name (e.g. `"papers"`, `"books"`), or `"all"` for everything (default).
  - `days` — Look back this many days from now. Use this for relative phrasing like "last 3 days" or "this week" — the date arithmetic happens inside the tool, not in your head. Ignored if `date_from` or `date_to` is set. Default `0` (no window).
  - `date_from` — Optional start of date range, ISO format (`YYYY-MM-DD`). If omitted with `date_to` set, searches from inception to `date_to`. Ignored if `days` is set.
  - `date_to` — Optional end of date range, ISO format (`YYYY-MM-DD`). If omitted with `date_from` set, searches from `date_from` to present.
  - `date_mode` — Which date concept to filter on: `"published"` (default) for article/episode publication date; `"indexed"` for when the item was added/saved to the knowledgebase. The tool resolves the correct payload field per collection internally. In `"published"` mode, items with no publication date (e.g. saved social media posts) fall back to matching on their ingest/save date instead, so a date window doesn't silently exclude undated saved items.
  - `tag` — Optional tag label to restrict results to (exact match, case-insensitive). Only filters Wallabag articles, RSS feed articles, podcasts, and AI-generated summaries; Kindle and document collections are unaffected. Use `get_articles_by_tag()` instead when you need a **complete listing** of all articles with a tag rather than a semantic top-K.
  - `author` — Optional author/byline filter. **Only article text is semantically searchable — author names live in metadata, so putting a name in `query` will not find that person's writing.** Matches the stored name as given plus a title-cased variant. Applies to Wallabag articles, feed articles, Kindle books, and documents; podcasts and summaries have no author field.
  - `domain` — Optional publication-source filter, same metadata caveat as `author`. For Wallabag it's the source host and accepts a bare host or a full URL (`"facebook.com"`, `"www.facebook.com"`, and `"https://facebook.com/x"` all work, and a bare host also matches `www.`/`m.` variants); for feeds it's the feed name; for podcasts/summaries it's the show name. Kindle and document collections have no source field.
  - `sort` — `"relevance"` (default): Cohere-reranked order. `"recent"`: newest-first by date instead, drawn from a wider semantic candidate pool, skipping the relevance rerank. Use `"recent"` whenever the user says "recent", "latest", "last N days", or "newest first" and time ordering matters more than topical rank — e.g. "recent Facebook posts by X". Still a semantic top-K, not an exhaustive chronological listing; for a complete inventory over a date window, use `list_recent_feed_articles` (feeds) or `list_recent_items` (any other collection) instead.

### `list_recent_feed_articles`
List every RSS/Atom feed article published in a time window — complete and deduplicated, with no ranking, categorization, or summarization applied.

- **When to use**: When you need an exhaustive inventory of feed articles for a time range (e.g. building a daily news roundup, or any downstream pipeline like an n8n workflow that does its own categorization/summarization) rather than a relevance-ranked subset. Use this instead of `search_knowledge` whenever completeness matters more than relevance — it's a plain Qdrant payload listing (no query embedding, no OpenAI call).
- **Parameters**:
  - `days` — How many days back from today (UTC) to include, e.g. `1` means today and yesterday. Ignored if `date_from` or `date_to` is set. Default `1`.
  - `date_from` — Optional explicit start of the window, ISO format (`YYYY-MM-DD`). Overrides `days` when set.
  - `date_to` — Optional explicit end of the window, ISO format (`YYYY-MM-DD`). Overrides `days` when set.
  - `feed_name` — Optional feed name to restrict results to (exact match, case-sensitive). Omit for all feeds.
- **Notes**: Articles are chunked at ingest time; this tool collapses each article's chunks down to a single entry (using the chunk with the title prepended), so you always get one row per article, never one per chunk. Results are sorted newest first and are uncapped — no `TOP_K`-style truncation — aside from an internal safety cap (20,000 raw chunk points scanned) that only exists to bound worst-case memory/latency on a badly misconfigured date range; it will not affect normal daily-window usage. Does not fetch full article text, categorize, cluster, or summarize — only title, feed name, URL, published date, author, tags, and the article's opening text (~1000 chars).

### `list_recent_items`
List every item in a single collection in a time window — complete and deduplicated, with no ranking, categorization, or summarization applied. Generalizes `list_recent_feed_articles` to Wallabag, podcasts, and AI-generated summaries as well as feeds.

- **When to use**: When you need an exhaustive inventory of one collection for a time range — e.g. "everything I saved to Wallabag this week", "every podcast I listened to in the last 7 days" — rather than a relevance-ranked subset. Same plain-Qdrant-listing approach as `list_recent_feed_articles` (no query embedding, no OpenAI call, no semantic ranking). For feeds specifically, prefer `list_recent_feed_articles` — it supports `feed_name` and its output format is depended on by an existing n8n workflow.
- **Parameters**:
  - `collection` — Which collection to list: `"articles"` for Wallabag (default), `"podcasts"`, `"feeds"`, `"summaries"`, or a specific document collection name (e.g. `"papers"`). **`"kindle"` is not supported** — Kindle highlights have no date fields to window on; use `get_kindle_highlights` or `search_knowledge(collection="kindle", ...)` instead.
  - `days` — How many days back from today (UTC) to include. Ignored if `date_from` or `date_to` is set. Default `7` (a week — matches the tool's primary use case of a weekly inventory).
  - `date_from` / `date_to` — Optional explicit window bounds, ISO format (`YYYY-MM-DD`). Override `days` when set.
  - `date_mode` — `"indexed"` (default) filters on when the item was added/saved to the knowledgebase — the right mode for "what did I save/listen to this week", and always uses a reliably-populated timestamp. `"published"` filters on article/episode publication date instead, falling back to the ingest/save date for undated items (this fallback now correctly catches Wallabag saves with no publish date, not just genuinely-missing dates).
  - `tag` — Optional tag filter (exact match, case-insensitive). Applies to Wallabag, feeds, podcasts, and summaries; returns an error (not a silent no-op) if given for a document collection, which has no tag field.
  - `source_type` — For `collection="summaries"` only: `"podcast"` or `"paper"`, to restrict to summaries of one subtype. Returns an error if given for any other collection.
  - `text_chars` — Optional max characters to show from each item's body text, truncated with a trailing note. `None` (default) returns full text.
  - `max_items` — Optional cap on the number of items returned, applied **after** sorting so it keeps the most recent N, not an arbitrary N. `None` (default) returns everything matched.
- **Notes**: Results are sorted newest first and use the same per-source header fields as `search_knowledge` (title, ID/URL, tags, dates, etc.) — see [Interpreting Results](#interpreting-results). Uncapped by default aside from the same internal 20,000-raw-point safety valve `list_recent_feed_articles` uses. Podcast dates are lower precision than other sources: `published_at` (when present) is a filename-derived date with no time component, and `modified_at` is the transcript file's local modification time with no UTC offset — so date-window edges can be off by a few hours for podcasts specifically.

### `get_full_article`
Fetch the complete text of a Wallabag article by its ID.

- **When to use**: After `search_knowledge` returns relevant snippets from a Wallabag article and the user wants the full content for deeper reading or summarization.
- **Parameters**: An integer `article_id` (obtained from `search_knowledge` results).
- **Images**: Embedded images are preserved as markdown (`![alt](src)`). For articles from `x.com`/`twitter.com`, any attached media image is automatically described with vision and annotated inline as `[Image shows: "..."]` — this covers screenshot-based quote-tweets (a screenshot of another post rather than a native in-platform quote-tweet with real text), photos, charts, and memes attached to the tweet. Treat this as you would any AI-derived content (see Edge Cases).
- **Annotations**: If the user highlighted passages in Wallabag (with or without an attached note), those appear in a "Your Annotations" section after the article body — each as the quoted passage plus, if present, a `Note:` line. Missing when the article has no annotations.

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

### `update_article_status`
Mark one or more Wallabag articles read/unread and/or starred/unstarred.

- **When to use**: **ONLY** when the user explicitly tells you to mark an article read/unread, or to star/unstar it. Never call this proactively, and never as a side-effect of searching, fetching, or summarizing — reading an article's content in chat does not mean the user wants it marked read in Wallabag.
- **Parameters**:
  - `article_ids` — One integer Wallabag article ID, or several separated by commas (e.g. `"412"` or `"412,413,414"`), up to 50 per call.
  - `read` — `true` marks the article(s) read, `false` marks them unread. Omit to leave read state untouched.
  - `starred` — `true` stars the article(s), `false` unstars them. Omit to leave star state untouched.
  - At least one of `read` or `starred` must be given; both can be set in the same call, applied to every ID in `article_ids`.
- **Notes**: "Read" in Wallabag is the *archived* state — `read=true` archives the article (clears it from the unread list), `read=false` returns it to unread. There is no separate archive concept to worry about. Each ID is updated independently, so one bad ID (deleted, mistyped) doesn't block the rest of the batch — the response lists successes and failures separately, each with the article's resulting state (e.g. `read · starred`) taken from Wallabag's response, not just echoed back from the request.

### `get_full_document`
Fetch the full text of a document or podcast transcript from the static file server.

- **When to use**: After `search_knowledge` returns relevant snippets from a document or podcast and the user wants the complete text for deeper analysis or summarization.
- **Parameters**:
  - `file_path` — The relative file path from search results (e.g., `"paper_name.md"` or `"ShowName/Episode.txt"`). Use the raw path with normal spaces — do NOT use URL-encoded paths from transcript/audio URLs.
  - `source_type` — The **collection name** from search results (the `Collection` field, e.g. `"papers"`, `"books"`). For podcasts use `"podcasts"`. The collection name determines the folder in the URL. Default: `"papers"`.
  - For AI-generated summary results specifically, use the result's own `Source file` and `Source type` fields directly (`source_type` will be `"podcast"` or `"paper"`) — these resolve the same way podcast/document source types already do, no translation needed.

### `get_kindle_highlights`
Fetch every saved highlight and annotation for a specific Kindle book.

- **When to use**: After `search_knowledge` returns a Kindle highlight snippet and the user wants the complete set of highlights from that book, not just the top semantic matches. This returns highlighted passages and personal annotations only — **not** the book's full text.
- **Parameters**: A `file_name` — the Kindle highlights JSON filename from search results (the `File` field on a Kindle result).

## Configuration (Valves)

Key settings that must be configured for the tool to work:

- **QDRANT_URL** — Qdrant server address (default: `http://host.docker.internal:6333`)
- **QDRANT_API_KEY** — Qdrant API key if authentication is enabled
- **OPENAI_API_KEY** — Required for generating query embeddings via `text-embedding-3-small`, and reused by `get_full_article` for vision analysis of X/Twitter images
- **ANALYZE_TWITTER_IMAGES** (default: `true`) — Automatically describe images attached to x.com/twitter.com articles in `get_full_article` using vision
- **VISION_MODEL** (default: `gpt-4o-mini`) — OpenAI vision-capable model used to describe X/Twitter images
- **MAX_IMAGES_TO_ANALYZE** (default: 3) — Maximum number of attached images to run through vision analysis per `get_full_article` call
- **TOP_K** (default: 8) — Total number of results to return
- **PER_ARTICLE_MAX** (default: 2) — Preferred max results from any single article/episode, to keep results diverse
- **WALLABAG_COLLECTION** / **PODCAST_COLLECTION** / **FEEDS_COLLECTION** / **KINDLE_COLLECTION** / **SUMMARIES_COLLECTION** — Qdrant collection names for Wallabag, podcasts, RSS/Atom feeds, Kindle highlights, and AI-generated summaries (defaults: `wallabag_articles`, `podcast_transcripts`, `news_feeds`, `kindle_highlights`, `summaries`)
- **DOCUMENT_COLLECTIONS** — Comma-separated Qdrant collection names for document collections (e.g., `"papers,books,manuals"`)
- **DOCUMENT_COLLECTIONS_BASE_URL** — Base URL for document collections; files are served at `{base_url}/{collection_name}/...` (default: `https://static-lan.maddock.net`)
- **PODCASTS_BASE_URL** — Base URL for podcast files on the static file server (default: `https://static-lan.maddock.net/podcasts`)
- **KINDLE_HIGHLIGHTS_BASE_URL** — Base URL for Kindle highlight JSON files on the static file server, used by `get_kindle_highlights` (default: `https://static-lan.maddock.net/kindle_highlights`)
- **Wallabag credentials** (URL, client ID, client secret, username, password) — Required for `get_full_article`, `list_wallabag_tags`, `get_articles_by_tag`, and `update_article_status`

## Usage Patterns

### Answering questions from personal context
When the user asks something that sounds like it could draw on their saved reading or listening:

1. Call `search_knowledge` with a descriptive query.
2. Review the returned snippets — they include source metadata (article title, domain, URL, podcast show/episode name).
3. Synthesize an answer from the relevant snippets, citing sources.

### Narrowing by collection
If the user specifically mentions articles, reading, or bookmarks, use `collection="articles"`. If they mention podcasts or episodes, use `collection="podcasts"`. If they mention news, feeds, or recent articles from a specific publication, use `collection="feeds"`. If they mention Kindle highlights, book highlights, or annotations, use `collection="kindle"`. If they mention papers, essays, books, or reference documents, use `collection="documents"` or a specific collection name like `"papers"`. Default to `"all"` when unsure — this already includes AI-generated summaries; only use `collection="summaries"` explicitly if the user specifically wants to browse summaries themselves (rare — see [Understanding AI-generated summaries](#understanding-ai-generated-summaries)).

### Linking to Wallabag articles
When the user asks for a link to a Wallabag article, use the article ID from search results to construct the URL:

`https://walla.maddock.net/view/{wallabag_article_id}`

For example, if the search result shows `Article ID: 1234`, the link is `https://walla.maddock.net/view/1234`.

### Getting full content
If a search snippet is promising but incomplete:

- **For Wallabag articles**: Note the `Article ID` from the search results, then call `get_full_article` with that ID.
- **For documents or podcasts**: Note the `File` path and `Collection` name from the search results, then call `get_full_document` with that `file_path` and the `Collection` value as `source_type`. The collection name determines which folder the file is served from.
- **For AI-generated summaries**: Note the `Source file` and `Source type` fields from the result (not `File`/`Collection` — a summary points back to a *different* underlying file). Call `get_full_document` with `file_path=<Source file>` and `source_type=<Source type>` to retrieve the actual transcript/paper the summary was generated from. Do this — or re-run `search_knowledge` against the full-text collection — before presenting any specifics from a summary as fact.
- **For Kindle highlights**: Note the `File` field from a Kindle search result, then call `get_kindle_highlights` with that `file_name` to get every highlight and annotation saved from that book (not the book's full text — just what was highlighted).

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
- The `tag` filter applies to Wallabag articles, RSS feed articles, podcasts, and AI-generated summaries; Kindle and document collections are unaffected and contribute their normal semantic results.

### Filtering by date or time range
When the user asks about content from a specific period (e.g. "articles I saved last month", "podcasts from 2023", "news from this week", "the last 3 days"):

- For **relative** windows ("last N days", "this week"), use `days` — the tool computes the date itself. Don't compute an absolute `date_from` from today's date yourself; you may not know today's date, and guessing produces a wrong or stale window.
- For **explicit calendar** windows, use `date_from` and/or `date_to` with ISO dates (`YYYY-MM-DD`). Either bound may be omitted. These take precedence over `days` if both are somehow given.
- Default `date_mode="published"` filters on the **publication date** of the article/episode — use this when the user refers to when something was *written* or *published*. Items with no publication date (saved social media posts, for example) fall back to matching on their ingest/save date, so they aren't silently excluded from a "published" window.
- Use `date_mode="indexed"` only when the user explicitly asks about when something was **added** or **saved** to the knowledgebase (e.g. "articles I saved in January", "what did I bookmark recently").
- The tool resolves the correct payload field per collection automatically — you never need to know field names.
- **When any filter is in effect — date, tag, author, or domain — every result you get back genuinely satisfies it.** A collection with no matches contributes nothing rather than falling back, and a collection lacking the filtered field entirely (Kindle has no dates; podcasts have no author) is skipped. A prepended note names each skipped collection and the reason. This means a filtered search can legitimately return fewer results than usual — that's the filter doing its job, not a failure.
- If *nothing* matched the window, the tool says so explicitly instead of returning stale results. Report that honestly to the user: the content may simply not exist in that period. You may then re-search without the date range to check — but if you do, say clearly that those results are from outside the window they asked about. Never present out-of-window content as recent.
- Kindle highlights have no date fields, so they're excluded from any date-scoped search. Drop the date range if you need them.

Examples:
- "What have I saved about AI this year?" → `date_from="2026-01-01"`, `date_mode="published"`
- "What did I bookmark in the last 3 days?" → `days=3`, `date_mode="indexed"`
- "Articles published before 2020 about climate" → `date_to="2019-12-31"`

### Finding content by author or source
**Only the article text is embedded and searchable.** Author names, source domains, feed names, and show names live in metadata, which the semantic search and the reranker never see. Putting "Kenneth Tanner" or "Facebook" in `query` will not find his posts — the search has no signal to match on and returns arbitrary content instead. Use the metadata filters:

```
search_knowledge(query="recent posts", domain="facebook.com", days=3, sort="recent")
search_knowledge(query="theology", author="Kenneth Tanner")
```

The `query` still drives ranking among whatever survives the filter, so it can be a loose topic hint — the filter is doing the real work of finding the right items. Combine freely with `days`, `tag`, and `sort`.

If the tool reports no matches, it names every filter it applied. That message means the filter values didn't match the stored metadata — not necessarily that the content is absent. Try the filters one at a time to find which one excluded everything, and report that honestly rather than falling back to unrelated results.

### Recent-first queries
When the user cares more about recency than topical relevance — "recent", "latest", "newest", "last N days" — combine `days` (or explicit dates) with `sort="recent"`:

```
search_knowledge(query="Kenneth Tanner", collection="articles", days=3, sort="recent")
```

This narrows to a 3-day window *and* orders the results newest-first instead of by Cohere relevance, so "his last few posts" actually returns his last few posts rather than whichever ones happened to rank highest semantically. `sort="recent"` alone (no `days`) is also valid — useful for "what's the latest thing I've saved about X" without a hard cutoff.

This is still a semantic top-K search — a wider candidate pool than usual, but capped, not exhaustive. If the user wants a **complete** inventory of everything in a window (e.g. "list every feed article from today" for a roundup), use `list_recent_feed_articles` (feeds) or `list_recent_items` (any other collection) instead — neither has ranking or truncation.

### Broad research across personal knowledge
If the user asks a broad question like "what have I saved about climate policy":

1. Call `search_knowledge` with a relevant query.
2. Results are diversified — the tool limits results per article/episode so you get breadth across different sources rather than multiple chunks from the same piece.
3. If a particular source looks especially relevant, use `get_full_article` (for Wallabag articles), `get_full_document` (for documents/podcasts), or `get_kindle_highlights` (for Kindle books) to go deeper.

### Understanding AI-generated summaries
The `summaries` collection (folded into `collection="all"`) holds AI-generated synthesis of podcast transcripts and papers — one summary per source file, written to describe themes and concepts rather than recap chronologically. They exist purely to widen semantic search recall: a summary may phrase an idea differently than the original transcript or paper, so a query can surface relevant content even when it shares no exact wording with the source.

**Critical: never treat a summary's text as ground truth.** It is a retrieval aid, not a citable fact. Whenever a `search_knowledge` result is an AI summary:

1. Use it to identify that the underlying podcast episode or paper is relevant.
2. Before stating specifics to the user, retrieve the real content — call `get_full_document(file_path=<result's "Source file">, source_type=<result's "Source type">)`, or run `search_knowledge` again against the full-text collection (`collection="podcasts"` or the papers/documents collection) — and cite that instead.
3. Only fall back to paraphrasing the summary itself if the full source is unavailable, and if you do, tell the user explicitly it's from an AI-generated summary, not the original text.

## Interpreting Results

- Results include a relevance score. Higher-scored results appear first.
- Wallabag results show: article title, article ID, source domain, URL, and tags.
- Podcast results show: show name, episode name, transcript URL, and audio URL.
- Feed results show: article title, feed name, article URL, published date, author, and tags. The article URL links directly to the original source — no `get_full_document` call is needed; direct the user to the URL if they want the full content.
- Kindle results show: book title, author, location value, an optional Kindle deep link, and the `File` field (the source JSON filename, usable with `get_kindle_highlights`). Each result is a highlight (annotation) from a Kindle book.
- Document results show: document name, collection name, file path, and a URL to the original file (if a base URL is configured for that collection). The `Collection` field indicates which Qdrant collection the result came from and should be used as the `source_type` when calling `get_full_document`.
- Summary results show: an explicit "AI Summary" label, podcast (show/episode) or paper (document name) identity, title/URL/tags when available, and the `Source file`/`Source type` needed to fetch the real content. **These are AI-generated synthesis, not the original text — never present their content as fact**; see [Understanding AI-generated summaries](#understanding-ai-generated-summaries).
- Each result contains a text snippet — the most relevant chunk from the original content.
- If no results are found, the topic likely wasn't covered in the user's saved content. Say so clearly rather than speculating.
- Every result that has a known date carries one or more labeled date lines: `Published:` (article/episode publication date), `Saved:` (Wallabag ingest date, shown when there's no publication date — e.g. saved social media posts), or `Indexed:` (ingest/save date for podcasts, summaries, and document collections). When both a publication and ingest date exist, `Published:` comes first. Use these — not the result's position in the list — to judge how recent something actually is. Kindle results carry no date line at all.
- A response may be prefixed with `_Note: ..._` lines naming collections that were skipped — because they had no matches for the filters, because they lack the filtered field (Kindle has no dates; podcasts have no author), or because the search failed. These are informational: the results below them all still satisfy the filters. They tell you what *wasn't* searched, which matters when judging how complete an answer is.

### Citing results
Every specific article, episode, or document you name in a response must be a markdown link (`[Title](URL)`), not a bare title or a `(Source, Date)` parenthetical — this applies to every item you mention, including ones offered only as corroborating or additional context, not just the ones you quote directly or fetch in full. Build the URL per the result type above: Wallabag → `https://walla.maddock.net/view/{article_id}`; feed articles → the article URL as-is; Kindle → the deep link when available; documents → the document URL when a base URL is configured. Do this yourself in the response text — don't rely on the platform's own citation/footnote panel to surface a source for you, since it doesn't reliably pick up every tool result.

## Edge Cases

- If `search_knowledge` returns no results, the user may not have saved anything on that topic. Suggest they try broader terms or a different collection.
- `get_full_article` and `list_wallabag_tags` require Wallabag credentials to be configured. If they aren't, the tool will return an error.
- `list_wallabag_tags` may not include article counts on older Wallabag instances (pre-2.5.2); in that case, the list still shows all tag names.
- `get_full_document` requires `DOCUMENT_COLLECTIONS_BASE_URL` to be configured. If not set, it will return an error for document collections.
- `search_knowledge` and `get_articles_by_tag` results never include read or starred state — that data isn't indexed in Qdrant. Don't guess whether an article is already read/starred; if it matters, ask the user or just perform the requested `update_article_status` call and let its response (drawn from Wallabag itself) confirm the resulting state.
- Podcast transcript quality depends on the upstream transcription. Some results may contain transcription artifacts.
- Document results depend on the quality of the .md/.txt conversion from the original format. Some converted documents may have formatting artifacts.
- AI-generated summaries synthesize themes conceptually rather than recapping verbatim — they may use terminology the original source never used. This is intentional (it's what makes them useful for recall), but their wording should never be quoted as if it were the source's own words.
- **Important**: When calling `get_full_document`, use the raw `file_path` value from search result metadata — the path with normal spaces and characters (e.g., `Show Name/Episode.txt`). Do NOT copy the path from a Transcript or Audio URL, as those are URL-encoded (e.g., `Show%20Name/Episode.txt`) and will cause a 404 error due to double-encoding.
- Vision-generated `[Image shows: "..."]` annotations in `get_full_article` (for X/Twitter images) are AI-derived, not guaranteed verbatim — treat any transcribed quote text as paraphrase-accurate rather than pixel-perfect when precision matters. If vision analysis fails or is disabled, the image still appears as a plain `![alt](src)` markdown link with no annotation.
- **Only `text` is embedded and reranked.** Titles, author names, source domains, feed names, show names, and tags are metadata and are invisible to both the vector search and the Cohere reranker. A query whose only distinguishing terms are metadata ("Kenneth Tanner", "Facebook") has nothing to match and will return arbitrary content rather than nothing — which looks like a working search returning bad results. Reach for `author=`, `domain=`, or `tag=` instead whenever the distinguishing term is a person or a publication.
- `sort="recent"` still draws from a bounded semantic candidate pool for the query (wider than the default, but capped) — it reorders the best topical matches by date, it does not scan the entire collection chronologically. If a user's "recent posts about X" don't show up, the item may exist but not have ranked into that candidate pool; narrowing `days` or trying a more specific `query` helps. For a truly exhaustive chronological listing over a window, use `list_recent_feed_articles` (feeds) or `list_recent_items` (any other collection) instead.
