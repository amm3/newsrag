# Qdrant RAG Loader

A personal RAG system for indexing Wallabag articles, podcast transcripts, research papers, RSS/Atom news feeds, and Kindle book highlights into Qdrant, queryable via OpenWebUI.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Deployment](#deployment)
4. [Configuration](#configuration)
5. [Qdrant Setup](#qdrant-setup)
6. [Running the Ingestion](#running-the-ingestion)
7. [OpenWebUI Integration](#openwebui-integration)
8. [Scheduling](#scheduling)
9. [Maintenance](#maintenance)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Wallabag   │  │ NAS (Pods)  │  │   Papers/   │  │  RSS/Atom   │  │   Kindle    │
│   (API)     │  │ /path/txts  │  │   Docs      │  │   Feeds     │  │  Highlights │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │                │
       └────────────────┴────────────────┴────────────────┴────────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────────┐
                        │       Ingestion Scripts        │
                        │    (Python venv on server)     │
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                                ┌───────────────┐
                                │    OpenAI     │
                                │  Embeddings   │
                                │     API       │
                                └───────┬───────┘
                                        │
                                        ▼
                                ┌───────────────┐
                                │    Qdrant     │
                                │   (server)    │
                                └───────┬───────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────┐
│                        OpenWebUI                            │
├─────────────────────────────────────────────────────────────┤
│  Built-in RAG (ChromaDB)      │  Custom "Knowledge Search"  │
│  ─────────────────────────    │  Tool (queries Qdrant)      │
│  • Knowledge bases you create │  ─────────────────────────  │
│  • Folder uploads             │  • Wallabag articles        │
│  • Ad-hoc file attachments    │  • Podcast transcripts      │
│  • Unchanged, works as normal │  • Papers/documents         │
│                               │  • RSS/Atom feed articles   │
│                               │  • Kindle book highlights   │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **Qdrant on dedicated server** - GPU proximity for future local embedding experiments
- **Storage on NVMe** - Low latency for vector operations
- **OpenAI `text-embedding-3-small`** - Best cost/quality ratio at $0.02/1M tokens
- **Separate collections** - `wallabag_articles`, `podcast_transcripts`, `papers`, `summaries`, `news_feeds`, `kindle_highlights`
- **OpenWebUI unchanged** - Qdrant accessed via custom Tool, not replacing ChromaDB
- **AI summaries as a two-phase pipeline** - `summarize` (LLM call, writes `.ai-summary.md` next to the source file) is decoupled from `summaries` (embeds/loads into Qdrant), so a hand-edited summary reloads without regenerating it. Not yet wired into the OpenWebUI Tool — see [AI Summaries](#ai-summaries)

---

## Prerequisites

On your workstation:
- Git
- SSH access to your server (`~/.ssh/config` with a `Host` entry defined)

On the server:
- Docker and Docker Compose
- Python 3.10+
- Network access to your Wallabag instance
- NFS/SMB mount to NAS podcast directory (for podcast ingestion)

---

## Deployment

### Setup

Clone this repo to your server, configure `config/.env`, then set up the environment:

```bash
cd /path/to/qdrant_loader

# Start Qdrant (if not already running)
docker compose up -d

# Verify Qdrant is healthy
curl -H "api-key: $(grep QDRANT_API_KEY config/.env | cut -d= -f2)" \
  http://localhost:6333/collections

# Test ingestion (dry run)
./run.sh wallabag --dry-run -v
./run.sh podcasts --podcast-dir /path/to/podcasts --dry-run -v
```

---

## Configuration

Copy the example and edit:

```bash
cp config/.env.example config/.env
```

Edit `config/.env` with your credentials. See [config/.env.example](config/.env.example) for all options.

### Wallabag API Credentials

1. Log into your Wallabag instance
2. Go to API clients management (usually `/developer`)
3. Create a new client
4. Copy client ID and secret to `.env`

### OpenAI API Key

Get from https://platform.openai.com/api-keys

### Qdrant API Key

Generate a secure key:
```bash
openssl rand -base64 32
```

### RSS/Atom Feeds

Copy the example feed config and edit:

```bash
cp config/feeds.yaml.example config/feeds.yaml
```

See [config/feeds.yaml.example](config/feeds.yaml.example) for the feed list format.

---

## Qdrant Setup

### Docker Compose

The included `docker-compose.yml` deploys Qdrant:

```bash
cd /path/to/qdrant_loader
docker compose up -d
```

See [docker-compose.yml](docker-compose.yml) for configuration.

### Storage Location

Data is stored in the volume mount defined in `docker-compose.yml`. Adjust the host path if needed.

### Web Dashboard

Access at `http://your-server:6333/dashboard` for browsing collections and testing queries.

### Backup

Add to cron for NAS backup:
```bash
0 3 * * * rsync -av /opt/qdrant/storage/ /path/to/backups/qdrant/
```

---

## Running the Ingestion

Use the `run.sh` wrapper script, which activates the venv automatically:

### Wallabag Articles

```bash
# Dry run (no writes)
./run.sh wallabag --dry-run -v

# Full sync (first time)
./run.sh wallabag --full -v

# Incremental sync (uses state file)
./run.sh wallabag -v
```

### Podcast Transcripts

```bash
# Dry run
./run.sh podcasts --podcast-dir /path/to/podcasts --dry-run -v

# Full sync
./run.sh podcasts --podcast-dir /path/to/podcasts --full -v

# Incremental sync
./run.sh podcasts --podcast-dir /path/to/podcasts -v
```

### Papers / Documents

```bash
# Dry run
./run.sh papers --dry-run -v

# Full sync
./run.sh papers --full -v

# Incremental sync
./run.sh papers -v

# Reprocess specific files
./run.sh papers --files /path/to/doc.pdf /path/to/other.txt -v
```

**Deletions and renames.** `papers` only ever adds. It embeds new and changed
files and never notices one that has gone, so a deleted or renamed document
stays searchable under its old path. `papers-cull` removes those chunks, and
their entries in the collection's state file. Run it with the same
`--papers-dir` and `--collection` as the `papers` run it follows:

```bash
# List what would be removed
./run.sh papers-cull --papers-dir /path/to/papers --collection papers --dry-run -v

# Remove it
./run.sh papers-cull --papers-dir /path/to/papers --collection papers -v
```

A chunk stays when `papers` would select its file today, by the same rules and
the same `file_id` (md5 of the path relative to `--papers-dir`). Every other
chunk with a `file_id` is removed:
- files that were deleted;
- files that were renamed or moved, since the new path was embedded as a new
  document;
- a `foo.txt` once a `foo.md` sits beside it;
- AI summaries (`*.ai-summary.md`). `papers` skips these now, because they
  belong in `summaries`, but older versions embedded them as documents.

Points from other loaders, and points with no `file_id`, are left alone.

**After upgrading from a version that embedded AI summaries:** summaries made up
about half the files in each collection, so the first cull removes about half
of each one's files. Where that is more than half, the mass-delete guard
described below refuses. Check the `--dry-run` list, then run that collection's
cull once by hand with `--force`.

Two guards refuse the whole run when the folder looks unreadable rather than
tidied. A refused run sends mail and changes nothing. They trip when:
- no files are found at all while chunks from the folder are indexed, which is
  what an unmounted NAS looks like;
- more than half of the indexed files are gone at once, and more than 10 of
  them.

Pass `--force` when the files really are gone. A wrong cull fixes itself: the
cull drops the file's state entry, so if the file comes back, the next `papers`
run embeds it again. `--self-test` checks these rules against an in-memory
Qdrant.

### AI Summaries

Two-phase pipeline that generates AI summaries for podcast transcripts and
papers (not Wallabag articles) and indexes them in their own `summaries`
collection.

**Phase 1 — generate.** For each transcript/paper without a summary yet,
writes a sibling `<name>.ai-summary.md` file next to it (e.g.
`Episode.txt` → `Episode.ai-summary.md`), with a `key: value` header
(`title`, `url`, `published_at`, `tags`, `source_file`, `generated_at`,
`model`) followed by the LLM-generated summary body. This step never
touches Qdrant, and never overwrites an existing `.ai-summary.md` file
unless you pass `--regenerate` or `--files` — so you can hand-edit a
summary and it's safe from routine re-runs.

If the source file's own header already lists `tags`, they're carried
forward unchanged. If it doesn't, the LLM proposes a handful alongside the
summary, guided by two things: the tag naming conventions (hyphenated
multi-word concepts, space-separated person names, bare acronyms) and the
tag vocabulary already in use — pulled live from Wallabag's `/api/tags.json`
so generated tags line up with the ones applied by hand. Proposed tags that
differ from an existing tag only in punctuation are snapped to the Wallabag
spelling. The lookup is best-effort: without Wallabag credentials, or if the
server is unreachable, summaries are still generated and tags simply follow
the conventions alone. `SUMMARY_TAG_VOCAB_LIMIT` caps how many tags go into
the prompt (`0` disables the lookup, as does `--no-tag-vocab`).

**Phase 2 — load.** Scans for `.ai-summary.md` files and embeds/upserts
them into the `summaries` collection, tagged with `source_type` (`podcast`
or `paper`). Re-running after you hand-edit a summary picks up the change
automatically via the same mtime-based incremental sync every other loader
uses. Hidden files are skipped, so the `._<name>.ai-summary.md` AppleDouble
files a Mac leaves on a network share are not loaded as summaries. `papers`
skips `.ai-summary.md` files altogether, so a summary is only ever indexed here.

```bash
# Phase 1: generate summaries (costs an LLM call per new file)
./run.sh summarize --type podcast --podcast-dir /path/to/podcasts -v
./run.sh summarize --type paper --papers-dir /path/to/papers -v

# Force-regenerate summaries that already exist
./run.sh summarize --type podcast --podcast-dir /path/to/podcasts --regenerate -v

# Regenerate specific source files
./run.sh summarize --type paper --files /path/to/paper.md -v

# Propose tags from the naming conventions only, without the Wallabag tag list
./run.sh summarize --type paper --papers-dir /path/to/papers --no-tag-vocab -v

# Phase 2: embed and load summaries into Qdrant (cheap — embeddings only)
./run.sh summaries --podcast-dir /path/to/podcasts --papers-dir /path/to/papers -v

# Dry runs work for both phases
./run.sh summarize --type podcast --podcast-dir /path/to/podcasts --dry-run -v
./run.sh summaries --podcast-dir /path/to/podcasts --papers-dir /path/to/papers --dry-run -v
```

`summaries` has the same blind spot as `papers`: it never removes anything.
`summaries-cull` removes the chunks of `.ai-summary.md` files that no longer
exist, and their entries in `.summaries_sync_state.json`. It follows the same
rules and has the same guards as [`papers-cull`](#papers--documents). Give it
the same directories as `summaries`. Only the roots it is given are culled, so
a run with only `--papers-dir` never touches podcast summaries.

```bash
./run.sh summaries-cull --podcast-dir /path/to/podcasts --papers-dir /path/to/papers --dry-run -v
./run.sh summaries-cull --podcast-dir /path/to/podcasts --papers-dir /path/to/papers -v
```

The cull goes by the summary file, not the document it summarizes. A summary
that is still on disk stays indexed even after its source document is deleted.

### RSS/Atom News Feeds

Requires `config/feeds.yaml` (see [Configuration](#configuration)).

```bash
# Dry run
./run.sh feeds --dry-run -v

# Full sync
./run.sh feeds --full -v

# Incremental sync
./run.sh feeds -v

# Ingest a specific feed URL
./run.sh feeds --feeds https://example.com/feed.rss -v
```

### Kindle Highlights

Expects [Bookcision](https://readwise.io/bookcision) JSON export files.

```bash
# Dry run
./run.sh kindle --kindle-dir /path/to/exports --dry-run -v

# Full sync
./run.sh kindle --kindle-dir /path/to/exports --full -v

# Incremental sync
./run.sh kindle --kindle-dir /path/to/exports -v

# Reprocess specific files
./run.sh kindle --kindle-dir /path/to/exports --files book1.json book2.json -v
```

### OpenWebUI Chats

Indexes your OpenWebUI chat history — each message becomes a searchable Qdrant
point. The `chat_id` UUID in every payload lets a retrieval tool fetch the full
conversation from OpenWebUI at query time.

**Prerequisites:** generate an API key in OpenWebUI → Settings → Account → API Key.
Add to `config/.env`:

```ini
OPENWEBUI_URL=http://your-server:3000
OPENWEBUI_API_KEY=your_api_key_here
OPENWEBUI_COLLECTION=openwebui_chats
```

```bash
# First run — index everything
./run.sh openwebui --full -v

# Incremental sync (only chats updated since last run)
./run.sh openwebui -v

# Dry run (no writes)
./run.sh openwebui --dry-run -v

# Reprocess specific chats by UUID
./run.sh openwebui --chats abc-uuid-1 def-uuid-2 -v
```

Tool calls, system prompts, and empty messages are automatically skipped. Each
chat's existing points are deleted and replaced on every re-index (idempotent).

### State Files

Incremental sync state is stored in `config/`:
- `.wallabag_sync_state.json`
- `.podcast_sync_state.json`
- `.papers_sync_state_<collection>.json`, one per `--collection` (an older
  `.papers_sync_state.json` is renamed to `.papers_sync_state_papers.json` on
  the first run)
- `.summaries_sync_state.json`
- `.feeds_sync_state.json`
- `.kindle_sync_state.json`
- `.openwebui_sync_state.json`

Delete a state file to force a full re-sync for that source.

A run claims its state file for the duration, via a `<state file>.lock` beside
it, so two runs of the same loader cannot overlap and lose each other's work.
Nothing schedules the loaders exclusively — cron fires on a timer whether or not
the last run finished, and a run started by hand knows nothing of either — so
the second run logs which pid holds the file and exits 0 without doing anything.
A skipped run deliberately does **not** stamp `last_sync`, so if every run were
to skip, the source would go stale and `./run.sh health` would say so.

The lock is keyed on the state file, not the command, because those do not
correspond: `papers` writes a different file per `--collection` and those runs
may safely overlap, while `summaries` takes the same flag and writes one file
regardless, so those runs may not. A `--dry-run` writes nothing, so it neither
claims the file nor is turned away from it. Set `QDRANT_LOADER_NO_LOCK=1` to
disable the whole mechanism — worth knowing if `config/` ever lives on NFS,
where `flock` is unreliable. Locks are released by the kernel when a run ends,
including if it is killed, so there is never a stale lock to clear by hand.

The loaders never remove anything from a `file_mtimes` map. `papers-cull` and
`summaries-cull` do. They drop the entry of every file under their root that the
loader would no longer select, and leave the rest of the file alone, `last_sync`
included, because a cull is not a sync. Each cull claims the same lock as the
loader whose file it edits, so a cull and a `papers` run for the same collection
never overlap. Entries under a root spelled differently from the one the cull
was given are kept, since the cull cannot tell whether they are live.

Each of these carries a `last_sync` timestamp, written when a run completes;
`./run.sh health` reads it to tell a stalled pipeline from a healthy one. A
`--dry-run` writes no state, so it never counts as a sync. A `.feeds_sync_state.json`
written before `last_sync` existed falls back to the file's mtime until the next
full run rewrites it.

`.feeds_sync_state.json` records a 32-character digest of each entry GUID rather
than the GUID itself. Nothing ever reads those identifiers back — they exist only
to answer "seen this one?" — and feeds that mint opaque tokens over a kilobyte
long had grown the file to 38 MB, 36 MB of it identifier text. The same ledger as
digests is about 8 MB. A file in the older shape is converted in place on the
first run, which changes no dedup decisions, and feeds no longer listed in
`feeds.yaml` are dropped from it. A `--feeds` run converts but never drops, since
it is told only about the feeds it was asked to reprocess.

`config/.health_state.json` lives alongside them but is not sync state — it holds
the health check's alert-suppression timestamps. Delete it to re-arm the alerts.

---

## Alerting

Every loader sends an email when it fails. Two mechanisms cover that:

- **Handled errors.** Each loader calls `log_fatal()` for the failures it
  anticipates — missing env vars, an absent config file, an authentication
  failure — which logs, mails and exits.
- **Everything else.** `main()` runs inside `run_guarded()`
  ([scripts/alert.py](scripts/alert.py)), so an exception no one anticipated
  mails the traceback instead of dying as stderr that cron throws away. This
  matters because a cron line like the ones below redirects stderr into a log
  file: without it, an invalid `config/feeds.yaml` can stop an index for days
  in complete silence.

Configure SMTP in `config/.env`:

```ini
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SMTP_USER=you@gmail.com
ALERT_SMTP_PASS=your_app_password      # Gmail: use an App Password
ALERT_FROM=qdrant-loader@example.com
ALERT_TO=you@example.com              # comma-separated for multiple recipients
```

Leave `ALERT_SMTP_HOST` blank to disable alerting entirely — the loaders will
behave exactly as before with no delay or error.

> **Gmail users:** generate an App Password at myaccount.google.com → Security →
> App passwords. Do not use your account login password.

### Health check

A failure that mails is the easy case. The harder one is a source that has
quietly stopped advancing — runs crashing, cron no longer firing, an upstream
that went away. Each loader stamps `last_sync` into its state file on every run
that finishes, including the runs that find nothing new to do, so a stale stamp
means runs have stopped rather than that the source has been quiet:

```bash
./run.sh health -v
```

It reports every watched source, mails once when one falls behind, exits
non-zero while anything is stale, and mails again when the source recovers.
Repeat alerts are rate-limited by `HEALTH_REALERT_HOURS` so a frequent cron
doesn't mail every run for as long as something stays broken. Suppression state
lives in `config/.health_state.json`.

```ini
HEALTH_CHECKS=feeds                  # feeds, wallabag, podcasts, papers, summaries, kindle, openwebui
HEALTH_STALE_HOURS=36                # alert once a source is this far behind
# HEALTH_STALE_HOURS_FEEDS=6         # per-source override
HEALTH_REALERT_HOURS=24              # minimum gap between repeat alerts
```

Because `papers` is run once per collection, with a state file each, listing it
covers every collection that has one — judged separately, so a collection that
stops syncing cannot hide behind one that is still running. `papers:<collection>`
names a single one, and is how to watch a collection that has yet to write state.
Thresholds resolve most-specific-first:
`HEALTH_STALE_HOURS_PAPERS_MY_THOUGHTS`, then `HEALTH_STALE_HOURS_PAPERS`, then
`HEALTH_STALE_HOURS`.

Set the threshold above the gap between scheduled runs, with room for one
missed run and the runtime itself: a feed ingest every two hours that takes up
to 45 minutes wants about 6, not 36. List only sources you run on a schedule —
one you run by hand has no recent stamp and would look permanently stale.

Give the check its own cron line rather than appending it to a script that runs
the loaders. A watchdog inside the thing it watches stops when that stops, and
it should run more often than the ingests so a missed run is noticed in hours.

`--dry-run` reports findings without mailing or recording them; `--force`
ignores the rate limit. Both are useful for confirming the thresholds do what
you expect. Exit status is 0 when everything is fresh and 1 while anything is
stale — including while its alert is being rate-limited, so a suppressed alert
never looks like success.

Two limits worth knowing. It runs from cron, so it cannot report that cron
itself has stopped. And because `feeds_ingest` stamps `last_sync` on any run
that finishes, a run where every single feed failed to fetch would otherwise
look healthy — that case is covered separately, by the loader itself mailing a
`[WARN] feeds_ingest` summary when no feed yielded anything.

---

## OpenWebUI Integration

### Custom Tool

Create a new Tool in OpenWebUI (Workspace → Tools → Create) using the code in [scripts/openwebui_tool.py](scripts/openwebui_tool.py).

Configure the Tool's Valves (settings):
- `QDRANT_URL`: `http://your-server:6333`
- `QDRANT_API_KEY`: Your Qdrant API key
- `OPENAI_API_KEY`: Your OpenAI API key
- `TOP_K`: Number of results per collection (default: 5)
- `WALLABAG_COLLECTION`: Qdrant collection name (default: `wallabag_articles`)
- `PODCAST_COLLECTION`: Qdrant collection name (default: `podcast_transcripts`)
- `FEEDS_COLLECTION`: Qdrant collection name (default: `news_feeds`)
- `KINDLE_COLLECTION`: Qdrant collection name (default: `kindle_highlights`)
- `DOCUMENT_COLLECTIONS`: Comma-separated collection names for document sources (default: `papers`)

### System Prompt Enhancement

For models that should automatically use your knowledge base, add to the system prompt:

```
You have access to a personal knowledge base via the search_knowledge tool.
When the user asks questions that might benefit from personal context, use
search_knowledge first.

Sources include:
- Saved articles from Wallabag
- Podcast transcripts
- Research papers and documents
- RSS/Atom news feed articles
- Kindle book highlights
- OpenWebUI chat history

The 'collection' parameter lets you target a specific source: 'articles',
'podcasts', 'feeds', 'kindle', 'documents', or 'all' (default).

Always cite which source you're drawing from when using retrieved information.
```

---

## Scheduling

### Cron

```bash
# /etc/cron.d/qdrant-ingest
0 3 * * * youruser /path/to/qdrant_loader/run.sh wallabag -v >> /var/log/qdrant-ingest.log 2>&1
15 3 * * * youruser /path/to/qdrant_loader/run.sh podcasts --podcast-dir /path/to/podcasts -v >> /var/log/qdrant-ingest.log 2>&1
30 3 * * * youruser /path/to/qdrant_loader/run.sh feeds -v >> /var/log/qdrant-ingest.log 2>&1

# Notice a source that has stopped syncing (see Alerting → Health check)
*/30 * * * * youruser /path/to/qdrant_loader/run.sh health >> /var/log/qdrant-ingest.log 2>&1
```

Papers and Kindle highlights are typically ingested on-demand rather than on a schedule.
If you do schedule `papers` or `summaries`, run the matching cull right after it,
with the same arguments. Otherwise deleted and renamed files stay searchable:

```bash
*/15 * * * * youruser { /path/to/qdrant_loader/run.sh papers --papers-dir /path/to/papers --collection papers -v; /path/to/qdrant_loader/run.sh papers-cull --papers-dir /path/to/papers --collection papers -v; } >> /var/log/qdrant-ingest.log 2>&1
```

A cull is cheap: one scroll of the collection's ids and paths, and the same disk
walk `papers` already does. If a guard refuses, every run mails until you look,
so use a slower schedule if that would be too much.

Note that `2>&1` into a log file is why the health check is worth having: it
sends a failing run's traceback somewhere nobody looks. The loaders mail on
their own account, and the health check covers the case where a run stops
happening at all.

---

## Maintenance

### Check Collection Stats

```bash
source venv/bin/activate
python -c "
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
load_dotenv('config/.env')
client = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ.get('QDRANT_API_KEY'))
for c in client.get_collections().collections:
    info = client.get_collection(c.name)
    print(f'{c.name}: {info.points_count} points, {info.vectors_count} vectors')
"
```

### Reset a Collection

```bash
source venv/bin/activate
python -c "
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
load_dotenv('config/.env')
client = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ.get('QDRANT_API_KEY'))
client.delete_collection('wallabag_articles')  # or any other collection name
print('Collection deleted')
"
```

Then delete the corresponding state file and run a full sync.

### Update Dependencies

```bash
source venv/bin/activate
pip install --upgrade qdrant-client openai requests python-dotenv
pip freeze > requirements.txt
```

---

## Troubleshooting

### Qdrant Connection Refused

```bash
# Check container status
docker compose ps
docker compose logs qdrant

# Test connectivity
curl http://localhost:6333/collections
```

### OpenAI Rate Limits

The scripts batch embeddings, but if you hit limits:
- Reduce batch size in the scripts
- Add delays between articles
- Use OpenAI's Batch API for large jobs (50% cheaper)

### Empty Search Results

```bash
# Check collection has points
curl -H "api-key: YOUR_KEY" "http://localhost:6333/collections/wallabag_articles" | jq '.result.points_count'
```

### Wallabag OAuth Errors

- Verify credentials at `https://your-wallabag/developer`
- Check token expiry
- Test with curl:
  ```bash
  curl -X POST "https://your-wallabag/oauth/v2/token" \
    -d "grant_type=password&client_id=ID&client_secret=SECRET&username=USER&password=PASS"
  ```

### OpenWebUI Tool Not Working

- Check Tool is enabled for the model
- Verify Valves are configured
- Check OpenWebUI container can reach your Qdrant host
- Test the Qdrant query manually (see Manual Query Testing below)

### Manual Query Testing

```bash
# Get embedding for a test query
curl https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "test query", "model": "text-embedding-3-small"}' \
  | jq '.data[0].embedding' > /tmp/vec.json

# Search Qdrant
curl -X POST "http://localhost:6333/collections/wallabag_articles/points/search" \
  -H "api-key: $QDRANT_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"vector\": $(cat /tmp/vec.json), \"limit\": 3, \"with_payload\": true}" \
  | jq '.result[] | {score, title: .payload.title}'
```

---

## Project Structure

```
qdrant_loader/
├── README.md
├── run.sh                    # Wrapper script (activates venv)
├── docker-compose.yml        # Qdrant container
├── requirements.txt          # Python dependencies
├── config/
│   ├── .env.example          # Template configuration
│   ├── .env                  # Your configuration (git-ignored)
│   ├── feeds.yaml.example    # Template feed list
│   └── feeds.yaml            # Your feed list (git-ignored)
└── scripts/
    ├── wallabag_ingest.py    # Wallabag article ingestion
    ├── wallabag_client.py    # Shared Wallabag API client
    ├── podcast_ingest.py     # Podcast transcript ingestion
    ├── papers_ingest.py      # Papers/document ingestion
    ├── papers_cull.py        # Removes chunks of deleted/renamed papers (./run.sh papers-cull)
    ├── summarize.py          # AI summary generation (Phase 1, no Qdrant writes)
    ├── summaries_ingest.py   # AI summary ingestion into Qdrant (Phase 2)
    ├── summaries_cull.py     # Removes chunks of deleted summaries (./run.sh summaries-cull)
    ├── file_cull.py          # Shared cull rules, guards and --self-test
    ├── feeds_ingest.py       # RSS/Atom feed ingestion
    ├── kindle_ingest.py      # Kindle highlights ingestion
    ├── alert.py              # Shared email alerting + the top-level crash guard
    ├── health_check.py       # Staleness watchdog (./run.sh health)
    └── openwebui_tool.py     # OpenWebUI Tool code
```

---

## Cost Estimate

With typical usage across all sources:
- **Initial indexing**: $0.10 - $1.00 (depending on corpus size)
- **Incremental updates**: Negligible (pennies/month)

OpenAI `text-embedding-3-small` pricing: $0.02 per 1M tokens

AI summary generation (`./run.sh summarize`) additionally makes one chat-completion
call per file, at `SUMMARY_MODEL` pricing — check current OpenAI pricing before
running it over a large backlog. It also makes one free Wallabag `/api/tags.json`
call per run to load the tag vocabulary. Loading summaries into Qdrant (`./run.sh summaries`)
is embeddings-only and costs the same negligible amount as any other incremental sync.

---

*Last updated: August 2026*
