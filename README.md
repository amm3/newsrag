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
- **Separate collections** - `wallabag_articles`, `podcast_transcripts`, `papers`, `news_feeds`, `kindle_highlights`
- **OpenWebUI unchanged** - Qdrant accessed via custom Tool, not replacing ChromaDB

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
- `.papers_sync_state.json`
- `.feeds_sync_state.json`
- `.kindle_sync_state.json`
- `.openwebui_sync_state.json`

Delete a state file to force a full re-sync for that source.

---

## Alerting

All loaders send an email when they exit with a fatal error (missing env vars,
authentication failure, etc.). Configure SMTP in `config/.env`:

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

### Skills File

[openwebui_skill.md](openwebui_skill.md) is a pre-written Skills document you can paste directly into OpenWebUI (Workspace → Skills → Create). It provides the AI model with detailed guidance on how and when to use every tool in the toolkit — when to call `search_knowledge` vs `get_articles_by_tag`, how to interpret results, how to filter by date or tag, how to fetch full content, and how to handle edge cases.

Copy the contents of `openwebui_skill.md` into the skill body and save it, then enable it alongside the Tool for the model you're using.

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
```

Papers and Kindle highlights are typically ingested on-demand rather than on a schedule.

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
├── openwebui_skill.md        # OpenWebUI Skill document (paste into Workspace → Skills)
├── config/
│   ├── .env.example          # Template configuration
│   ├── .env                  # Your configuration (git-ignored)
│   ├── feeds.yaml.example    # Template feed list
│   └── feeds.yaml            # Your feed list (git-ignored)
└── scripts/
    ├── wallabag_ingest.py    # Wallabag article ingestion
    ├── podcast_ingest.py     # Podcast transcript ingestion
    ├── papers_ingest.py      # Papers/document ingestion
    ├── feeds_ingest.py       # RSS/Atom feed ingestion
    ├── kindle_ingest.py      # Kindle highlights ingestion
    └── openwebui_tool.py     # OpenWebUI Tool code
```

---

## Cost Estimate

With typical usage across all sources:
- **Initial indexing**: $0.10 - $1.00 (depending on corpus size)
- **Incremental updates**: Negligible (pennies/month)

OpenAI `text-embedding-3-small` pricing: $0.02 per 1M tokens

---

*Last updated: February 2026*
