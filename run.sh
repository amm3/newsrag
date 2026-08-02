#!/usr/bin/env bash
#
# run.sh - Wrapper script for running ingestion scripts
#
# Activates the Python venv and runs the appropriate script.
#
# Usage:
#   ./run.sh [options] wallabag [options]      - Run Wallabag ingestion
#   ./run.sh [options] podcasts [options]      - Run podcast transcript ingestion
#   ./run.sh [options] papers [options]        - Run papers/documents ingestion
#   ./run.sh [options] summarize [options]     - Generate AI summaries (Phase 1)
#   ./run.sh [options] summaries [options]     - Load AI summaries into Qdrant (Phase 2)
#   ./run.sh [options] feeds [options]         - Run RSS/Atom feed ingestion
#   ./run.sh help                              - Show this help
#
# The command name can appear anywhere in the argument list.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
CONFIG_DIR="$SCRIPT_DIR/config"

# Check venv exists
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Run deploy.sh first, or create manually:"
    echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check .env exists
if [[ ! -f "$CONFIG_DIR/.env" ]]; then
    echo "Error: Configuration file not found at $CONFIG_DIR/.env"
    echo "Copy from template: cp $CONFIG_DIR/.env.example $CONFIG_DIR/.env"
    exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Export config directory for scripts to find .env
export QDRANT_LOADER_CONFIG_DIR="$CONFIG_DIR"

show_help() {
    cat << EOF
Qdrant RAG Loader - Ingestion Runner

Usage: ./run.sh <command> [options]

Commands:
  wallabag      Ingest articles from Wallabag
  wallabag-cull Remove orphaned Qdrant entries for deleted Wallabag articles
  podcasts      Ingest podcast transcripts from filesystem
  papers        Ingest papers/documents from filesystem
  summarize     Generate AI summaries for podcasts/papers (Phase 1, writes .ai-summary.md files)
  summaries     Load AI summaries into Qdrant (Phase 2, reads .ai-summary.md files)
  kindle        Ingest Kindle highlights from Bookcision JSON exports
  openwebui     Ingest chat history from OpenWebUI
  help          Show this help message

Wallabag Options:
  --entries ID [ID ...]  Reprocess specific Wallabag entry IDs
  --full                 Full re-sync (ignore state file)
  --dry-run              Don't write to Qdrant
  -v                     Verbose output
  -vv                    Debug output

Wallabag-Cull Options:
  --dry-run    Show what would be deleted without writing to Qdrant
  -v           Verbose output
  -vv          Debug output

Podcast Options:
  --podcast-dir PATH   Directory containing podcast folders (required)
  --full               Full re-sync (ignore state file)
  --dry-run            Don't write to Qdrant
  -v                   Verbose output
  -vv                  Debug output

Papers Options:
  --papers-dir PATH    Directory containing papers/documents (required)
  --collection NAME    Qdrant collection name (default: 'papers')
  --full               Full re-sync (ignore state file)
  --dry-run            Don't write to Qdrant
  -v                   Verbose output
  -vv                  Debug output

Summarize Options (Phase 1 - generate .ai-summary.md files, no Qdrant writes):
  --type {podcast,paper}  Content type to summarize (required)
  --podcast-dir PATH      Directory containing podcasts (used with --type podcast)
  --papers-dir PATH       Directory containing papers/documents (used with --type paper)
  --files PATH [...]      Regenerate summaries for specific source files (overwrites existing)
  --regenerate            Overwrite existing .ai-summary.md files found during discovery
  --limit N               Cap the number of summaries generated this run
  --dry-run               Don't call the LLM or write files
  -v                      Verbose output
  -vv                     Debug output

Summaries Options (Phase 2 - load .ai-summary.md files into Qdrant):
  --podcast-dir PATH   Directory containing podcasts (scanned for .ai-summary.md files)
  --papers-dir PATH    Directory containing papers/documents (scanned for .ai-summary.md files)
  --collection NAME    Qdrant collection name (default: 'summaries')
  --full               Full re-sync (ignore state file)
  --dry-run            Don't write to Qdrant
  -v                   Verbose output
  -vv                  Debug output

Feeds Options:
  --feeds URL [URL ...]  Reprocess specific feed URLs only
  --config PATH          Path to feeds YAML config (default: config/feeds.yaml)
  --full                 Full re-sync (ignore state file)
  --dry-run              Don't write to Qdrant
  -v                     Verbose output
  -vv                    Debug output

Kindle Options:
  --kindle-dir PATH    Directory containing Bookcision JSON files (required)
  --files PATH [...]   Reprocess specific JSON file paths
  --full               Full re-sync (ignore state file)
  --dry-run            Don't write to Qdrant
  -v                   Verbose output
  -vv                  Debug output

OpenWebUI Options:
  --chats UUID [UUID ...]  Reprocess specific chat UUIDs only
  --full                   Full re-sync (ignore state file)
  --dry-run                Don't write to Qdrant
  -v                       Verbose output
  -vv                      Debug output

Examples:
  ./run.sh wallabag -v
  ./run.sh wallabag --entries 1234 5678 -v
  ./run.sh wallabag --full --dry-run -v
  ./run.sh podcasts --podcast-dir /mnt/nas/podcasts -v
  ./run.sh podcasts --podcast-dir /mnt/nas/podcasts --full -v
  ./run.sh papers --papers-dir /mnt/nas/papers -v
  ./run.sh papers --papers-dir /mnt/nas/papers --collection my-papers -v
  ./run.sh papers --papers-dir /mnt/nas/papers --full -v
  ./run.sh summarize --type podcast --podcast-dir /mnt/nas/podcasts -v
  ./run.sh summarize --type paper --papers-dir /mnt/nas/papers -v
  ./run.sh summarize --type podcast --podcast-dir /mnt/nas/podcasts --regenerate -v
  ./run.sh summaries --podcast-dir /mnt/nas/podcasts --papers-dir /mnt/nas/papers -v
  ./run.sh feeds -v
  ./run.sh feeds --full --dry-run -v
  ./run.sh feeds --feeds https://example.com/feed.rss -v
  ./run.sh kindle --kindle-dir /path/to/exports -v
  ./run.sh kindle --kindle-dir /path/to/exports --full -v
  ./run.sh openwebui -v
  ./run.sh openwebui --full --dry-run -v
  ./run.sh openwebui --chats abc123 def456 -v

State files are stored in $CONFIG_DIR/
Delete them to force a full re-sync.
EOF
}

KNOWN_COMMANDS=(wallabag wallabag-cull cull podcasts podcast papers paper summarize summaries feeds feed kindle openwebui owui help --help -h)

COMMAND=""
REMAINING=()
for arg in "$@"; do
    is_cmd=0
    if [[ -z "$COMMAND" ]]; then
        for c in "${KNOWN_COMMANDS[@]}"; do [[ "$arg" == "$c" ]] && is_cmd=1 && break; done
    fi
    if [[ $is_cmd -eq 1 ]]; then
        COMMAND="$arg"
    else
        REMAINING+=("$arg")
    fi
done

COMMAND="${COMMAND:-help}"

case "$COMMAND" in
    wallabag)
        exec python "$SCRIPT_DIR/scripts/wallabag_ingest.py" "${REMAINING[@]}"
        ;;
    wallabag-cull|cull)
        exec python "$SCRIPT_DIR/scripts/wallabag_cull.py" "${REMAINING[@]}"
        ;;
    podcasts|podcast)
        exec python "$SCRIPT_DIR/scripts/podcast_ingest.py" "${REMAINING[@]}"
        ;;
    papers|paper)
        exec python "$SCRIPT_DIR/scripts/papers_ingest.py" "${REMAINING[@]}"
        ;;
    summarize)
        exec python "$SCRIPT_DIR/scripts/summarize.py" "${REMAINING[@]}"
        ;;
    summaries)
        exec python "$SCRIPT_DIR/scripts/summaries_ingest.py" "${REMAINING[@]}"
        ;;
    feeds|feed)
        exec python "$SCRIPT_DIR/scripts/feeds_ingest.py" "${REMAINING[@]}"
        ;;
    kindle)
        exec python "$SCRIPT_DIR/scripts/kindle_ingest.py" "${REMAINING[@]}"
        ;;
    openwebui|owui)
        exec python "$SCRIPT_DIR/scripts/openwebui_chat_loader.py" "${REMAINING[@]}"
        ;;
    help|--help|-h)
        show_help
        exit 0
        ;;
esac
