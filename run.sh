#!/usr/bin/env bash
#
# run.sh - Wrapper script for running ingestion scripts
#
# Activates the Python venv and runs the appropriate script.
#
# Usage:
#   ./run.sh wallabag [options]      - Run Wallabag ingestion
#   ./run.sh podcasts [options]      - Run podcast transcript ingestion
#   ./run.sh papers [options]        - Run papers/documents ingestion
#   ./run.sh help                    - Show this help
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
  wallabag    Ingest articles from Wallabag
  podcasts    Ingest podcast transcripts from filesystem
  papers      Ingest papers/documents from filesystem
  help        Show this help message

Wallabag Options:
  --entries ID [ID ...]  Reprocess specific Wallabag entry IDs
  --full                 Full re-sync (ignore state file)
  --dry-run              Don't write to Qdrant
  -v                     Verbose output
  -vv                    Debug output

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

Examples:
  ./run.sh wallabag -v
  ./run.sh wallabag --entries 1234 5678 -v
  ./run.sh wallabag --full --dry-run -v
  ./run.sh podcasts --podcast-dir /mnt/nas/podcasts -v
  ./run.sh podcasts --podcast-dir /mnt/nas/podcasts --full -v
  ./run.sh papers --papers-dir /mnt/nas/papers -v
  ./run.sh papers --papers-dir /mnt/nas/papers --collection my-papers -v
  ./run.sh papers --papers-dir /mnt/nas/papers --full -v

State files are stored in $CONFIG_DIR/
Delete them to force a full re-sync.
EOF
}

case "${1:-help}" in
    wallabag)
        shift
        exec python "$SCRIPT_DIR/scripts/wallabag_ingest.py" "$@"
        ;;
    podcasts|podcast)
        shift
        exec python "$SCRIPT_DIR/scripts/podcast_ingest.py" "$@"
        ;;
    papers|paper)
        shift
        exec python "$SCRIPT_DIR/scripts/papers_ingest.py" "$@"
        ;;
    help|--help|-h)
        show_help
        exit 0
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './run.sh help' for usage information."
        exit 1
        ;;
esac
