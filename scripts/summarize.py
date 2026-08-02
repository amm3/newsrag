#!/usr/bin/env python3
"""
summarize.py - Generate AI summaries for podcast transcripts and papers (Phase 1)

Scans a podcast or papers directory for source files that don't yet have an
.ai-summary.md sibling, asks an LLM to summarize each one, and writes the
summary to disk next to the source file with a key: value header (title,
url, published_at, tags, plus generation provenance).

If the source file's header already has tags, they're carried forward as-is.
If not, the LLM is asked to infer a handful of topic tags from the content
alongside the summary.

This script never touches Qdrant. Run summaries_ingest.py (Phase 2) to load
generated (or hand-edited) summaries into the 'summaries' collection.

Existing .ai-summary.md files are never overwritten unless --regenerate or
--files is used, so a manually edited summary is safe from routine re-runs.
"""

import sys
import os
import re
import json
import argparse
import logging
import socket
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from alert import send_alert
from openai import OpenAI

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'

AUDIO_EXTENSIONS = ['.mp3', '.m4a', '.mp4', '.wav', '.ogg', '.flac']

SUMMARY_SYSTEM_PROMPT = {
    'podcast': (
        "You write summaries of podcast transcripts for a semantic search index - not a "
        "recap of what was said and in what order. Identify the core themes, arguments, "
        "and ideas conveyed, and describe each conceptually in your own words, including "
        "related terminology and synonyms someone might search for even if the transcript "
        "itself never uses those exact words. Group related points together by theme "
        "rather than following the conversation's chronological order, and favor breadth "
        "of ideas and their implications over anecdotal or narrative detail. Do not "
        "narrate the flow of the conversation (avoid phrasing like 'the interview opens "
        "with' or 'he then argues') - state each idea directly. Write a few thematic "
        "bullet points or short paragraphs. Do not include a title or heading, just the "
        "summary body."
    ),
    'paper': (
        "You write summaries of papers/documents for a semantic search index - not a "
        "restatement of the document's structure. Identify the core themes, claims, "
        "methodology, findings, and implications, and describe each conceptually in your "
        "own words, including related terminology and synonyms someone might search for "
        "even if the document itself never uses those exact words. Group related points "
        "together by theme rather than following the document's section order, and favor "
        "breadth of ideas and their implications over restating specific details. Do not "
        "narrate the document's structure (avoid phrasing like 'the paper then discusses') "
        "- state each idea directly. Write a few thematic bullet points or short "
        "paragraphs. Do not include a title or heading, just the summary body."
    ),
}

# Used instead of SUMMARY_SYSTEM_PROMPT when the source has no existing tags,
# so the model can propose some alongside the summary in one call.
SUMMARY_WITH_TAGS_SYSTEM_PROMPT = {
    'podcast': (
        "You summarize podcast transcripts for a semantic search index, not a "
        "chronological recap. Respond with a JSON object with exactly two keys: "
        "\"summary\" (identify the core themes, arguments, and ideas conveyed, described "
        "conceptually in your own words - including related terminology and synonyms "
        "someone might search for even if the transcript itself never uses those exact "
        "words; group related points by theme rather than the conversation's chronological "
        "order; favor breadth of ideas over narrative detail; do not narrate the flow of "
        "the conversation, e.g. 'the interview opens with' - state each idea directly; a "
        "few thematic bullet points or short paragraphs, no title or heading) and \"tags\" "
        "(a list of 3-6 short, lowercase topic tags describing the episode's subject "
        "matter, e.g. [\"economy\", \"politics\"])."
    ),
    'paper': (
        "You summarize research papers and documents for a semantic search index, not a "
        "restatement of the document's structure. Respond with a JSON object with exactly "
        "two keys: \"summary\" (identify the core themes, claims, methodology, findings, "
        "and implications, described conceptually in your own words - including related "
        "terminology and synonyms someone might search for even if the document itself "
        "never uses those exact words; group related points by theme rather than section "
        "order; favor breadth of ideas over restating specific details; do not narrate the "
        "document's structure, e.g. 'the paper then discusses' - state each idea directly; "
        "a few thematic bullet points or short paragraphs, no title or heading) and "
        "\"tags\" (a list of 3-6 short, lowercase topic tags describing the paper's "
        "subject matter, e.g. [\"machine-learning\", \"nlp\"])."
    ),
}


def main():
    # Load .env early so env vars are available for argument defaults
    config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
    load_dotenv(config_dir / '.env')

    parser = argparse.ArgumentParser(description='Generate AI summaries for podcast transcripts and papers')
    parser.add_argument("--type", choices=["podcast", "paper"], required=True, help="Content type to summarize")
    parser.add_argument("--podcast-dir", default=os.environ.get('PODCAST_DIR'), help="Root directory containing podcasts (used with --type podcast)")
    parser.add_argument("--papers-dir", default=os.environ.get('PAPERS_DIR'), help="Root directory containing papers/documents (used with --type paper)")
    parser.add_argument("--files", nargs='+', metavar='PATH', help="Regenerate summaries for specific source files (overwrites existing)")
    parser.add_argument("--regenerate", action="store_true", help="Overwrite existing .ai-summary.md files found during discovery")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of summaries generated this run")
    parser.add_argument("--dry-run", action="store_true", help="Don't call the LLM or write files")
    parser.add_argument("-v", action="store_true", default=False, help="Print extra info")
    parser.add_argument("-vv", action="store_true", default=False, help="Print (more) extra info")
    args = parser.parse_args()

    if args.vv:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.DEBUG)
    elif args.v:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.INFO)
    else:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.WARNING)

    # Validate required config
    required_vars = ['OPENAI_API_KEY']
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        log_fatal(f"Missing required environment variables: {', '.join(missing)}")

    model = os.environ.get('SUMMARY_MODEL', 'gpt-5-mini')
    max_input_chars = int(os.environ.get('SUMMARY_MAX_INPUT_CHARS', 100000))

    dir_arg = args.podcast_dir if args.type == 'podcast' else args.papers_dir
    root_dir = Path(dir_arg) if dir_arg else None

    openai_client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        max_retries=int(os.environ.get('OPENAI_MAX_RETRIES', 5)),
    )

    if args.files:
        candidates = []
        for fp in args.files:
            p = Path(fp).resolve()
            if not p.exists():
                logging.error(f"File not found: {fp}")
                continue
            candidates.append(p)
        logging.info(f"Regenerating {len(candidates)} of {len(args.files)} requested files")
    else:
        if root_dir is None:
            dir_flag = '--podcast-dir' if args.type == 'podcast' else '--papers-dir'
            env_var = 'PODCAST_DIR' if args.type == 'podcast' else 'PAPERS_DIR'
            log_fatal(f"{dir_flag} is required (or set {env_var})")
        if not root_dir.exists():
            log_fatal(f"Directory does not exist: {root_dir}")

        if args.type == 'podcast':
            sources = find_podcast_sources(root_dir)
        else:
            sources = find_paper_sources(root_dir)

        candidates = []
        for source_path in sources:
            summary_path = source_path.with_suffix('.ai-summary.md')
            if summary_path.exists() and not args.regenerate:
                logging.debug(f"Skipping {source_path.name}: summary already exists")
                continue
            candidates.append(source_path)

        logging.info(f"Found {len(sources)} {args.type} source(s), {len(candidates)} need summaries")

    if args.limit is not None:
        candidates = candidates[:args.limit]

    if not candidates:
        logging.info("No summaries to generate")
        return 0

    generated = 0
    for i, source_path in enumerate(candidates, 1):
        try:
            if process_source(source_path, root_dir, args.type, openai_client, model, max_input_chars, args.dry_run):
                generated += 1
                logging.info(f"[{i}/{len(candidates)}] Summarized: {source_path.name}")
        except Exception as e:
            logging.error(f"Failed to process {source_path}: {e}")

    logging.warning(f"Completed: {generated}/{len(candidates)} summaries generated")
    return 0


def find_audio_for(txt_path: Path) -> Path | None:
    """Find the audio file corresponding to a transcript .txt file."""
    for ext in AUDIO_EXTENSIONS:
        candidate = txt_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def find_podcast_sources(root_dir: Path) -> list[Path]:
    """Find transcript .txt files that have a corresponding audio file."""
    sources = []
    for txt_file in root_dir.rglob("*.txt"):
        if find_audio_for(txt_file) is not None:
            sources.append(txt_file)
        else:
            logging.debug(f"Skipping {txt_file}: no audio counterpart")
    return sorted(sources)


def find_paper_sources(root_dir: Path) -> list[Path]:
    """
    Find processable papers, preferring .md over .txt over originals
    (same selection logic as papers_ingest.py). Summary files themselves
    are excluded so they never get treated as a new source to summarize.
    """
    stem_map: dict[tuple[Path, str], dict[str, Path]] = {}

    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith('.'):
            continue
        if file_path.name.endswith('.ai-summary.md'):
            continue
        key = (file_path.parent, file_path.stem)
        stem_map.setdefault(key, {})[file_path.suffix.lower()] = file_path

    selected = []
    for (parent, stem), ext_paths in stem_map.items():
        if '.md' in ext_paths:
            selected.append(ext_paths['.md'])
        elif '.txt' in ext_paths:
            selected.append(ext_paths['.txt'])
        else:
            logging.debug(f"Skipping {parent / stem}: no .md or .txt version")

    return sorted(selected)


def parse_header(content: str) -> tuple[dict, str]:
    """
    Parse optional key: value metadata from the top of a file.
    Parsing stops at the first blank line or non-matching line.
    'tags' values are split by comma, lowercased, and whitespace-stripped.
    Returns (metadata dict, remaining content with header stripped).
    """
    metadata = {}
    lines = content.split('\n')
    end = 0
    for line in lines:
        if not line.strip():
            end += 1  # consume the blank separator line
            break
        m = re.match(r'^(\w[\w\s]*?)\s*:\s*(.+)$', line)
        if not m:
            break
        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        metadata[key] = [t.strip().lower() for t in value.split(',')] if key == 'tags' else value
        end += 1
    return metadata, '\n'.join(lines[end:])


def extract_filename_published_at(stem: str, name: str) -> str | None:
    """Same DD MMM YYYY / YYYY-MM-DD_ filename-date convention as podcast_ingest.py."""
    embedded_match = re.search(r'\b(\d{1,2} \w{3} \d{4})\b', stem)
    if embedded_match:
        try:
            return datetime.strptime(embedded_match.group(1), "%d %b %Y").strftime("%Y-%m-%d")
        except ValueError:
            pass
    prefix_match = re.match(r'^(\d{4}-\d{2}-\d{2})_', name)
    if prefix_match:
        return prefix_match.group(1)
    return None


def truncate_body(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def relative_or_str(path: Path, root_dir: Path | None) -> str:
    """Path relative to root_dir when possible, else the path as given."""
    if root_dir is not None:
        try:
            return str(path.relative_to(root_dir))
        except ValueError:
            pass
    return str(path)


def generate_summary(body: str, openai_client: OpenAI, model: str, content_type: str) -> str:
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT[content_type]},
            {"role": "user", "content": body},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_summary_with_tags(body: str, openai_client: OpenAI, model: str, content_type: str) -> tuple[str, list[str]]:
    """Like generate_summary, but also asks the model to propose topic tags (JSON mode)."""
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_WITH_TAGS_SYSTEM_PROMPT[content_type]},
            {"role": "user", "content": body},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ValueError("Model returned an empty summary")
    raw_tags = data.get("tags") or []
    tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
    return summary, tags


def format_header(fields: dict) -> str:
    lines = []
    for key, value in fields.items():
        if value is None or value == '':
            continue
        if isinstance(value, list):
            if not value:
                continue
            value = ', '.join(str(v) for v in value)
        lines.append(f"{key}: {value}")
    return '\n'.join(lines)


def write_summary_file(summary_path: Path, header_fields: dict, body: str):
    header = format_header(header_fields)
    content = f"{header}\n\n{body.strip()}\n"
    summary_path.write_text(content, encoding='utf-8')


def process_source(source_path: Path, root_dir: Path | None, source_type: str,
                    openai_client: OpenAI, model: str, max_input_chars: int,
                    dry_run: bool = False) -> bool:
    """Generate and write a .ai-summary.md sibling for one source file."""
    summary_path = source_path.with_suffix('.ai-summary.md')

    try:
        content = source_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        logging.error(f"Failed to read {source_path}: {e}")
        return False

    if not content.strip():
        logging.debug(f"Skipping {source_path}: empty content")
        return False

    header_meta, body = parse_header(content)

    if not body.strip():
        logging.debug(f"Skipping {source_path}: no body content after header")
        return False

    title = header_meta.get('title') or source_path.stem
    url = header_meta.get('url')
    tags = header_meta.get('tags')

    published_at = None
    if source_type == 'podcast':
        published_at = extract_filename_published_at(source_path.stem, source_path.name)
    if header_meta.get('published_at'):
        published_at = header_meta['published_at']

    needs_ai_tags = not tags

    truncated_body, was_truncated = truncate_body(body.strip(), max_input_chars)
    if was_truncated:
        logging.warning(f"{source_path.name}: source text truncated to {max_input_chars} chars for summarization")

    if dry_run:
        tag_note = " (+ AI-generated tags)" if needs_ai_tags else ""
        logging.info(f"[dry-run] Would generate summary for {source_path} -> {summary_path.name}{tag_note}")
        return True

    try:
        if needs_ai_tags:
            try:
                summary_text, tags = generate_summary_with_tags(truncated_body, openai_client, model, source_type)
            except Exception as e:
                logging.warning(f"{source_path.name}: AI tag generation failed ({e}), falling back to a plain summary with no tags")
                summary_text = generate_summary(truncated_body, openai_client, model, source_type)
        else:
            summary_text = generate_summary(truncated_body, openai_client, model, source_type)
    except Exception as e:
        logging.error(f"Failed to generate summary for {source_path}: {e}")
        return False

    header_fields = {
        'title': title,
        'url': url,
        'published_at': published_at,
        'tags': tags,
        'source_file': relative_or_str(source_path, root_dir),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'model': model,
    }

    try:
        write_summary_file(summary_path, header_fields, summary_text)
    except Exception as e:
        logging.error(f"Failed to write {summary_path}: {e}")
        return False

    return True


def log_fatal(msg, exit_code=-1):
    logging.critical(f"Fatal Err: {msg}")
    send_alert(
        subject=f"[ALERT] summarize failed on {socket.gethostname()}",
        body=msg
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
