#!/usr/bin/env python3
"""
file_cull.py - Shared core of papers_cull.py and summaries_cull.py

papers_ingest.py and summaries_ingest.py only ever add. A run embeds new and
modified files and never looks for files that have gone, so a deleted or
renamed document keeps its chunks searchable forever, and keeps its entry in
the loader's file_mtimes. The two cull commands tidy up after the loaders, and
this module is the part they share.

The rule is the loader's own. A file's chunks are live when the loader would
select that file today: the same find function, the same file_id. Anything else
in scope that carries a file_id is an orphan. It was deleted, renamed (a new
path is a new document), shadowed (foo.txt once foo.md exists beside it), or is
a kind of file the loader no longer selects at all, such as the AI summaries
papers_ingest.py once embedded as documents.

A cull that cannot see the disk would delete everything, so two guards stand in
front of the deletes. Each refuses the whole run, mails, and leaves Qdrant and
the state file untouched, and --force overrides both:
  - a root with no files at all, while chunks from it are indexed. That is what
    an unmounted NAS looks like;
  - more than half of a root's indexed files gone at once (and more than
    MASS_DELETE_FLOOR of them). That is more likely a wrong path or a partial
    mount than a cleanup.

Being wrong here is not a disaster. The cull drops the state entry of every file
whose chunks it removes, so if a file it took for gone comes back, the next
loader run sees it as new and embeds it again. The AI summaries on disk are not
touched at all.
"""

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchAny,
)

import papers_ingest
import summaries_ingest
from sync_state import write_state

# Below this many orphans, the more-than-half guard stays quiet. A small
# collection can lose most of its files in an ordinary cleanup, and if it was a
# glitch after all, the files come back on the next loader run.
MASS_DELETE_FLOOR = 10

# file_ids per delete request
DELETE_BATCH = 200


@dataclass
class Scope:
    """One root directory, and the indexed points that came from it."""
    label: str                      # how the logs name this root
    root: Path
    live: dict[str, Path]           # file_id -> file the loader would select today
    owns: Callable[[dict], bool]    # does this payload belong to this root?
    path_key: str                   # payload key holding the path relative to root


@dataclass
class Indexed:
    path: str                       # relative to the root, as the payload records it
    points: int = 0


@dataclass
class Orphan:
    file_id: str
    path: str
    points: int
    reason: str


def papers_scope(papers_dir: Path) -> Scope:
    """Everything papers_ingest.py indexed from papers_dir into one collection."""
    live = {papers_ingest.file_id_for(p.relative_to(papers_dir)): p
            for p in papers_ingest.find_documents(papers_dir)}
    return Scope(
        label=f"papers dir {papers_dir}",
        root=papers_dir,
        live=live,
        # Every papers chunk says source 'paper'. Allow a missing source too,
        # but never touch another loader's points if a collection is shared.
        owns=lambda payload: payload.get('source', 'paper') == 'paper',
        path_key='file_path',
    )


def summary_scope(root: Path, source_type: str) -> Scope:
    """The summaries collection's points for one --podcast-dir or --papers-dir."""
    live = {summaries_ingest.file_id_for(source_type, p.relative_to(root)): p
            for p in summaries_ingest.find_summaries(root)}
    return Scope(
        label=f"{source_type} summaries under {root}",
        root=root,
        live=live,
        owns=lambda payload: payload.get('source_type') == source_type,
        path_key='summary_file',
    )


def scan_collection(qdrant: QdrantClient, collection: str,
                    scopes: list[Scope]) -> tuple[list[dict[str, Indexed]], int]:
    """Count indexed points per file_id, per scope.

    Returns ({file_id: Indexed} for each scope, number of points no scope owns).
    """
    keys = sorted({'file_id', 'source', 'source_type'} | {s.path_key for s in scopes})
    indexed: list[dict[str, Indexed]] = [{} for _ in scopes]
    unowned = 0
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection,
            with_payload=keys,
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            file_id = payload.get('file_id')
            owner = next((i for i, s in enumerate(scopes) if s.owns(payload)), None)
            if not file_id or owner is None:
                unowned += 1
                continue
            path = payload.get(scopes[owner].path_key) or ''
            indexed[owner].setdefault(file_id, Indexed(path)).points += 1
        if offset is None:
            break
    return indexed, unowned


def find_orphans(scope: Scope, indexed: dict[str, Indexed]) -> list[Orphan]:
    orphans = []
    for file_id, entry in indexed.items():
        if file_id in scope.live:
            continue
        if not entry.path:
            reason = 'no path in payload'
        elif (scope.root / entry.path).exists():
            reason = 'on disk but not selected'   # e.g. foo.txt once foo.md exists
        else:
            reason = 'gone from disk'
        orphans.append(Orphan(file_id, entry.path or '?', entry.points, reason))
    return sorted(orphans, key=lambda o: o.path)


def guard_refusal(scope: Scope, indexed_files: int, orphans: int) -> str | None:
    """Why deleting these orphans looks like a mistake, or None if it doesn't."""
    if not orphans:
        return None
    if not scope.live:
        return (f"{scope.label}: no files found, yet {indexed_files} are indexed "
                f"from it. That looks like a missing mount, not an empty folder.")
    if orphans > MASS_DELETE_FLOOR and orphans * 2 > indexed_files:
        return (f"{scope.label}: {orphans} of {indexed_files} indexed files are "
                f"gone. More than half at once looks like a wrong path or a "
                f"partial mount rather than a cleanup.")
    return None


def prune_state(state_file: Path, scopes: list[Scope], dry_run: bool) -> int:
    """Drop file_mtimes entries for files under these roots the loader would
    no longer select. Everything else in the file, last_sync included, stays.

    Keyed the way the loaders key it: str(<root as given> / <relative path>).
    Entries under a root spelled differently are left alone, because this
    cannot tell whether they are live.
    """
    if not state_file.exists():
        logging.info(f"No {state_file.name}, so no sync state to prune")
        return 0
    with open(state_file) as f:
        state = json.load(f)
    mtimes = state.get('file_mtimes') if isinstance(state, dict) else None
    if not isinstance(mtimes, dict):
        logging.warning(f"{state_file.name} has no file_mtimes map; leaving it alone")
        return 0

    live = {str(p) for s in scopes for p in s.live.values()}
    stale = [k for k in mtimes
             if k not in live and any(Path(k).is_relative_to(s.root) for s in scopes)]
    for k in stale:
        logging.debug(f"State entry for a file no longer selected: {k}")
    if stale and not dry_run:
        for k in stale:
            del mtimes[k]
        write_state(state_file, state)
    return len(stale)


def cull(qdrant: QdrantClient, collection: str, scopes: list[Scope],
         state_file: Path, *, name: str, dry_run: bool, force: bool,
         alert: Callable[[str, str], None]) -> int:
    """Delete orphaned chunks and their state entries. Returns an exit status.

    The caller has already claimed state_file with claim_state(), unless this
    is a dry run, and must have done so before building the scopes. A loader
    run that started between reading the disk and the Qdrant scan could
    otherwise write back entries this prunes.
    """
    if dry_run:
        logging.warning("Dry run: nothing will be deleted from Qdrant or the state file")

    if not qdrant.collection_exists(collection):
        logging.warning(f"Collection {collection} does not exist; nothing to cull")
        return 0

    indexed, unowned = scan_collection(qdrant, collection, scopes)
    if unowned:
        logging.info(f"{unowned} point(s) in {collection} have no file_id or belong "
                     f"to another root or loader; left alone")

    plans: list[tuple[Scope, list[Orphan]]] = []
    refusals = []
    for scope, found in zip(scopes, indexed):
        orphans = find_orphans(scope, found)
        logging.info(f"{scope.label}: {len(scope.live)} file(s) on disk, "
                     f"{len(found)} indexed in {collection}, {len(orphans)} orphaned")
        refusal = guard_refusal(scope, len(found), len(orphans))
        if refusal:
            refusals.append(refusal)
        plans.append((scope, orphans))

    refused = bool(refusals) and not force
    verb = "Orphaned, kept:" if refused else "Would remove" if dry_run else "Removing"
    for scope, orphans in plans:
        for o in orphans:
            logging.info(f"{verb} {o.path} ({o.reason}, {o.points} point(s))")

    if refused:
        for r in refusals:
            logging.error(f"Refusing to cull {collection}: {r}")
        logging.error("Nothing was deleted. If the files really are gone, re-run "
                      "with --force.")
        if not dry_run:
            alert(f"{name} refused to cull {collection}",
                  "\n\n".join(refusals) + "\n\nNothing was deleted. Check that the "
                  "directory is mounted and is the one the loader is run with; if "
                  "the files really are gone, re-run with --force.")
        return 1
    for r in refusals:
        logging.warning(f"--force given, culling anyway: {r}")

    failed = 0
    removed_files = removed_points = 0
    for scope, orphans in plans:
        for b in range(0, len(orphans), DELETE_BATCH):
            batch = orphans[b:b + DELETE_BATCH]
            if not dry_run:
                try:
                    qdrant.delete(
                        collection_name=collection,
                        points_selector=Filter(must=[FieldCondition(
                            key='file_id', match=MatchAny(any=[o.file_id for o in batch]))]),
                    )
                except Exception as e:
                    logging.warning(f"Delete failed for {len(batch)} file(s), from "
                                    f"{batch[0].path}: {e}")
                    failed += len(batch)
                    continue
            removed_files += len(batch)
            removed_points += sum(o.points for o in batch)

    dropped = prune_state(state_file, scopes, dry_run)

    suffix = " (dry run)" if dry_run else ""
    logging.warning(f"Cull complete{suffix}: {removed_files} orphaned file(s), "
                    f"{removed_points} point(s) removed from {collection}; "
                    f"{dropped} entr{'y' if dropped == 1 else 'ies'} dropped "
                    f"from {state_file.name}")
    if failed:
        logging.error(f"{failed} orphaned file(s) could not be deleted; the next "
                      f"run will try again")
        return 1
    return 0


def self_test(verbose: bool = False) -> int:
    """Check the cull rules against known cases, using an in-memory Qdrant.

    Same shape as feeds_ingest.py's --self-test. Touches no real collection,
    state file or mail server.
    """
    if not verbose:
        logging.disable(logging.CRITICAL)
    try:
        return _self_test()
    finally:
        logging.disable(logging.NOTSET)


def _self_test() -> int:
    dim = 4

    def add(qdrant, collection, payloads):
        if not qdrant.collection_exists(collection):
            qdrant.create_collection(collection, vectors_config=VectorParams(
                size=dim, distance=Distance.COSINE))
        start = qdrant.count(collection).count
        qdrant.upsert(collection, points=[
            PointStruct(id=start + i, vector=[1.0, 0.5, 0.25, float(i % 7)], payload=p)
            for i, p in enumerate(payloads)])

    def paper_points(rel, n, **extra):
        return [{'file_id': papers_ingest.file_id_for(Path(rel)), 'chunk_index': i,
                 'file_path': rel, 'source': 'paper', **extra} for i in range(n)]

    def summary_points(source_type, rel, n):
        return [{'file_id': summaries_ingest.file_id_for(source_type, Path(rel)),
                 'chunk_index': i, 'summary_file': rel, 'source_type': source_type,
                 'source': 'summary'} for i in range(n)]

    def paths_left(qdrant, collection, key):
        points, _ = qdrant.scroll(collection, limit=10000, with_payload=[key])
        return sorted({p.payload.get(key) for p in points if p.payload.get(key)})

    def touch(root, *rels):
        for rel in rels:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"{rel}\n")

    alerts = []
    def record(subject, body):
        alerts.append(subject)

    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        qdrant = QdrantClient(":memory:")

        # -- papers: deleted, renamed, shadowed, an AI summary embedded as a
        #    document (as papers_ingest.py did before it learned to skip them),
        #    live files, and someone else's points
        papers = tmp / 'misc'
        touch(papers, 'keep.md', 'keep.pdf', 'keep.ai-summary.md',
              'shadow.txt', 'shadow.md', 'sub/renamed_new.md')
        add(qdrant, 'misc',
            paper_points('keep.md', 2) + paper_points('keep.ai-summary.md', 1)
            + paper_points('shadow.txt', 2) + paper_points('shadow.md', 1)
            + paper_points('sub/renamed_old.md', 3) + paper_points('sub/renamed_new.md', 3)
            + paper_points('deleted.md', 2)
            + [{'text': 'no file_id at all'},
               {'file_id': 'f' * 32, 'file_path': 'gone.json', 'source': 'kindle'}])
        state = tmp / '.papers_sync_state_misc.json'
        write_state(state, {
            'file_mtimes': {str(papers / r): 1 for r in (
                'keep.md', 'keep.ai-summary.md', 'shadow.txt', 'shadow.md',
                'sub/renamed_old.md', 'sub/renamed_new.md', 'deleted.md')}
                | {'/elsewhere/x.md': 1},
            'last_sync': '2026-11-12T00:00:00+00:00',
        })
        before_state = state.read_text()
        before_count = qdrant.count('misc').count

        rc = cull(qdrant, 'misc', [papers_scope(papers)], state, name='t',
                  dry_run=True, force=False, alert=record)
        cases += [
            ("a dry run exits 0", rc, 0),
            ("a dry run deletes no points", qdrant.count('misc').count, before_count),
            ("a dry run leaves the state file as it was", state.read_text(), before_state),
        ]

        rc = cull(qdrant, 'misc', [papers_scope(papers)], state, name='t',
                  dry_run=False, force=False, alert=record)
        after = json.loads(state.read_text())
        cases += [
            ("the cull exits 0", rc, 0),
            ("deleted, renamed-away, shadowed and AI-summary files lose their "
             "chunks; the rest stay",
             paths_left(qdrant, 'misc', 'file_path'),
             ['gone.json', 'keep.md', 'shadow.md', 'sub/renamed_new.md']),
            ("so do a point with no file_id and another loader's point",
             qdrant.count('misc').count, 2 + 1 + 3 + 2),
            ("their state entries go, entries outside the root stay",
             sorted(after['file_mtimes']),
             sorted(['/elsewhere/x.md'] + [str(papers / r) for r in (
                 'keep.md', 'shadow.md', 'sub/renamed_new.md')])),
            ("last_sync is not touched", after.get('last_sync'), '2026-11-12T00:00:00+00:00'),
            ("a second run has nothing to do",
             (cull(qdrant, 'misc', [papers_scope(papers)], state, name='t',
                   dry_run=False, force=False, alert=record), qdrant.count('misc').count),
             (0, 8)),
            ("no alert so far", list(alerts), []),
        ]

        # -- guards: an empty root, and a mass delete
        empty = tmp / 'unmounted'
        empty.mkdir()
        add(qdrant, 'thoughts', paper_points('a.md', 1) + paper_points('b.md', 1))
        rc = cull(qdrant, 'thoughts', [papers_scope(empty)], tmp / 'none.json', name='t',
                  dry_run=False, force=False, alert=record)
        cases += [
            ("an empty root refuses", (rc, qdrant.count('thoughts').count), (1, 2)),
            ("and mails", list(alerts), ['t refused to cull thoughts']),
        ]
        rc = cull(qdrant, 'thoughts', [papers_scope(empty)], tmp / 'none.json', name='t',
                  dry_run=False, force=True, alert=record)
        cases.append(("--force overrides it", (rc, qdrant.count('thoughts').count), (0, 0)))

        mass = tmp / 'mass'
        touch(mass, 'live.md')
        add(qdrant, 'mass', paper_points('live.md', 1)
            + [p for i in range(11) for p in paper_points(f'gone{i}.md', 1)])
        rc = cull(qdrant, 'mass', [papers_scope(mass)], tmp / 'none.json', name='t',
                  dry_run=False, force=False, alert=record)
        cases.append(("11 of 12 files gone refuses", (rc, qdrant.count('mass').count), (1, 12)))
        rc = cull(qdrant, 'mass', [papers_scope(mass)], tmp / 'none.json', name='t',
                  dry_run=False, force=True, alert=record)
        cases.append(("--force overrides that too", (rc, qdrant.count('mass').count), (0, 1)))

        small = tmp / 'small'
        touch(small, 'one.md')
        add(qdrant, 'small', paper_points('one.md', 1) + paper_points('two.md', 1)
            + paper_points('three.md', 1))
        rc = cull(qdrant, 'small', [papers_scope(small)], tmp / 'none.json', name='t',
                  dry_run=False, force=False, alert=record)
        cases.append(("2 of 3 gone is under the floor and goes ahead",
                      (rc, paths_left(qdrant, 'small', 'file_path')), (0, ['one.md'])))

        # -- summaries: only the roots given are in scope, and a Mac's
        #    AppleDouble ._ file is not a summary even while it is on disk
        docs = tmp / 'indexed_docs'
        touch(docs, 'misc/keep.ai-summary.md', 'misc/._keep.ai-summary.md')
        add(qdrant, 'summaries',
            summary_points('paper', 'misc/keep.ai-summary.md', 1)
            + summary_points('paper', 'misc/._keep.ai-summary.md', 1)
            + summary_points('paper', 'misc/news_research/gone.ai-summary.md', 2)
            + summary_points('podcast', 'Show/ep1.ai-summary.md', 1))
        sstate = tmp / '.summaries_sync_state.json'
        write_state(sstate, {'file_mtimes': {
            str(docs / 'misc/keep.ai-summary.md'): 1,
            str(docs / 'misc/news_research/gone.ai-summary.md'): 1,
            '/podcasts/Show/ep1.ai-summary.md': 1}})
        rc = cull(qdrant, 'summaries', [summary_scope(docs, 'paper')], sstate, name='t',
                  dry_run=False, force=False, alert=record)
        cases += [
            ("deleted and ._ paper summaries go; podcast summaries are out of scope",
             (rc, paths_left(qdrant, 'summaries', 'summary_file')),
             (0, ['Show/ep1.ai-summary.md', 'misc/keep.ai-summary.md'])),
            ("and so is their state",
             sorted(json.loads(sstate.read_text())['file_mtimes']),
             sorted(['/podcasts/Show/ep1.ai-summary.md',
                     str(docs / 'misc/keep.ai-summary.md')])),
            ("a missing collection is nothing to do",
             cull(qdrant, 'nope', [summary_scope(docs, 'paper')], sstate, name='t',
                  dry_run=False, force=False, alert=record), 0),
        ]

    failures = 0
    for label, got, want in cases:
        if got != want:
            print(f"  FAIL {label}: {got!r} != {want!r}", file=sys.stderr)
            failures += 1
        else:
            print(f"  ok   {label}")
    print("self-test failed" if failures else "self-test passed")
    return 1 if failures else 0
