#!/usr/bin/env python3
"""
sync_state.py - Shared sync-state handling for qdrant-rag loaders

Each loader keeps an incremental sync state file in config/, and every one of
them carries a 'last_sync' timestamp. health_check.py reads that timestamp to
tell a stalled pipeline from a healthy one, which only works if the stamp means
"a run completed" rather than "a run found something to do" -- see
stamp_last_sync() for why that distinction cost a day of silence.

Nothing schedules the loaders exclusively -- cron fires on a timer whether or
not the last run finished, and a run started by hand knows nothing of either.
claim_state() keeps two runs off the same state file, because the alternative
is a lost update that silently re-embeds work already paid for.
"""

import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path


def write_state(path: Path, state: dict) -> None:
    """Write a sync state file atomically.

    health_check.py reads last_sync from these files to decide whether
    ingestion has stalled, so a half-written file would blind the very thing
    watching for trouble. Write beside it and rename, which is atomic on POSIX.

    indent=2 stays. On a ledger of 200k entries it costs about 1.4 MB -- real
    money against a 38 MB file, but hashing the identifiers took that file to
    8 MB, and the remaining padding buys a state file you can still read with
    head(1) instead of one 7 MB line. last_sync is the first thing anyone looks
    for in here when ingestion seems stuck.
    """
    # The pid keeps two writers off one temp path, which matters on the
    # fail-open paths in claim_state() where no lock is held at all.
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        # Leave no half-written file behind. These sit in config/ beside real
        # state, and a feeds state file names every feed URL.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def stamp_last_sync(path: Path, extra: dict | None = None) -> None:
    """Record that a run completed, merging into whatever is already on disk.

    Every loader returns early when it finds nothing new, and used to do so
    before writing state at all. That made last_sync mean "the last run that
    had work to do", which is a useless health signal: a source with nothing
    new for a week is perfectly healthy but looks a week stale, while a source
    whose runs are crashing looks exactly the same. Calling this at the early
    return makes the stamp mean what health_check.py assumes it means.

    Merging rather than replacing matters because these files hold real data
    beside the stamp -- a file_mtimes map, a digest ledger -- and the early
    return has no business rewriting any of it.
    """
    state: dict = {}
    if path.exists():
        try:
            with open(path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                state = loaded
            else:
                logging.warning(f"Ignoring unexpected state shape in {path.name}: "
                                f"{type(loaded).__name__}")
        except (OSError, json.JSONDecodeError) as e:
            # Don't fail a run that otherwise succeeded. The next real sync
            # rewrites this file from scratch anyway, and health_check.py
            # reports an unreadable state file on its own account.
            logging.warning(f"Could not read {path.name} to stamp last_sync: {e}")
            return

    if extra:
        state.update(extra)
    state['last_sync'] = datetime.now(timezone.utc).isoformat()
    write_state(path, state)


# Held for the life of the process on purpose. An flock belongs to the open file
# description, so closing the fd -- or letting it be garbage collected -- hands
# the lock to whoever is waiting, in the middle of our run.
_held_fds: list[int] = []


def claim_state(state_file: Path) -> bool:
    """Claim exclusive use of one state file for this run.

    Returns True when this run may proceed. Returns False, having logged why,
    when another run already holds the file -- the caller should give up rather
    than race it.

    Keyed on the state file rather than on the loader or its command line,
    because the two do not correspond: papers_ingest writes a different file per
    --collection and those runs must not block each other, while
    summaries_ingest takes the same flag and writes one file regardless, so
    those runs must. The file is the resource; the file is the lock.

    Never released explicitly. The kernel drops an flock when the process exits
    -- cleanly, on an unhandled exception, or on SIGKILL -- so a killed run
    leaves nothing stale behind and there is no unwinding for run_guarded() to
    get wrong.
    """
    if os.environ.get('QDRANT_LOADER_NO_LOCK', '').strip():
        logging.warning("QDRANT_LOADER_NO_LOCK is set; not claiming "
                        f"{state_file.name}. Overlapping runs can lose state.")
        return True

    lock_path = state_file.with_name(state_file.name + '.lock')
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as e:
        # Fail open. A lock we cannot create is a reason to warn, not a reason
        # to stop ingesting -- and on a filesystem where flock misbehaves,
        # QDRANT_LOADER_NO_LOCK is the deliberate way to say so.
        logging.warning(f"Could not open {lock_path.name} ({e}); "
                        f"proceeding without a lock")
        return True

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = _lock_holder(fd)
        os.close(fd)
        logging.warning(f"{holder} already holds {state_file.name}; skipping this "
                        f"run rather than racing it. Nothing is wrong if this is "
                        f"occasional; if every run skips, the sync has stalled and "
                        f"./run.sh health will say so.")
        return False
    except OSError as e:
        # Only EWOULDBLOCK means someone else has it. Anything else -- no lock
        # records left, or a filesystem that does not implement flock, which is
        # the usual NFS story -- means locking is unavailable here. Treating that
        # as contention would skip every run forever while blaming a phantom
        # process, so fail open and say so plainly.
        os.close(fd)
        logging.warning(f"Cannot lock {lock_path.name} ({e}); proceeding without "
                        f"a lock. Overlapping runs are possible until this is "
                        f"resolved.")
        return True

    _held_fds.append(fd)
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
    except OSError as e:
        # The lock is held, which is the part that matters; the pid is only so a
        # skipped run can name who it yielded to.
        logging.debug(f"Could not record pid in {lock_path.name}: {e}")
    return True


def _lock_holder(fd: int) -> str:
    """Describe whoever holds the lock, for the skip message."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        # errors='replace': whatever is in there, reading it must not raise.
        # UnicodeDecodeError is a ValueError, so it would sail past an OSError
        # handler and turn a routine skip into a crash alert.
        pid = os.read(fd, 32).decode('utf-8', 'replace').strip()
    except OSError:
        pid = ''
    return f"pid {pid}" if pid.isdigit() else "another run"
