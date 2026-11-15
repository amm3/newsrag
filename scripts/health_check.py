#!/usr/bin/env python3
"""
health_check.py - Alert when an ingest source stops keeping itself up to date

The loaders alert when they fail loudly. This catches the quieter failure: a
source that simply stops advancing, whether because a run is crashing, because
cron stopped firing it, or because something upstream went away. Every loader
stamps 'last_sync' into its state file on a run that completes, so the absence
of a recent stamp is the signal.

Run it from its own cron line, more often than the ingests themselves:
    */30 * * * * /home/adam/qdrant_loader/run.sh health

Configuration (from .env, all optional):
    HEALTH_CHECKS             Comma-separated sources to watch (default: feeds)
                              One or more of: feeds, wallabag, podcasts, papers,
                              summaries, kindle, openwebui
    HEALTH_STALE_HOURS        Alert once a source is this far behind (default: 36)
    HEALTH_STALE_HOURS_<NAME> Per-source override, e.g. HEALTH_STALE_HOURS_FEEDS
    HEALTH_REALERT_HOURS      Minimum gap between repeat alerts for the same
                              source, so a frequent cron doesn't send mail every
                              run while something stays broken (default: 24)

Alerting itself is the shared SMTP config in alert.py; leave ALERT_SMTP_HOST
blank and this still runs and still sets its exit code, it just sends nothing.
"""

import sys
import os
import argparse
import logging
import json
import re
import socket
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from alert import send_alert, run_guarded
from sync_state import claim_state, write_state

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'

DEFAULT_STALE_HOURS = 36
DEFAULT_REALERT_HOURS = 24

# source name -> description. Only loaders that keep a sync state file appear
# here; the rest track their progress elsewhere or write no state at all, so
# there is no timestamp to read.
SOURCES = {
    'feeds':     'news feed ingest',
    'wallabag':  'Wallabag article ingest',
    'podcasts':  'podcast transcript ingest',
    'papers':    'papers ingest',
    'summaries': 'AI summary ingest',
    'kindle':    'Kindle highlight ingest',
    'openwebui': 'OpenWebUI chat ingest',
}

# config/ filenames to look for, newest wins. One each; papers is absent on
# purpose -- it is run once per collection and keeps a state file per
# collection, so _papers_checks() discovers its filenames instead.
STATE_FILES = {
    'feeds':     ('.feeds_sync_state.json',),
    'wallabag':  ('.wallabag_sync_state.json',),
    'podcasts':  ('.podcast_sync_state.json',),
    'summaries': ('.summaries_sync_state.json',),
    'kindle':    ('.kindle_sync_state.json',),
    'openwebui': ('.openwebui_sync_state.json',),
}


# One thing to watch: a label to report it by, a description for the alert, and
# the state files to look in (newest wins, where more than one spelling exists).
Check = namedtuple('Check', 'label description paths')


def resolve_checks(names: list[str], config_dir: Path) -> tuple[list[Check], list[str]]:
    """Turn configured source names into the concrete things to check.

    Mostly one-to-one, but papers_ingest is run once per collection and keeps a
    separate state file for each, so a bare 'papers' expands to one check per
    collection found. They have to stay separate: collapsing them would let a
    collection that stopped syncing hide behind one that is still running.

    'papers:<collection>' names a single one, which is also how to watch a
    collection that has not written state yet.
    """
    checks: list[Check] = []
    unknown: list[str] = []

    for name in names:
        source, _, qualifier = name.partition(':')
        if source not in SOURCES:
            unknown.append(name)
        elif source == 'papers':
            checks.extend(_papers_checks(config_dir, qualifier))
        elif qualifier:
            unknown.append(f"{name} ('{source}' takes no ':collection')")
        else:
            checks.append(Check(source, SOURCES[source],
                                tuple(config_dir / n for n in STATE_FILES[source])))

    return checks, unknown


def _papers_checks(config_dir: Path, collection: str = '') -> list[Check]:
    """One check per papers collection."""
    description = SOURCES['papers']
    if collection:
        return [Check(f'papers:{collection}', f'{description} ({collection})',
                      (config_dir / f'.papers_sync_state_{collection}.json',))]

    prefix, suffix = '.papers_sync_state_', '.json'
    found = [Check(f'papers:{p.name[len(prefix):-len(suffix)]}',
                   f'{description} ({p.name[len(prefix):-len(suffix)]})', (p,))
             for p in sorted(config_dir.glob(f'{prefix}*{suffix}'))]

    # papers_ingest renames this to the per-collection name the first time it
    # runs for the default collection, so it only lingers on an older install.
    legacy = config_dir / '.papers_sync_state.json'
    if legacy.exists():
        found.append(Check('papers', f'{description} (pre-collection state file)',
                           (legacy,)))

    if not found:
        # Nothing has ever synced. Report against the collection papers_ingest
        # would use, so the alert names a real file rather than a glob.
        collection = os.environ.get('PAPERS_COLLECTION', 'papers')
        found.append(Check(f'papers:{collection}', f'{description} ({collection})',
                           (config_dir / f'.papers_sync_state_{collection}.json',)))
    return found


def _threshold_for(label: str, default: float) -> float:
    """Stale threshold for one check: the most specific override wins.

    papers:my_thoughts takes HEALTH_STALE_HOURS_PAPERS_MY_THOUGHTS, falling back
    to HEALTH_STALE_HOURS_PAPERS for all collections, then HEALTH_STALE_HOURS.
    """
    specific = re.sub(r'[^A-Z0-9]+', '_', label.upper())
    source = label.partition(':')[0].upper()
    names = [f'HEALTH_STALE_HOURS_{specific}']
    if source != specific:
        names.append(f'HEALTH_STALE_HOURS_{source}')
    return _env_float(*names, default=default)


def main():
    parser = argparse.ArgumentParser(
        description='Alert when an ingest source has stopped keeping up to date')
    parser.add_argument("-v", action="store_true", default=False, help="Print extra info")
    parser.add_argument("-vv", action="store_true", default=False, help="Print (more) extra info")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report findings but send no mail and don't record them")
    parser.add_argument("--force", action="store_true",
                        help="Ignore HEALTH_REALERT_HOURS and alert about every problem found")
    args = parser.parse_args()

    if args.vv:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.DEBUG)
    elif args.v:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.INFO)
    else:
        logging.basicConfig(format=LOGGING_FORMAT, datefmt=DEFAULT_TIME_FORMAT, level=logging.WARNING)

    # alert.py reads os.environ only, so the SMTP settings have to be loaded here
    # or send_alert() would silently no-op.
    config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR',
                                     Path(__file__).parent.parent / 'config'))
    load_dotenv(config_dir / '.env')

    checks = [c.strip().lower() for c in os.environ.get('HEALTH_CHECKS', 'feeds').split(',')
              if c.strip()]
    if not checks:
        log_fatal("HEALTH_CHECKS is set but empty; nothing to check")

    resolved, unknown = resolve_checks(checks, config_dir)
    if unknown:
        log_fatal(f"Unknown source(s) in HEALTH_CHECKS: {', '.join(unknown)}\n"
                  f"Known sources: {', '.join(sorted(SOURCES))}\n"
                  f"papers may also be given as papers:<collection>")
    if not resolved:
        log_fatal("HEALTH_CHECKS resolved to nothing to check")

    realert_hours = _env_float('HEALTH_REALERT_HOURS', default=DEFAULT_REALERT_HOURS)
    default_stale = _env_float('HEALTH_STALE_HOURS', default=DEFAULT_STALE_HOURS)
    logging.info(f"Checking {len(resolved)}: {', '.join(c.label for c in resolved)}")

    health_state_file = config_dir / '.health_state.json'

    # Another run may already be working on this file: cron fires on a timer
    # whether or not the last run finished, and a run started by hand knows
    # nothing of either. Racing would lose one of the two runs' state.
    if not args.dry_run and not claim_state(health_state_file):
        return 0
    alerted = _load_health_state(health_state_file)
    now = datetime.now(timezone.utc)

    problems = []       # (source, detail) currently failing
    recovered = []      # sources that were failing and now aren't

    for chk in resolved:
        ok, detail = check_state(chk, _threshold_for(chk.label, default_stale))
        if ok:
            logging.info(f"{chk.label}: ok — {detail}")
            if chk.label in alerted:
                recovered.append((chk.label, detail))
        else:
            logging.warning(f"{chk.label}: PROBLEM — {detail}")
            problems.append((chk.label, detail))

    # Report only what hasn't been reported recently, so a half-hourly cron
    # doesn't mail every run for as long as something stays broken.
    to_report, suppressed = [], []
    for source, detail in problems:
        last = _parse_timestamp(alerted.get(source))
        age_hours = (now - last).total_seconds() / 3600 if last else None
        if args.force or age_hours is None or age_hours >= realert_hours:
            to_report.append((source, detail))
        else:
            suppressed.append((source, detail))
            logging.info(f"{source}: alert suppressed, last sent "
                         f"{age_hours:.1f}h ago (HEALTH_REALERT_HOURS={realert_hours:g})")

    if to_report:
        _send_problem_alert(to_report, suppressed, realert_hours, args.dry_run)
        if not args.dry_run:
            for source, _ in to_report:
                alerted[source] = now.isoformat()

    if recovered:
        _send_recovery_alert(recovered, args.dry_run)
        if not args.dry_run:
            for source, _ in recovered:
                alerted.pop(source, None)

    if not args.dry_run:
        _save_health_state(health_state_file, alerted)

    if problems:
        logging.warning(f"Health check: {len(problems)} of {len(resolved)} sources unhealthy "
                        f"({', '.join(s for s, _ in problems)})")
        return 1

    logging.warning(f"Health check: all {len(resolved)} sources healthy "
                    f"({', '.join(c.label for c in resolved)})")
    return 0


def check_state(chk: Check, threshold_hours: float) -> tuple[bool, str]:
    """Decide whether one check has synced recently enough.

    Returns (ok, human-readable detail) — the detail goes straight into the
    alert body, so it says both what is wrong and what the threshold was.
    """
    description = chk.description

    existing = [p for p in chk.paths if p.exists()]
    if not existing:
        names = ' or '.join(p.name for p in chk.paths)
        return False, (f"{description}: no state file in {chk.paths[0].parent} "
                       f"({names}). It has never completed a sync, or its state "
                       f"was deleted.")
    # Newest wins, where a loader has more than one state filename.
    state_file = max(existing, key=lambda p: p.stat().st_mtime)

    try:
        with open(state_file) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return False, (f"{description}: state file {state_file.name} is unreadable: {e}. "
                       f"The next run will treat it as a full re-sync.")

    source_of_time = 'last_sync'
    last_sync = _parse_timestamp(state.get('last_sync') if isinstance(state, dict) else None)
    if last_sync is None:
        # feeds only started stamping last_sync when this check was added, and a
        # state file written by an older build has none. Its mtime is the next
        # best thing: it is rewritten on every completed run.
        last_sync = datetime.fromtimestamp(state_file.stat().st_mtime, tz=timezone.utc)
        source_of_time = 'state file mtime'

    age_hours = (datetime.now(timezone.utc) - last_sync).total_seconds() / 3600
    when = f"{last_sync.strftime('%Y-%m-%d %H:%M:%S %Z')} ({source_of_time})"

    if age_hours > threshold_hours:
        return False, (f"{description}: last completed sync {when}, "
                       f"{age_hours:.1f}h ago -- over the {threshold_hours:g}h threshold.")
    return True, f"{description}: last completed sync {when}, {age_hours:.1f}h ago."


def _send_problem_alert(to_report, suppressed, realert_hours: float, dry_run: bool) -> None:
    names = ', '.join(source for source, _ in to_report)
    subject_plural = ('1 ingest source is' if len(to_report) == 1
                      else f'{len(to_report)} ingest sources are')
    body = (f"{subject_plural} not keeping up to date on this host:\n\n"
            + "\n".join(f"  - {detail}" for _, detail in to_report))
    if suppressed:
        body += ("\n\nAlso still unhealthy, but already reported within the last "
                 f"{realert_hours:g}h:\n"
                 + "\n".join(f"  - {detail}" for _, detail in suppressed))
    body += ("\n\nCheck the ingest log on this host for the failing run. This alert repeats "
             f"at most every {realert_hours:g}h per source until the source syncs again.")

    subject = f"[ALERT] ingest stale on {socket.gethostname()}: {names}"
    if dry_run:
        logging.warning(f"[dry-run] would send: {subject}\n{body}")
        return
    send_alert(subject=subject, body=body)


def _send_recovery_alert(recovered, dry_run: bool) -> None:
    names = ', '.join(source for source, _ in recovered)
    plural = 'source is' if len(recovered) == 1 else 'sources are'
    body = (f"Previously stale ingest {plural} syncing again:\n\n"
            + "\n".join(f"  - {detail}" for _, detail in recovered))

    subject = f"[OK] ingest recovered on {socket.gethostname()}: {names}"
    if dry_run:
        logging.warning(f"[dry-run] would send: {subject}\n{body}")
        return
    send_alert(subject=subject, body=body)


def _load_health_state(path: Path) -> dict:
    """Read the per-source last-alert times. A damaged file is not worth failing
    over: start clean, which at worst re-sends one alert."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning(f"Ignoring unreadable health state {path}: {e}")
        return {}
    alerted = state.get('alerted') if isinstance(state, dict) else None
    return alerted if isinstance(alerted, dict) else {}


def _save_health_state(path: Path, alerted: dict) -> None:
    try:
        write_state(path, {'alerted': alerted,
                           'updated': datetime.now(timezone.utc).isoformat()})
    except OSError as e:
        # Losing suppression state only means a repeated alert; not worth exiting over.
        logging.error(f"Could not write health state {path}: {e}")


def _parse_timestamp(value) -> datetime | None:
    """Parse an ISO timestamp from a state file, treating a naive one as UTC."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        logging.debug(f"Unparseable timestamp: {value!r}")
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _env_float(*names: str, default: float) -> float:
    """First of these env vars with a non-empty value wins.

    '0' is a value, not an absence -- HEALTH_STALE_HOURS=0 is the simplest way
    to make the check fire on demand.
    """
    for name in names:
        raw = os.environ.get(name, '').strip()
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            logging.warning(f"{name}={raw!r} is not a number; using {default:g}")
            return default
    return default


def log_fatal(msg, exit_code=-1):
    logging.critical(f"Fatal Err: {msg}")
    send_alert(
        subject=f"[ALERT] health_check failed on {socket.gethostname()}",
        body=msg
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    sys.exit(run_guarded('health_check', main))
