#!/usr/bin/env python3
"""
alert.py - Shared alert utility for qdrant-rag loaders

Sends a plain-text email via SMTP when a loader encounters a fatal error.
All config is optional — if ALERT_SMTP_HOST is not set, silently does nothing.

Two entry points:
    send_alert()   Send one email. Called by each loader's own log_fatal().
    run_guarded()  Wrap a loader's main() so that a failure nobody anticipated
                   still sends mail instead of dying as a bare traceback.

Configuration (from .env):
    ALERT_SMTP_HOST   SMTP server hostname (leave blank to disable alerting)
    ALERT_SMTP_PORT   SMTP port (default: 587)
    ALERT_SMTP_USER   SMTP username
    ALERT_SMTP_PASS   SMTP password
    ALERT_FROM        Sender address (defaults to ALERT_SMTP_USER)
    ALERT_TO          Comma-separated recipient addresses
"""

import smtplib
import socket
import logging
import os
import sys
import traceback
from email.mime.text import MIMEText
from datetime import datetime, timezone
from pathlib import Path


_env_loaded = False


def _ensure_env() -> None:
    """Load config/.env if the caller hasn't already.

    Every loader calls load_dotenv() near the top of main(), so normally this
    finds the settings already in place. It matters for run_guarded(): an
    exception raised before that call would otherwise alert into a void, since
    this module reads os.environ only. Memoised, so it costs one read per run.
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    try:
        from dotenv import load_dotenv
        config_dir = Path(os.environ.get('QDRANT_LOADER_CONFIG_DIR',
                                         Path(__file__).parent.parent / 'config'))
        load_dotenv(config_dir / '.env')  # never overrides an existing value
    except Exception as e:
        logging.debug(f"Could not load .env for alerting: {e}")


def send_alert(subject: str, body: str) -> None:
    """Send a plain-text alert email. No-ops silently if SMTP not configured."""
    _ensure_env()
    host = os.environ.get('ALERT_SMTP_HOST', '').strip()
    if not host:
        return  # alerting not configured

    port = int(os.environ.get('ALERT_SMTP_PORT', 587))
    user = os.environ.get('ALERT_SMTP_USER', '')
    password = os.environ.get('ALERT_SMTP_PASS', '')
    from_addr = os.environ.get('ALERT_FROM', user)
    to_addrs = [a.strip() for a in os.environ.get('ALERT_TO', '').split(',') if a.strip()]

    if not to_addrs:
        logging.warning("ALERT_TO not set; skipping alert email")
        return

    hostname = socket.gethostname()
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    full_body = f"Host: {hostname}\nTime: {timestamp}\n\n{body}"

    msg = MIMEText(full_body, 'plain')
    msg['Subject'] = subject
    msg['From'] = from_addr

    try:
        server = smtplib.SMTP(host, port)
        server.ehlo()
        server.starttls()
        if user and password:
            server.login(user, password)
        for to in to_addrs:
            msg['To'] = to
            server.sendmail(from_addr, to, msg.as_string())
            del msg['To']
            logging.info(f"Alert sent to {to}")
        server.quit()
    except Exception as e:
        logging.error(f"Failed to send alert email: {e}")
        # Never raise — alerting failure must not mask the original error


def run_guarded(loader_name: str, main_fn) -> int:
    """Run a loader's main(), emailing a traceback for anything it didn't handle.

    Every loader has a log_fatal() for the failures it anticipates, but an
    unanticipated exception used to escape main() and die as a bare traceback on
    stderr -- which cron redirects into a log file nobody reads. That is how an
    invalid feeds.yaml stopped the news index for a day without a word. This
    catches whatever log_fatal() never saw.

    log_fatal() exits via sys.exit, raising SystemExit. That is a BaseException,
    not an Exception, so it passes straight through here and a handled fatal
    error still sends exactly one email.
    """
    try:
        return main_fn()
    except KeyboardInterrupt:
        return 1
    except Exception:
        tb = traceback.format_exc()
        logging.critical(f"Unhandled exception in {loader_name}:\n{tb}")
        send_alert(
            subject=f"[ALERT] {loader_name} crashed on {socket.gethostname()}",
            body=f"Command: {' '.join(sys.argv)}\n\n"
                 f"The run died on an error it does not handle explicitly "
                 f"and did not finish.\n\n{tb}",
        )
        return -1
