#!/usr/bin/env python3
"""
alert.py - Shared alert utility for qdrant-rag loaders

Sends a plain-text email via SMTP when a loader encounters a fatal error.
All config is optional — if ALERT_SMTP_HOST is not set, silently does nothing.

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
from email.mime.text import MIMEText
from datetime import datetime, timezone


def send_alert(subject: str, body: str) -> None:
    """Send a plain-text alert email. No-ops silently if SMTP not configured."""
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
