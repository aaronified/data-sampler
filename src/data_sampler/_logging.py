"""Central logging setup, controlled by the DATA_SAMPLER_LOG env var.

Levels: ``quiet`` (warnings only), ``info``, ``verbose`` (debug).
Defaults to ``verbose`` during development. Set ``DATA_SAMPLER_LOG=quiet``
to silence, or ``DATA_SAMPLER_LOG_FILE=<path>`` to divert output to a file
(the TUI does this automatically so log lines cannot corrupt the display).
"""

from __future__ import annotations

import logging
import os
import sys

_LEVELS = {
    "quiet": logging.WARNING,
    "info": logging.INFO,
    "verbose": logging.DEBUG,
    "debug": logging.DEBUG,
}

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger("data_sampler")
    level_name = os.environ.get("DATA_SAMPLER_LOG", "verbose").lower()
    root.setLevel(_LEVELS.get(level_name, logging.DEBUG))

    log_file = os.environ.get("DATA_SAMPLER_LOG_FILE")
    handler: logging.Handler
    if log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a package logger; configures handlers on first use."""
    _configure()
    return logging.getLogger(name)


def redirect_to_file(path: str) -> None:
    """Swap all package log handlers for a file handler (used by the TUI)."""
    root = logging.getLogger("data_sampler")
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root.addHandler(handler)
