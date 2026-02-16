#!/usr/bin/env python3
"""Project logging setup helpers."""

from __future__ import annotations

import logging
import sys


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def configure_logging(level: int = logging.INFO) -> None:
    """Route INFO/WARNING to stdout and ERROR/CRITICAL to stderr."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if getattr(root_logger, "_fuse_logging_configured", False):
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(_MaxLevelFilter(logging.WARNING))
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    root_logger.handlers = [stdout_handler, stderr_handler]
    root_logger._fuse_logging_configured = True
