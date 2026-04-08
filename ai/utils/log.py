"""
log.py — minimal structured JSON logger for AIOI.

Design goals:
  - One import, one call: log.info("event", key=value)
  - Outputs newline-delimited JSON to stdout (CloudWatch-compatible)
  - Never logs free-text fields (chunk_text, adjuster_notes, narratives)
  - No external dependencies beyond stdlib

Usage:
  from ai.utils.log import get_logger
  log = get_logger(__name__)
  log.info("model_trained", model="fraud", roc_auc=0.91, rows=5000)
  log.error("db_connect_failed", error=str(e))
"""
import json
import logging
import os
import sys
import time
from typing import Any


class _JsonHandler(logging.Handler):
    """Emits one JSON object per log record to stdout."""

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra kwargs passed via log.info("msg", key=val)
        if hasattr(record, "extra"):
            payload.update(record.extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        print(json.dumps(payload, default=str), file=sys.stdout, flush=True)


class _StructuredLogger(logging.Logger):
    """Extends Logger to accept keyword arguments as structured fields."""

    def _log_structured(self, level: int, msg: str, **kwargs: Any) -> None:
        if self.isEnabledFor(level):
            record = self.makeRecord(
                self.name, level, "(unknown)", 0, msg, (), None
            )
            record.extra = kwargs  # type: ignore[attr-defined]
            self.handle(record)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_structured(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_structured(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_structured(logging.ERROR, msg, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_structured(logging.DEBUG, msg, **kwargs)


_initialized = False


def get_logger(name: str) -> _StructuredLogger:
    """Return a named structured logger. Safe to call multiple times."""
    global _initialized
    if not _initialized:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.setLoggerClass(_StructuredLogger)
        root = logging.getLogger()
        if not root.handlers:
            root.addHandler(_JsonHandler())
        root.setLevel(level)
        _initialized = True
    logging.setLoggerClass(_StructuredLogger)
    return logging.getLogger(name)  # type: ignore[return-value]