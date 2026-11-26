"""Structured JSON logging helpers shared by API and training entry points."""

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Render a log record as one machine-readable JSON object."""

    def format(self, record):
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        standard_fields = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
        for key, value in record.__dict__.items():
            if key not in standard_fields:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False)


def get_json_logger(name):
    """Return a logger with a single JSON stream handler."""
    logger = logging.getLogger(name)
    if not any(getattr(handler, "_agrishield_json", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._agrishield_json = True
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
