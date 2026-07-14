"""Central logging configuration for INFOBOT.

Every module should get its own named logger via:

    from src.logger import get_logger
    logger = get_logger(__name__)

Handlers (console + rotating file) are attached to the root logger exactly
once, the first time get_logger() is called; every module logger then
propagates up to those shared handlers, so there's a single place that
controls format, level, and where logs go.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "infobot.log"

_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 3  # keep infobot.log + 3 rotated backups

_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Third-party libraries are chatty at INFO (e.g. httpx logs every HTTP
    # request made during model downloads). Keep them at WARNING so
    # application logs from src.* aren't drowned out; this doesn't affect
    # what our own loggers emit.
    for noisy_logger in ("httpx", "httpcore", "urllib3", "sentence_transformers", "filelock"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring shared handlers on first use."""
    _configure_root_logger()
    return logging.getLogger(name)
