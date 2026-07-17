import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("data/logs")
LOG_FILE = LOG_DIR / "app.log"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once: console output for live viewing, plus
    a rotating file handler so logs survive across restarts without growing
    unbounded. Every module's `logging.getLogger(__name__)` calls inherit
    this configuration automatically since they propagate up to the root
    logger. Safe to call more than once (clears existing handlers first) so
    it can't accidentally double-configure logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
