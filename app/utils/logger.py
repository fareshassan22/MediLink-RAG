import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

# Create logs directory if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
    f"medilink_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
)

logger = logging.getLogger("MediLink")
logger.setLevel(logging.INFO)


class JSONLineFileHandler(logging.Handler):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception:
            pass


formatter = logging.Formatter('%(message)s')
file_handler = JSONLineFileHandler(log_filename)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def log_request(data: Dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **data
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
