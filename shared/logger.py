import logging, os

def setup_logging(name: str) -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    )
    return logging.getLogger(name)
