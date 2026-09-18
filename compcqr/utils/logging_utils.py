import logging
import sys

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def get_logger(name: str = "compcqr") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
