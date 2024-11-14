import logging

from rich.logging import RichHandler

LOG_LEVEL = logging.INFO

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True))
logger.setLevel(LOG_LEVEL)
