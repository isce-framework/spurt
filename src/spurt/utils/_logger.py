import logging
import os

loglevel = os.getenv("SPURT_LOGLEVEL", "INFO")
logger = logging.getLogger("spurt")
logger.setLevel(getattr(logging, loglevel))

_handler = logging.StreamHandler()
_fmt = "%(asctime)s [%(process)d] [%(levelname)s] %(name)-3s: %(message)s"
_handler.setFormatter(logging.Formatter(_fmt))

logger.addHandler(_handler)
