"""Various logging modules."""
import logging


def get_logger():
    """Initialize Python logger that outputs to file and console."""
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("run.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(filename)12s | %(levelname)8s | %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
