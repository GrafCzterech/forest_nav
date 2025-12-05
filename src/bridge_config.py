import logging
from typing import Iterable


def setup_logging(targets: Iterable[str], level: int = logging.DEBUG) -> logging.Logger:
    """Set up logging for the given targets.
    This function creates a stream handler with a specific format and adds it to
    the specified loggers. The loggers are set to the specified logging level.

    Mainly a convenience function to reduce boilerplate code in the scripts.

    Args:
        targets (Iterable[str]): List of logger names to set up.
        level (int, optional): The level to set the loggers to. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: The last logger that was set up.
    """
    handler = logging.StreamHandler()

    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s: %(message)s"
        )
    )

    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(level)

    for logger in targets:
        logger = logging.getLogger(logger)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
