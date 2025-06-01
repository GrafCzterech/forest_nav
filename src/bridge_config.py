import logging
from typing import Iterable, Any
import argparse

# this script is a centralized place to set up the logging configuration

from isaaclab.app import AppLauncher


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

    for logger in targets:
        logger = logging.getLogger(logger)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def setup() -> tuple[Any, Any]:
    """Initialize Isaac Sim
    Convenience function to reduce boilerplate code in the scripts.

    Returns:
        tuple[object, object]: simulation_app, args_cli
    """
    parser = argparse.ArgumentParser(
        description="Basic scene loading from NeuroforgeLab Interface"
    )

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    simulation_app = AppLauncher(args_cli).app
    return simulation_app, args_cli
