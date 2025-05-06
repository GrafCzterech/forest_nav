import logging
from typing import Iterable, Any
import argparse

from isaaclab.app import AppLauncher


def setup_logging(
    targets: Iterable[str], level: int = logging.DEBUG
) -> logging.Logger:
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
