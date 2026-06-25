from __future__ import annotations

import logging
from typing import Any


def configure_logging(level: int | str = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("posteditbench")


def get_logger(name: str, **extra: Any) -> logging.Logger:
    logger = logging.getLogger(f"posteditbench.{name}")
    for key, value in extra.items():
        setattr(logger, key, value)
    return logger
