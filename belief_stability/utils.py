"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Utilities

File        : utils.py

Description
-----------
Common utility functions shared across the Belief
Stability module.

These functions are generic and contain no business logic.
---------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_logger(
    name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger.

    Parameters
    ----------
    name : str
        Logger name.

    level : int
        Logging level.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if not logger.handlers:

        logger.setLevel(level)

        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s - %(message)s"
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


def load_json(filepath: str | Path) -> Any:
    """
    Load a JSON file.

    Parameters
    ----------
    filepath : str | Path

    Returns
    -------
    Any
    """

    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Any,
    filepath: str | Path,
    indent: int = 4,
) -> None:
    """
    Save data as JSON.

    Parameters
    ----------
    data : Any

    filepath : str | Path

    indent : int
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with filepath.open("w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=indent,
            ensure_ascii=False,
        )


def ensure_directory(
    directory: str | Path,
) -> Path:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    directory : str | Path

    Returns
    -------
    Path
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def normalize_whitespace(
    text: str,
) -> str:
    """
    Collapse multiple whitespaces into one.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    return " ".join(text.split())


def safe_lower(
    text: str,
) -> str:
    """
    Lowercase after stripping whitespace.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    return text.strip().lower()


def is_empty(
    text: str | None,
) -> bool:
    """
    Check whether a string is empty.

    Parameters
    ----------
    text : str | None

    Returns
    -------
    bool
    """

    return text is None or text.strip() == ""


def pretty_print(title: str) -> None:
    """
    Print a formatted section header.

    Parameters
    ----------
    title : str
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)