"""This module provides functions to load JSON files from a default location or a specific path."""

import importlib.resources as pkg_resources
import json
import os

from .logger import Logger


def load_default_json(file_name: str, default_package: str = '..data'):
    """Load a JSON file from the default location.

    Args:
        file_name: The name of the JSON file to load.
        default_package: The package to load the JSON file from.

    Returns:
        The JSON data from the file.
    """
    with pkg_resources.files(default_package).joinpath(file_name).open('r') as f:
        return json.load(f)


def load_json(file_path: str):
    """Load a JSON file from a specific path or the default location.

    Args:
        file_path: The path to the JSON file to load.

    Returns:
        The JSON data from the file.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, IOError):
        try:
            # Extract filename from path and try loading from default location
            file_name = os.path.basename(file_path)
            logger = Logger.get_logger('JsonLoader')
            logger.warning(f"File not found at '{file_path}', falling back to default location")
            return load_default_json(file_name)
        except Exception as e:
            raise FileNotFoundError(f"Could not load JSON file from '{file_path}' or default location") from e
