from pathlib import Path
from config import settings

__all__ = [
    "get_project_root",
    "get_data_dir",
    "get_raw_dir",
    "get_processed_dir",
    "get_cache_dir",
]


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return settings.BASE_DIR


def get_data_dir() -> Path:
    """Return the top-level data directory."""
    return settings.DATA_DIR


def get_raw_dir() -> Path:
    """Return the directory containing raw data files."""
    return settings.RAW_DATA_DIR


def get_processed_dir() -> Path:
    """Return the directory for processed data files."""
    return settings.PROCESSED_DATA_DIR


def get_cache_dir() -> Path:
    """Return the directory used for cached data."""
    return settings.CACHE_DIR
