"""Wrapper that delegates workout transformation to long_format_transformer."""
from __future__ import annotations

import logging
import sys

from .long_format_transformer import transform_workouts


def main() -> None:
    transform_workouts()


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    main()
