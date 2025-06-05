#!/usr/bin/env python
"""Run various stages of the horse racing data pipeline."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from horse_racing_project.config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    DRF_PATTERN,
)
from horse_racing_project.src.parsers.bris_spec_new import main as parse_bris_main
from horse_racing_project.src.transformers.current_race_info import main as create_current_info_main
from horse_racing_project.src.transformers.transform_workouts import main as transform_workouts_main
from horse_racing_project.src.transformers.transform_past_starts import main as transform_past_starts_main
from horse_racing_project.src.transformers.feature_engineering import main as engineer_features_main

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def find_latest_drf_file() -> Path:
    """Return the most recent DRF file in the raw data directory."""
    logger.info("Searching for DRF files in %s with pattern %s", RAW_DATA_DIR, DRF_PATTERN)
    drf_files = list(RAW_DATA_DIR.glob(DRF_PATTERN))
    if not drf_files:
        raise FileNotFoundError(
            f"No DRF files found matching {DRF_PATTERN} in {RAW_DATA_DIR}"
        )
    latest_file = max(drf_files, key=lambda p: p.stat().st_mtime)
    logger.info("Using DRF file: %s", latest_file.name)
    return latest_file


def stage_parse_drf():
    parse_bris_main(drf_file_path_arg=find_latest_drf_file())


def stage_current_info():
    create_current_info_main()


def stage_workouts():
    transform_workouts_main()


def stage_past_starts():
    transform_past_starts_main()


def stage_feature_engineering():
    engineer_features_main()


def stage_fitness_metrics():
    from horse_racing_project.src.transformers.advanced_fitness_metrics import main as run_fitness_metrics

    run_fitness_metrics()


def stage_workout_analysis():
    from horse_racing_project.src.transformers.sophisticated_workout_analysis import main as run_workout_analysis

    run_workout_analysis()


def stage_pace_projection():
    from horse_racing_project.src.transformers.advanced_pace_projection import main as run_pace_analysis

    run_pace_analysis()


def stage_class_assessment():
    from horse_racing_project.src.transformers.multi_dimensional_class_assessment import main as run_class_assessment

    run_class_assessment()


def stage_form_cycle():
    from horse_racing_project.src.transformers.form_cycle_detector import main as run_form_cycle

    run_form_cycle()


def stage_integrated_analytics():
    from horse_racing_project.src.transformers.integrated_analytics_system import main as run_integrated_analytics

    run_integrated_analytics()


STAGE_FUNCTIONS = {
    "parse": stage_parse_drf,
    "current_info": stage_current_info,
    "workouts": stage_workouts,
    "past_starts": stage_past_starts,
    "feature_eng": stage_feature_engineering,
    "fitness_metrics": stage_fitness_metrics,
    "workout_analysis": stage_workout_analysis,
    "pace_projection": stage_pace_projection,
    "class_assessment": stage_class_assessment,
    "form_cycle": stage_form_cycle,
    "integrated_analytics": stage_integrated_analytics,
}


def run_complete_pipeline():
    for name in [
        "parse",
        "current_info",
        "workouts",
        "past_starts",
        "feature_eng",
        "fitness_metrics",
        "workout_analysis",
        "pace_projection",
        "class_assessment",
        "form_cycle",
        "integrated_analytics",
    ]:
        logger.info("--- Running stage: %s ---", name)
        STAGE_FUNCTIONS[name]()
    logger.info("=== Pipeline completed successfully! ===")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Horse Racing Data Pipeline")
    parser.add_argument(
        "stages",
        nargs="*",
        default=["all"],
        help=(
            "Stages to run. Default 'all'. Available stages: "
            + ", ".join(STAGE_FUNCTIONS.keys())
        ),
    )
    args = parser.parse_args(argv)

    if args.stages == ["all"]:
        run_complete_pipeline()
    else:
        for stage in args.stages:
            func = STAGE_FUNCTIONS.get(stage)
            if not func:
                parser.error(f"Unknown stage: {stage}")
            logger.info("--- Running stage: %s ---", stage)
            func()


if __name__ == "__main__":
    main()
