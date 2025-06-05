from .transform_workouts import main as _transform_workouts_main
from .transform_past_starts import main as _transform_past_starts_main


def transform_workouts():
    """Wrapper to generate long-format workouts file."""
    _transform_workouts_main()


def transform_past_starts():
    """Wrapper to generate long-format past starts file."""
    _transform_past_starts_main()

__all__ = ["transform_workouts", "transform_past_starts"]
