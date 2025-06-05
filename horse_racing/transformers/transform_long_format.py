from horse_racing.transformers.transform_workouts import main as transform_workouts
from horse_racing.transformers.transform_past_starts import main as transform_past_starts

__all__ = ["main"]

def main():
    """Run both workout and past performance transformations."""
    transform_workouts()
    transform_past_starts()

if __name__ == "__main__":
    main()

