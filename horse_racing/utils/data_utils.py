"""Data utility functions."""
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Optional

def check_parquet_columns(file_path: Path,
                         expected_columns: Optional[List[str]] = None) -> List[str]:
    """Check columns in a parquet file (replaces check_columns.py)."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read schema without loading data
    parquet_file = pq.ParquetFile(file_path)
    columns = parquet_file.schema.names

    print(f"\nColumns in {file_path.name}:")
    print("-" * 50)
    for col in columns:
        print(f"  {col}")
    print(f"\nTotal columns: {len(columns)}")

    if expected_columns:
        missing = set(expected_columns) - set(columns)
        if missing:
            print(f"\n⚠️  Missing expected columns: {missing}")
        else:
            print("\n✓ All expected columns found")

    return columns

def verify_id_variables(df: pd.DataFrame,
                       id_vars: List[str] = ['race', 'post_position',
                                            'horse_name']) -> bool:
    """Verify ID variables exist in dataframe."""
    missing = [var for var in id_vars if var not in df.columns]
    if missing:
        print(f"Missing ID variables: {missing}")
        return False
    return True

def quick_eda(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Quick exploratory data analysis."""
    print(f"\n=== EDA for {name} ===")
    print(f"Shape: {df.shape}")
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")