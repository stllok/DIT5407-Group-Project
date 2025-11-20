"""
HKO Daily Data Reader

This module provides functionality to read and validate Hong Kong Observatory (HKO)
daily meteorological data from CSV files.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class HKODailyRecord:
    """
    Data structure representing a single daily HKO record.

    Attributes:
        date: The date of the observation
        value: The meteorological value (e.g., temperature)
    """

    date: date
    value: float

    def __repr__(self) -> str:
        return f"HKODailyRecord(date={self.date}, value={self.value:.1f})"


def read_hko_daily_csv(file_path: Path) -> List[HKODailyRecord]:
    """
    Read HKO daily meteorological data from a CSV file.

    This function reads the CSV file, validates the data, filters out invalid records,
    and returns a list of HKODailyRecord objects. Invalid records (marked with '***')
    are skipped and logged to the console.

    Parameters:
        file_path: Path to the CSV file (pathlib.Path object)

    Returns:
        List of HKODailyRecord objects containing valid data

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the CSV structure is invalid or cannot be parsed

    Example:
        >>> from pathlib import Path
        >>> data = read_hko_daily_csv(Path("daily_HKO_GMT_2025.csv"))
        >>> print(f"Loaded {len(data)} valid records")
        >>> print(data[0])
    """
    # Validate input
    if not isinstance(file_path, Path):
        raise TypeError(
            f"file_path must be a pathlib.Path object, got {type(file_path)}"
        )

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read CSV file
    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            na_values=["***"],  # Treat '***' as missing values
            keep_default_na=True,
        )
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Validate expected columns
    expected_columns = [
        "年/Year",
        "月/Month",
        "日/Day",
        "數值/Value",
        "數據完整性/data Completeness",
    ]
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(
            f"CSV file missing required columns. Expected: {expected_columns}, Got: {list(df.columns)}"
        )

    # Rename columns for easier access
    df.columns = ["year", "month", "day", "value", "completeness"]

    # Process data
    records = []
    invalid_count = 0

    for idx, row in df.iterrows():
        try:
            # Check if value is missing (NaN) or invalid
            if pd.isna(row["value"]):
                invalid_count += 1
                print(
                    f"Skipping invalid record at row {idx + 2}: "
                    f"{int(row['year'])}-{int(row['month']):02d}-{int(row['day']):02d} - "
                    f"Value: unavailable (***)"
                )
                continue

            # Validate date components
            year = int(row["year"])
            month = int(row["month"])
            day = int(row["day"])

            # Create date object (will raise ValueError if invalid date)
            record_date = date(year, month, day)

            # Validate value is numeric
            value = float(row["value"])

            # Create record
            record = HKODailyRecord(date=record_date, value=value)

            records.append(record)

        except (ValueError, TypeError) as e:
            invalid_count += 1
            print(
                f"Skipping invalid record at row {idx + 2}: "
                f"{row['year']}-{row['month']}-{row['day']} - "
                f"Error: {e}"
            )
            continue

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("Data Loading Summary")
    print(f"{'=' * 60}")
    print(f"Total records processed: {len(df)}")
    print(f"Valid records loaded: {len(records)}")
    print(f"Invalid records skipped: {invalid_count}")
    print(f"Success rate: {len(records) / len(df) * 100:.2f}%")
    print(f"{'=' * 60}\n")

    if records:
        # Calculate basic statistics using numpy
        values = np.array([r.value for r in records])
        print("Data Statistics:")
        print(f"  Min value: {np.min(values):.1f}")
        print(f"  Max value: {np.max(values):.1f}")
        print(f"  Mean value: {np.mean(values):.2f}")
        print(f"  Std deviation: {np.std(values):.2f}")
        print(
            f"  Date range: {min(r.date for r in records)} to {max(r.date for r in records)}"
        )
        print(f"{'=' * 60}\n")

    return records


if __name__ == "__main__":
    # Example usage
    import sys

    # Default file path
    default_path = Path("daily_HKO_GMT_2025.csv")

    # Use command line argument if provided
    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    try:
        # Read data
        print(f"Reading HKO data from: {file_path}\n")
        records = read_hko_daily_csv(file_path)

        # Display first few records
        if records:
            print("First 5 records:")
            for record in records[:5]:
                print(f"  {record}")

            print("\nLast 5 records:")
            for record in records[-5:]:
                print(f"  {record}")

            # Example: Filter by date range
            print(f"\n{'=' * 60}")
            print("Example: Records for January 2025")
            print(f"{'=' * 60}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
