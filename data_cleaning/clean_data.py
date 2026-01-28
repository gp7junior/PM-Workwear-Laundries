"""Data cleaning utility

Features:
- Remove outliers (fixed numeric range) from numeric columns.
- Outlier actions: drop rows, mark as NaN, or replace with median.
- Uniform timestamps: parse timestamp column, set as index, resample to fixed frequency and aggregate/interpolate.
- CLI for easy use.

Usage example (Task 1 compliance):
python clean_data.py --input raw/telemetry.csv --output cleaned/telemetry.csv \
    --timestamp-col timestamp --freq 1h \
    --min-val 0 --max-val 100 --outlier-action median --interpolate
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable

import numpy as np
import pandas as pd


def read_csv(path: str, timestamp_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def remove_outliers_fixed_range(
    df: pd.DataFrame, 
    cols: Iterable[str], 
    min_val: float = 0.0, 
    max_val: float = 100.0, 
    action: str = "mark"
) -> pd.DataFrame:
    """
    Removes or replaces values outside a fixed specific range [min_val, max_val].
    Used for physical constraints like temperature (0-100).
    """
    result = df.copy()
    for c in cols:
        mask = (result[c] < min_val) | (result[c] > max_val)
        
        if not mask.any():
            continue

        if action == "drop":
            result = result.loc[~mask]
        elif action == "median":
            # Calculate median from VALID data only
            valid_data = result.loc[~mask, c]
            if not valid_data.empty:
                med = valid_data.median()
                result.loc[mask, c] = med
            else:
                # If all data is outlier, just set to NaN to be safe
                result.loc[mask, c] = np.nan
        else:
            result.loc[mask, c] = np.nan
    return result

def uniform_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    freq: str = "1min",
    numeric_interp: str = "linear",
    agg_method: str = "mean",
) -> pd.DataFrame:
    df = df.copy()
    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp column '{timestamp_col}' not found")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()

    num_cols = numeric_columns(df)
    non_num = [c for c in df.columns if c not in num_cols]

    agg = {}
    if agg_method == "mean":
        for c in num_cols:
            agg[c] = "mean"
    elif agg_method == "median":
        for c in num_cols:
            agg[c] = "median"
    else:
        for c in num_cols:
            agg[c] = agg_method

    for c in non_num:
        agg[c] = "first"

    resampled = df.resample(freq).agg(agg)

    # interpolate numeric columns
    for c in num_cols:
        if c in resampled.columns:
            resampled[c] = resampled[c].interpolate(method=numeric_interp, limit_direction="both")

    # forward-fill non-numeric using non-deprecated APIs
    for c in non_num:
        if c in resampled.columns:
            resampled[c] = resampled[c].ffill().bfill()

    resampled = resampled.reset_index()
    return resampled


def clean(
    input_path: str,
    output_path: str,
    timestamp_col: str = "timestamp",
    freq: str | None = "1min",
    outlier_action: str = "median",
    min_val: float = 0.0,
    max_val: float = 100.0,
    interpolate: bool = True,
    agg_method: str = "mean",
):
    df = read_csv(input_path, timestamp_col=timestamp_col)

    num_cols = numeric_columns(df)

    # Always apply fixed range cleaning as requested
    df = remove_outliers_fixed_range(df, num_cols, min_val=min_val, max_val=max_val, action=outlier_action)

    # If timestamp column exists and user wants resampling, resample
    if freq and timestamp_col in df.columns:
        df = uniform_timestamps(
            df, 
            timestamp_col=timestamp_col, 
            freq=freq, 
            numeric_interp=("linear" if interpolate else "nearest"), 
            agg_method=agg_method
        )
    else:
        if freq and timestamp_col not in df.columns:
            print(f"[info] timestamp column '{timestamp_col}' not found in {input_path}; skipping resample")
        if interpolate:
            for c in num_cols:
                if c in df.columns:
                    df[c] = df[c].interpolate(method="linear", limit_direction="both")

    df.to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean time series CSV: remove outliers (fixed range) and uniform timestamps")
    p.add_argument("--input", required=False, help="Input CSV path (use together with --output for single file)")
    p.add_argument("--input-dir", required=False, help="Input directory to process all CSV files inside")
    p.add_argument("--output", required=False, help="Output CSV path for single input")
    p.add_argument("--output-dir", required=False, help="Output directory for processed files when using --input-dir")
    p.add_argument("--timestamp-col", default="timestamp", help="Name of timestamp column")
    p.add_argument("--freq", default="1min", help="Resample frequency. Use empty string to skip. Ex: '1min', '1h'.")
    
    # Updated outlier arguments - Removed IQR/Zscore specific args
    p.add_argument("--outlier-action", choices=["drop", "median"], default="median", help="Action: drop=remove rows, median=replace with median")
    
    # Specific parameters for Fixed Range
    p.add_argument("--min-val", type=float, default=0.0, help="Min value for fixed_range method")
    p.add_argument("--max-val", type=float, default=100.0, help="Max value for fixed_range method")
    
    p.add_argument("--no-interpolate", dest="interpolate", action="store_false", help="Disable interpolation")
    p.add_argument("--agg-method", default="mean", help="Aggregation method for numeric cols (mean, median)")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    freq = args.freq if args.freq not in (None, "", "None") else None
    
    # Determine files to process
    files = []
    if args.input_dir:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    elif args.input and os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.csv")))
    elif args.input:
        files = [args.input]

    if not files:
        raise SystemExit("No input files found. Provide --input (file) or --input-dir (directory).")

    # Output handling
    output_dir = args.output_dir if getattr(args, 'output_dir', None) else None

    for f in files:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, os.path.basename(f))
        else:
            # if single input and output provided, use it; otherwise prefix cleaned_
            if len(files) == 1 and args.output:
                out_path = args.output
            else:
                out_path = os.path.join(os.getcwd(), f"cleaned_{os.path.basename(f)}")

        print(f"Processing {f} -> {out_path}...")
        
        # Cleaned up call signature to match the simplified function
        clean(
            input_path=f,
            output_path=out_path,
            timestamp_col=args.timestamp_col,
            freq=freq,
            outlier_action=args.outlier_action,
            min_val=args.min_val,
            max_val=args.max_val,
            interpolate=args.interpolate,
            agg_method=args.agg_method,
        )


if __name__ == "__main__":
    main()