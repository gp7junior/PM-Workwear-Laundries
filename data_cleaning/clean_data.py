"""Data cleaning utility

Features:
- Remove outliers (IQR or z-score) from numeric columns (drop or mark as NaN).
- Uniform timestamps: parse timestamp column, set as index, resample to fixed frequency and aggregate/interpolate.
- CLI for easy use.

Usage example:
python PM-Workwear-Laundries/data_cleaning/clean_data.py --input PM-Workwear-Laundries/data/raw/telemetry_data_full.csv --output cleaned.csv \
    --timestamp-col timestamp --freq 1min --outlier-method iqr --outlier-action mark --interpolate
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


def remove_outliers_iqr(df: pd.DataFrame, cols: Iterable[str], k: float = 1.5, action: str = "mark") -> pd.DataFrame:
    result = df.copy()
    for c in cols:
        q1 = result[c].quantile(0.25)
        q3 = result[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - k * iqr
        high = q3 + k * iqr
        mask = (result[c] < low) | (result[c] > high)
        if action == "drop":
            result = result.loc[~mask]
        elif action == "median":
            med = result[c].median()
            result.loc[mask, c] = med
        else:
            result.loc[mask, c] = np.nan
    return result


def remove_outliers_zscore(df: pd.DataFrame, cols: Iterable[str], threshold: float = 3.0, action: str = "mark") -> pd.DataFrame:
    result = df.copy()
    for c in cols:
        col = result[c]
        mean = col.mean()
        std = col.std()
        if std == 0 or pd.isna(std):
            continue
        z = (col - mean) / std
        mask = z.abs() > threshold
        if action == "drop":
            result = result.loc[~mask]
        elif action == "median":
            med = result[c].median()
            result.loc[mask, c] = med
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
    outlier_method: str = "iqr",
    outlier_action: str = "mark",
    iqr_k: float = 1.5,
    z_thresh: float = 3.0,
    interpolate: bool = True,
    agg_method: str = "mean",
):
    df = read_csv(input_path, timestamp_col=timestamp_col)

    num_cols = numeric_columns(df)

    if outlier_method == "iqr":
        df = remove_outliers_iqr(df, num_cols, k=iqr_k, action=outlier_action)
    elif outlier_method == "zscore":
        df = remove_outliers_zscore(df, num_cols, threshold=z_thresh, action=outlier_action)

    # If timestamp column exists and user wants resampling, resample. Otherwise skip resample
    if freq and timestamp_col in df.columns:
        df = uniform_timestamps(df, timestamp_col=timestamp_col, freq=freq, numeric_interp=("linear" if interpolate else "nearest"), agg_method=agg_method)
    else:
        if freq and timestamp_col not in df.columns:
            print(f"[info] timestamp column '{timestamp_col}' not found in {input_path}; skipping resample")
        if interpolate:
            for c in num_cols:
                if c in df.columns:
                    df[c] = df[c].interpolate(method="linear", limit_direction="both")

    df.to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean time series CSV: remove outliers and uniform timestamps")
    p.add_argument("--input", required=False, help="Input CSV path (use together with --output for single file)")
    p.add_argument("--input-dir", required=False, help="Input directory to process all CSV files inside")
    p.add_argument("--output", required=False, help="Output CSV path for single input")
    p.add_argument("--output-dir", required=False, help="Output directory for processed files when using --input-dir")
    p.add_argument("--timestamp-col", default="timestamp", help="Name of timestamp column")
    p.add_argument("--freq", default="1min", help="Resample frequency (pandas offset alias). Use empty string to skip resample. Examples: '1min', '1S', 'H'.")
    p.add_argument("--outlier-method", choices=["iqr", "zscore", "none"], default="iqr")
    p.add_argument("--outlier-action", choices=["mark", "drop", "median"], default="mark", help="Mark outliers as NaN, drop rows, or replace with median")
    p.add_argument("--iqr-k", type=float, default=1.5)
    p.add_argument("--z-thresh", type=float, default=3.0)
    p.add_argument("--no-interpolate", dest="interpolate", action="store_false")
    p.add_argument("--agg-method", default="mean", help="Aggregation for resample of numeric columns (mean, median, etc.)")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    freq = args.freq if args.freq not in (None, "", "None") else None
    outlier_method = args.outlier_method if args.outlier_method != "none" else ""

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

        clean(
            input_path=f,
            output_path=out_path,
            timestamp_col=args.timestamp_col,
            freq=freq,
            outlier_method=(args.outlier_method if args.outlier_method != "none" else ""),
            outlier_action=args.outlier_action,
            iqr_k=args.iqr_k,
            z_thresh=args.z_thresh,
            interpolate=args.interpolate,
            agg_method=args.agg_method,
        )


if __name__ == "__main__":
    main()
