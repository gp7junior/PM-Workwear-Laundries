# data_cleaning

Small helper script to remove outliers and uniform timestamps in CSV time-series data.

Dependencies
- pandas
- numpy

Example

```bash
python PM-Workwear-Laundries/data_cleaning/clean_data.py \
  --input PM-Workwear-Laundries/data/raw/telemetry_data_full.csv \
  --output cleaned.csv \
  --timestamp-col timestamp \
  --freq 1min \
  --outlier-method iqr \
  --outlier-action mark
```

Options
-- `--freq`: pandas resample frequency (e.g. `1min` = 1 minute, `1S` = 1 second, `H` = hourly). Leave empty to skip resampling.
- `--outlier-method`: `iqr`, `zscore`, or `none`.
- `--outlier-action`: `mark` (replace outliers with NaN), `drop` (remove rows containing outliers), or `median` (replace outliers with column median).

Batch processing

Process all CSVs inside a directory and write cleaned versions to an output directory:

```bash
python PM-Workwear-Laundries/data_cleaning/clean_data.py --input-dir PM-Workwear-Laundries/data/raw --output-dir PM-Workwear-Laundries/data/cleaned --timestamp-col timestamp --freq 1min --outlier-action median
```

After running, check `PM-Workwear-Laundries/data/cleaned` and adjust parameters as needed.
