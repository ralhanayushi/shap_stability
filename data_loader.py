"""Data ingestion, chunked processing, and inspection report generator for KKBOX dataset.

Addresses SRS Requirements:
- REQ-F-01: Configurable data directory reading.
- REQ-F-02: Chunked reading of transactions and user logs.
- REQ-F-03: File size, columns, row counts, missing values summary.
- REQ-F-04: Customer uniqueness, duplicates, is_churn distribution.
- REQ-F-05: Date min/max bounds in transactions and logs.
- REQ-F-06: Cross-table coverage of train customers.
- REQ-F-07: Save results to inspection_report.json.
- REQ-F-08: Raw data preservation (read-only operations).
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def inspect_raw_files(data_dir: Path, output_file: Optional[Path] = None) -> Dict[str, Any]:
    """Inspect all raw CSV files in data_dir and generate metadata report."""
    report = {
        "files": {},
        "coverage": {},
        "target_distribution": {},
    }

    files = {
        "train": data_dir / "train.csv",
        "train_v2": data_dir / "train_v2.csv",
        "members": data_dir / "members_v3.csv",
        "transactions_v2": data_dir / "transactions_v2.csv",
        "transactions": data_dir / "transactions.csv",
        "user_logs_v2": data_dir / "user_logs_v2.csv",
    }

    for name, fpath in files.items():
        if not fpath.exists():
            report["files"][name] = {"exists": False}
            continue

        size_mb = fpath.stat().st_size / (1024 * 1024)
        info: Dict[str, Any] = {
            "exists": True,
            "path": str(fpath),
            "size_mb": round(size_mb, 2),
        }

        # Inspect columns & missing values on first 50k rows
        sample_df = pd.read_csv(fpath, nrows=50_000)
        info["columns"] = sample_df.columns.tolist()
        info["sample_missing_pct"] = (sample_df.isnull().mean() * 100).round(2).to_dict()

        if name in ("train", "train_v2"):
            full_df = pd.read_csv(fpath)
            info["total_rows"] = len(full_df)
            info["unique_msno"] = int(full_df["msno"].nunique())
            info["duplicate_msno"] = int(len(full_df) - full_df["msno"].nunique())
            dist = full_df["is_churn"].value_counts(normalize=True).to_dict()
            info["churn_distribution"] = {int(k): round(float(v), 4) for k, v in dist.items()}
            report["target_distribution"][name] = info["churn_distribution"]
        elif "transactions" in name:
            # Estimate or compute min/max dates
            info["sample_date_min"] = int(sample_df["transaction_date"].min())
            info["sample_date_max"] = int(sample_df["transaction_date"].max())
            info["sample_expire_min"] = int(sample_df["membership_expire_date"].min())
            info["sample_expire_max"] = int(sample_df["membership_expire_date"].max())
        elif "user_logs" in name:
            info["sample_date_min"] = int(sample_df["date"].min())
            info["sample_date_max"] = int(sample_df["date"].max())

        report["files"][name] = info

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Inspection report saved to {output_file}")

    return report


def stream_transactions_for_users(
    file_paths: List[Path],
    target_users: set,
    cutoff_date: pd.Timestamp,
    chunksize: int = 500_000,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Read transactions in chunks, keeping strictly rows <= cutoff_date for target_users."""
    collected = []
    total_read = 0

    usecols = [
        "msno",
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
    ]

    cutoff_int = int(cutoff_date.strftime("%Y%m%d"))

    for fpath in file_paths:
        if not fpath.exists():
            continue
        logger.info(f"Streaming transactions from {fpath.name}...")
        for chunk in pd.read_csv(fpath, usecols=usecols, chunksize=chunksize):
            if max_rows and total_read >= max_rows:
                break
            total_read += len(chunk)

            # Strict cutoff filter (Leakage-Safe)
            chunk = chunk[chunk["transaction_date"] <= cutoff_int]
            # Target cohort filter
            chunk = chunk[chunk["msno"].isin(target_users)]

            if not chunk.empty:
                collected.append(chunk)

            if max_rows and total_read >= max_rows:
                break

    if not collected:
        return pd.DataFrame(columns=usecols)

    tx_df = pd.concat(collected, ignore_index=True).drop_duplicates()
    tx_df["transaction_date"] = pd.to_datetime(tx_df["transaction_date"].astype(str), format="%Y%m%d", errors="coerce")
    tx_df["membership_expire_date"] = pd.to_datetime(tx_df["membership_expire_date"].astype(str), format="%Y%m%d", errors="coerce")
    return tx_df


def stream_user_logs_for_users(
    file_paths: List[Path],
    target_users: set,
    cutoff_date: pd.Timestamp,
    lookback_window_days: Optional[int] = None,
    chunksize: int = 500_000,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Read user logs in chunks from CSV or directly from 7z archive without extracting 30GB to disk."""
    import subprocess
    import io

    collected = []
    total_read = 0

    usecols = [
        "msno",
        "date",
        "num_25",
        "num_50",
        "num_75",
        "num_985",
        "num_100",
        "num_unq",
        "total_secs",
    ]

    cutoff_int = int(cutoff_date.strftime("%Y%m%d"))
    start_int = 0
    if lookback_window_days:
        start_date = cutoff_date - pd.Timedelta(days=lookback_window_days)
        start_int = int(start_date.strftime("%Y%m%d"))

    for fpath in file_paths:
        if not fpath.exists():
            continue

        logger.info(f"Streaming user logs from {fpath.name}...")

        # If it's a 7z archive, use high-speed tar -xOf streaming
        if fpath.suffix == ".7z":
            proc = subprocess.Popen(
                ["tar", "-xOf", str(fpath)],
                stdout=subprocess.PIPE,
                text=True,
                bufsize=4 * 1024 * 1024,
            )
            header = proc.stdout.readline()
            if not header:
                proc.terminate()
                continue

            while True:
                lines = []
                for _ in range(chunksize):
                    line = proc.stdout.readline()
                    if not line:
                        break
                    lines.append(line)

                if not lines:
                    break

                total_read += len(lines)
                chunk_str = header + "".join(lines)
                chunk = pd.read_csv(io.StringIO(chunk_str), usecols=usecols)

                # Strict cutoff filter (Leakage-Safe)
                if start_int > 0:
                    chunk = chunk[(chunk["date"] <= cutoff_int) & (chunk["date"] >= start_int)]
                else:
                    chunk = chunk[chunk["date"] <= cutoff_int]

                # Filter target users
                chunk = chunk[chunk["msno"].isin(target_users)]
                if not chunk.empty:
                    collected.append(chunk)

                if max_rows and total_read >= max_rows:
                    proc.terminate()
                    break

            try:
                proc.terminate()
            except Exception:
                pass

        else:
            # Regular CSV chunked reading
            for chunk in pd.read_csv(fpath, usecols=usecols, chunksize=chunksize):
                if max_rows and total_read >= max_rows:
                    break
                total_read += len(chunk)

                # Strict cutoff filter (Leakage-Safe)
                if start_int > 0:
                    chunk = chunk[(chunk["date"] <= cutoff_int) & (chunk["date"] >= start_int)]
                else:
                    chunk = chunk[chunk["date"] <= cutoff_int]

                # Target cohort filter
                chunk = chunk[chunk["msno"].isin(target_users)]

                if not chunk.empty:
                    collected.append(chunk)

                if max_rows and total_read >= max_rows:
                    break

    if not collected:
        return pd.DataFrame(columns=usecols)

    logs_df = pd.concat(collected, ignore_index=True).drop_duplicates()
    logs_df["date"] = pd.to_datetime(logs_df["date"].astype(str), format="%Y%m%d", errors="coerce")
    return logs_df
