#!/usr/bin/env python
"""
EDA for the SimpleTM benchmark datasets plus the cleaned smartbuilding dataset.

The smartbuilding file is the cleaned hourly Floor-6 CU-BEMS data derived by
`cleancode.py`. The original CU-BEMS documentation is in `smart.pdf`.

This script:
1. discovers benchmark datasets under a dataset root
2. includes `smartbuilding/smart.csv` when present
3. saves a column reference and dataset summary
4. generates lightweight per-dataset EDA plots
5. adds smartbuilding-specific plots for load, temperature, lux, and zone totals
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None


plt.style.use("seaborn-v0_8-whitegrid")
if sns is not None:
    sns.set_palette("tab10")

DATE_CANDIDATES = {"date", "datetime", "timestamp", "time"}
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for all datasets including smartbuilding")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./dataset/SimpleTM_datasets",
        help="Root folder that contains ETT-small, weather, electricity, traffic, Solar, PEMS, smartbuilding",
    )
    parser.add_argument(
        "--smart-csv",
        type=str,
        default="./smart.csv",
        help="Fallback path for the cleaned smartbuilding CSV when it is not yet inside the dataset zip",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results_complete/eda_all_datasets_with_smart",
        help="Directory where summaries and plots will be saved",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Maximum number of points to use in single-series plots",
    )
    return parser.parse_args()


def resolve_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_dataset_specs(dataset_root: Path, smart_csv: Path) -> List[Dict[str, object]]:
    specs = [
        {
            "name": "ETTh1",
            "kind": "csv",
            "path": resolve_existing_path([dataset_root / "ETT-small" / "ETTh1.csv"]),
        },
        {
            "name": "ETTh2",
            "kind": "csv",
            "path": resolve_existing_path([dataset_root / "ETT-small" / "ETTh2.csv"]),
        },
        {
            "name": "ETTm1",
            "kind": "csv",
            "path": resolve_existing_path([dataset_root / "ETT-small" / "ETTm1.csv"]),
        },
        {
            "name": "ETTm2",
            "kind": "csv",
            "path": resolve_existing_path([dataset_root / "ETT-small" / "ETTm2.csv"]),
        },
        {
            "name": "weather",
            "kind": "csv",
            "path": resolve_existing_path([dataset_root / "weather" / "weather.csv"]),
        },
        {
            "name": "electricity",
            "kind": "csv",
            "path": resolve_existing_path([dataset_root / "electricity" / "electricity.csv"]),
        },
        {
            "name": "traffic",
            "kind": "csv",
            "path": resolve_existing_path([dataset_root / "traffic" / "traffic.csv"]),
        },
        {
            "name": "Solar",
            "kind": "solar_txt",
            "path": resolve_existing_path([dataset_root / "Solar" / "solar_AL.txt"]),
        },
        {
            "name": "PEMS03",
            "kind": "pems_npz",
            "path": resolve_existing_path([dataset_root / "PEMS" / "PEMS03.npz"]),
        },
        {
            "name": "PEMS04",
            "kind": "pems_npz",
            "path": resolve_existing_path([dataset_root / "PEMS" / "PEMS04.npz"]),
        },
        {
            "name": "PEMS07",
            "kind": "pems_npz",
            "path": resolve_existing_path([dataset_root / "PEMS" / "PEMS07.npz"]),
        },
        {
            "name": "PEMS08",
            "kind": "pems_npz",
            "path": resolve_existing_path([dataset_root / "PEMS" / "PEMS08.npz"]),
        },
        {
            "name": "smartbuilding",
            "kind": "csv",
            "path": resolve_existing_path(
                [dataset_root / "smartbuilding" / "smart.csv", smart_csv]
            ),
        },
    ]
    return [spec for spec in specs if spec["path"] is not None]


def find_date_column(columns: Iterable[str]) -> Optional[str]:
    for col in columns:
        if col.lower() in DATE_CANDIDATES:
            return col
    return None


def load_dataset(spec: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    path = Path(spec["path"])
    kind = spec["kind"]
    name = str(spec["name"])

    if kind == "csv":
        raw = pd.read_csv(path)
        date_col = find_date_column(raw.columns)
        if date_col is not None:
            raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
            raw = raw.dropna(subset=[date_col]).set_index(date_col)
        numeric = raw.select_dtypes(include=[np.number]).copy()
    elif kind == "solar_txt":
        raw = pd.read_csv(path, header=None)
        raw.columns = [f"feature_{i:03d}" for i in range(raw.shape[1])]
        numeric = raw
    elif kind == "pems_npz":
        data = np.load(path, allow_pickle=True)["data"][:, :, 0]
        numeric = pd.DataFrame(data, columns=[f"node_{i:03d}" for i in range(data.shape[1])])
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    metadata = {
        "name": name,
        "path": str(path),
        "kind": kind,
        "has_datetime_index": isinstance(numeric.index, pd.DatetimeIndex),
    }
    return numeric, metadata


def infer_frequency(index: pd.Index) -> str:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 4:
        return ""
    try:
        freq = pd.infer_freq(index[: min(len(index), 200)])
        return freq or ""
    except ValueError:
        return ""


def classify_column(name: str) -> str:
    lower = name.lower()
    if "total(kw)" in lower:
        return "zone_or_floor_total_kw"
    if "(kw)" in lower:
        return "power_kw"
    if "degc" in lower or "temp" in lower:
        return "temperature"
    if "rh%" in lower or "humidity" in lower:
        return "humidity"
    if "lux" in lower:
        return "light"
    return "other"


def spectral_entropy(signal: np.ndarray) -> float:
    signal = signal[np.isfinite(signal)]
    if signal.size < 4:
        return float("nan")
    centered = signal - np.mean(signal)
    power = np.abs(np.fft.rfft(centered)) ** 2
    power = power[1:]
    if power.size == 0 or np.sum(power) <= EPS:
        return 0.0
    probs = power / np.sum(power)
    entropy = -np.sum(probs * np.log(probs + EPS))
    return float(entropy / np.log(len(probs) + EPS))


def dominant_period(signal: np.ndarray) -> float:
    signal = signal[np.isfinite(signal)]
    if signal.size < 4:
        return float("nan")
    centered = signal - np.mean(signal)
    power = np.abs(np.fft.rfft(centered)) ** 2
    freq = np.fft.rfftfreq(len(centered), d=1.0)
    if len(power) <= 1:
        return float("nan")
    power[0] = 0.0
    best = int(np.argmax(power))
    if best <= 0 or freq[best] <= 0:
        return float("nan")
    return float(1.0 / freq[best])


def safe_series(df: pd.DataFrame) -> pd.Series:
    if "Floor_Total(kW)" in df.columns:
        return df["Floor_Total(kW)"]
    return df.iloc[:, 0]


def save_column_reference(
    dataset_name: str,
    df: pd.DataFrame,
    output_dir: Path,
    rows: List[Dict[str, object]],
) -> None:
    lines = [f"Dataset: {dataset_name}", f"Columns: {len(df.columns)}", ""]
    for idx, col in enumerate(df.columns, start=1):
        role = classify_column(col)
        dtype = str(df[col].dtype)
        lines.append(f"{idx:03d}. {col} | dtype={dtype} | role={role}")
        rows.append(
            {
                "Dataset": dataset_name,
                "Column": col,
                "DType": dtype,
                "Role": role,
            }
        )
    (output_dir / "columns.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_representative_series(df: pd.DataFrame, output_dir: Path, dataset_name: str, max_points: int) -> None:
    series = safe_series(df).iloc[:max_points]
    plt.figure(figsize=(13, 4))
    plt.plot(series.index if isinstance(series.index, pd.DatetimeIndex) else np.arange(len(series)), series.values, linewidth=1.0)
    plt.title(f"{dataset_name} - Representative Series ({series.name})")
    plt.ylabel(series.name)
    plt.tight_layout()
    plt.savefig(output_dir / "representative_series.png", dpi=180)
    plt.close()


def plot_corr_heatmap(df: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    subset = df.copy()
    if subset.shape[1] > 12:
        top_cols = subset.var().sort_values(ascending=False).head(12).index
        subset = subset[top_cols]
    corr = subset.corr()
    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
    else:
        plt.imshow(corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
        plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
    plt.title(f"{dataset_name} - Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=180)
    plt.close()


def plot_daily_profile(df: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        return
    series = safe_series(df)
    hourly = series.groupby(df.index.hour).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(hourly.index, hourly.values, marker="o")
    plt.title(f"{dataset_name} - Mean Hour-of-Day Profile")
    plt.xlabel("Hour")
    plt.ylabel(series.name)
    plt.tight_layout()
    plt.savefig(output_dir / "daily_profile.png", dpi=180)
    plt.close()


def plot_power_spectrum(df: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    series = safe_series(df).dropna().values
    if len(series) < 8:
        return
    centered = series - series.mean()
    fft_vals = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=1.0)
    upper = max(8, len(freqs) // 4)
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs[1:upper], fft_vals[1:upper] + EPS)
    plt.title(f"{dataset_name} - Power Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.tight_layout()
    plt.savefig(output_dir / "power_spectrum.png", dpi=180)
    plt.close()


def plot_smartbuilding_specific(df: pd.DataFrame, output_dir: Path) -> None:
    zone_total_cols = [
        col for col in df.columns
        if col.endswith("_Total(kW)") and col != "Floor_Total(kW)"
    ]

    summary_rows = []
    for label, matcher in [
        ("power_kw", lambda c: "(kW)" in c and "Total" not in c),
        ("zone_total_kw", lambda c: c.endswith("_Total(kW)") and c != "Floor_Total(kW)"),
        ("temperature", lambda c: "degC" in c or "Temp" in c),
        ("humidity", lambda c: "RH%" in c or "humidity" in c.lower()),
        ("lux", lambda c: "lux" in c.lower()),
    ]:
        cols = [col for col in df.columns if matcher(col)]
        summary_rows.append({"Group": label, "Count": len(cols), "Columns": ", ".join(cols)})

    pd.DataFrame(summary_rows).to_csv(output_dir / "smartbuilding_column_groups.csv", index=False)

    if zone_total_cols:
        zoom = df[zone_total_cols + ["Floor_Total(kW)"]].iloc[: min(len(df), 24 * 14)]
        plt.figure(figsize=(13, 5))
        for col in zone_total_cols:
            plt.plot(zoom.index, zoom[col], alpha=0.75, linewidth=1.0, label=col)
        plt.plot(zoom.index, zoom["Floor_Total(kW)"], color="black", linewidth=2.0, label="Floor_Total(kW)")
        plt.title("smartbuilding - Zone Totals and Floor Total")
        plt.ylabel("kW")
        plt.legend(loc="upper right", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(output_dir / "smartbuilding_zone_totals.png", dpi=180)
        plt.close()

    if {"Floor_Total(kW)", "Floor_Mean_Temp"}.issubset(df.columns):
        plt.figure(figsize=(7, 5))
        plt.scatter(df["Floor_Mean_Temp"], df["Floor_Total(kW)"], alpha=0.35, s=8)
        plt.title("smartbuilding - Floor Total vs Mean Temperature")
        plt.xlabel("Floor_Mean_Temp")
        plt.ylabel("Floor_Total(kW)")
        plt.tight_layout()
        plt.savefig(output_dir / "smartbuilding_temp_vs_energy.png", dpi=180)
        plt.close()

    if isinstance(df.index, pd.DatetimeIndex) and "Floor_Total(kW)" in df.columns:
        hourly_heatmap = (
            df.assign(hour=df.index.hour, weekday=df.index.day_name())
            .pivot_table(index="weekday", columns="hour", values="Floor_Total(kW)", aggfunc="mean")
            .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        )
        plt.figure(figsize=(12, 4))
        if sns is not None:
            sns.heatmap(hourly_heatmap, cmap="viridis")
        else:
            plt.imshow(hourly_heatmap.values, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.xticks(range(hourly_heatmap.shape[1]), hourly_heatmap.columns, fontsize=8)
            plt.yticks(range(hourly_heatmap.shape[0]), hourly_heatmap.index, fontsize=8)
        plt.title("smartbuilding - Mean Floor Total by Weekday and Hour")
        plt.tight_layout()
        plt.savefig(output_dir / "smartbuilding_weekday_hour_heatmap.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    smart_csv = Path(args.smart_csv)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_specs = build_dataset_specs(dataset_root, smart_csv)
    if not dataset_specs:
        raise FileNotFoundError(
            "No datasets were found. Check --dataset-root and --smart-csv."
        )

    summary_rows: List[Dict[str, object]] = []
    column_rows: List[Dict[str, object]] = []

    for spec in dataset_specs:
        df, metadata = load_dataset(spec)
        dataset_name = str(spec["name"])
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        save_column_reference(dataset_name, df, dataset_dir, column_rows)
        df.describe().transpose().to_csv(dataset_dir / "describe.csv")

        missing_total = int(df.isna().sum().sum())
        zero_ratio = float((df == 0).sum().sum() / max(df.size, 1))
        series = safe_series(df).dropna().values

        summary_rows.append(
            {
                "Dataset": dataset_name,
                "Kind": metadata["kind"],
                "Path": metadata["path"],
                "Rows": len(df),
                "NumericColumns": df.shape[1],
                "MissingValues": missing_total,
                "ZeroRatio": zero_ratio,
                "Start": df.index.min() if isinstance(df.index, pd.DatetimeIndex) else "",
                "End": df.index.max() if isinstance(df.index, pd.DatetimeIndex) else "",
                "InferredFreq": infer_frequency(df.index),
                "RepresentativeSeries": safe_series(df).name,
                "SpectralEntropy": spectral_entropy(series),
                "DominantPeriod": dominant_period(series),
            }
        )

        plot_representative_series(df, dataset_dir, dataset_name, args.max_points)
        plot_corr_heatmap(df, dataset_dir, dataset_name)
        plot_daily_profile(df, dataset_dir, dataset_name)
        plot_power_spectrum(df, dataset_dir, dataset_name)

        if dataset_name == "smartbuilding":
            plot_smartbuilding_specific(df, dataset_dir)

        print(f"Saved EDA outputs for {dataset_name} -> {dataset_dir}")

    summary_df = pd.DataFrame(summary_rows).sort_values("Dataset").reset_index(drop=True)
    columns_df = pd.DataFrame(column_rows).sort_values(["Dataset", "Column"]).reset_index(drop=True)

    summary_df.to_csv(output_root / "dataset_summary.csv", index=False)
    columns_df.to_csv(output_root / "column_reference.csv", index=False)

    print("\nDataset summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {output_root / 'dataset_summary.csv'}")
    print(f"Column reference saved to {output_root / 'column_reference.csv'}")


if __name__ == "__main__":
    main()
