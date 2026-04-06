#!/usr/bin/env python
"""
Run the full SimpleTM experiment grid across all supported datasets,
including the cleaned smartbuilding dataset.

This script mirrors the logic of `dag-all-dataset.ipynb`, but as a reusable
Python entrypoint that can be run locally or on Kaggle.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


EXPERIMENTS = [
    {"tag": "SWT_original", "model": "SimpleTM_SWT", "attention_mode": "original"},
    {"tag": "SWT_dual", "model": "SimpleTM_SWT", "attention_mode": "dual"},
    {"tag": "FFT_original", "model": "SimpleTM_FFT", "attention_mode": "original"},
    {"tag": "FFT_dual", "model": "SimpleTM_FFT", "attention_mode": "dual"},
]

METRIC_ORDER = [
    "mse",
    "mae",
    "rmse",
    "mape",
    "mspe",
    "rse",
    "corr",
    "smape",
    "wape",
    "r2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all SimpleTM datasets including smartbuilding")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./dataset/SimpleTM_datasets",
        help="Root folder that contains benchmark datasets",
    )
    parser.add_argument(
        "--smart-csv",
        type=str,
        default="./smart.csv",
        help="Fallback smartbuilding CSV path if it is not yet inside the dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results_complete/all_datasets_with_smart_results",
        help="Where checkpoints, copied plots, logs, and CSV summaries will be written",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="Optional global override for train epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Optional global override for early stopping patience",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset filter, for example ETTh1,smartbuilding,Solar",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running training",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear previous outputs before running",
    )
    return parser.parse_args()


def resolve_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_dataset_configs(dataset_root: Path, smart_csv: Path) -> List[Dict[str, object]]:
    configs = [
        {
            "name": "ETTh1",
            "data": "ETTh1",
            "root": resolve_existing_path([dataset_root / "ETT-small"]),
            "path": "ETTh1.csv",
            "enc_in": 7,
            "d_model": 32,
            "d_ff": 32,
            "wv": "db1",
            "m": 3,
            "alpha": 1.0,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "train_epochs": 10,
            "patience": 3,
            "use_norm": 1,
            "lradj": "type1",
            "freq": "h",
        },
        {
            "name": "ETTh2",
            "data": "ETTh2",
            "root": resolve_existing_path([dataset_root / "ETT-small"]),
            "path": "ETTh2.csv",
            "enc_in": 7,
            "d_model": 32,
            "d_ff": 32,
            "wv": "db1",
            "m": 3,
            "alpha": 1.0,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "train_epochs": 10,
            "patience": 3,
            "use_norm": 1,
            "lradj": "type1",
            "freq": "h",
        },
        {
            "name": "ETTm1",
            "data": "ETTm1",
            "root": resolve_existing_path([dataset_root / "ETT-small"]),
            "path": "ETTm1.csv",
            "enc_in": 7,
            "d_model": 32,
            "d_ff": 32,
            "wv": "db1",
            "m": 3,
            "alpha": 1.0,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "train_epochs": 10,
            "patience": 3,
            "use_norm": 1,
            "lradj": "type1",
            "freq": "t",
        },
        {
            "name": "ETTm2",
            "data": "ETTm2",
            "root": resolve_existing_path([dataset_root / "ETT-small"]),
            "path": "ETTm2.csv",
            "enc_in": 7,
            "d_model": 32,
            "d_ff": 32,
            "wv": "db1",
            "m": 3,
            "alpha": 1.0,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "train_epochs": 10,
            "patience": 3,
            "use_norm": 1,
            "lradj": "type1",
            "freq": "t",
        },
        {
            "name": "weather",
            "data": "custom",
            "root": resolve_existing_path([dataset_root / "weather"]),
            "path": "weather.csv",
            "enc_in": 21,
            "d_model": 32,
            "d_ff": 32,
            "wv": "db1",
            "m": 3,
            "alpha": 1.0,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "train_epochs": 10,
            "patience": 3,
            "use_norm": 1,
            "lradj": "type1",
            "freq": "h",
        },
        {
            "name": "electricity",
            "data": "custom",
            "root": resolve_existing_path([dataset_root / "electricity"]),
            "path": "electricity.csv",
            "enc_in": 321,
            "d_model": 128,
            "d_ff": 128,
            "wv": "db1",
            "m": 3,
            "alpha": 1.0,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "train_epochs": 10,
            "patience": 3,
            "use_norm": 1,
            "lradj": "type1",
            "freq": "h",
        },
        {
            "name": "traffic",
            "data": "custom",
            "root": resolve_existing_path([dataset_root / "traffic"]),
            "path": "traffic.csv",
            "enc_in": 862,
            "d_model": 256,
            "d_ff": 256,
            "wv": "db1",
            "m": 3,
            "alpha": 1.0,
            "learning_rate": 1e-4,
            "batch_size": 8,
            "train_epochs": 10,
            "patience": 3,
            "use_norm": 1,
            "lradj": "type1",
            "freq": "h",
        },
        {
            "name": "Solar",
            "data": "Solar",
            "root": resolve_existing_path([dataset_root / "Solar"]),
            "path": "solar_AL.txt",
            "enc_in": 137,
            "d_model": 128,
            "d_ff": 256,
            "wv": "db8",
            "m": 3,
            "alpha": 0.0,
            "learning_rate": 0.003,
            "batch_size": 128,
            "train_epochs": 10,
            "patience": 3,
            "use_norm": 0,
            "lradj": "TST",
            "freq": "h",
        },
        {
            "name": "PEMS03",
            "data": "PEMS",
            "root": resolve_existing_path([dataset_root / "PEMS"]),
            "path": "PEMS03.npz",
            "enc_in": 358,
            "d_model": 256,
            "d_ff": 1024,
            "wv": "bior3.1",
            "m": 3,
            "alpha": 0.1,
            "learning_rate": 0.002,
            "batch_size": 16,
            "train_epochs": 20,
            "patience": 10,
            "use_norm": 0,
            "lradj": "TST",
            "freq": "h",
        },
        {
            "name": "PEMS04",
            "data": "PEMS",
            "root": resolve_existing_path([dataset_root / "PEMS"]),
            "path": "PEMS04.npz",
            "enc_in": 307,
            "d_model": 256,
            "d_ff": 1024,
            "wv": "bior3.1",
            "m": 3,
            "alpha": 0.1,
            "learning_rate": 0.002,
            "batch_size": 16,
            "train_epochs": 20,
            "patience": 10,
            "use_norm": 0,
            "lradj": "TST",
            "freq": "h",
        },
        {
            "name": "PEMS07",
            "data": "PEMS",
            "root": resolve_existing_path([dataset_root / "PEMS"]),
            "path": "PEMS07.npz",
            "enc_in": 883,
            "d_model": 256,
            "d_ff": 512,
            "wv": "db1",
            "m": 3,
            "alpha": 0.1,
            "learning_rate": 0.002,
            "batch_size": 8,
            "train_epochs": 20,
            "patience": 10,
            "use_norm": 0,
            "lradj": "TST",
            "freq": "h",
        },
        {
            "name": "PEMS08",
            "data": "PEMS",
            "root": resolve_existing_path([dataset_root / "PEMS"]),
            "path": "PEMS08.npz",
            "enc_in": 170,
            "d_model": 256,
            "d_ff": 1024,
            "wv": "db12",
            "m": 3,
            "alpha": 0.0,
            "learning_rate": 0.001,
            "batch_size": 16,
            "train_epochs": 20,
            "patience": 10,
            "use_norm": 0,
            "lradj": "TST",
            "freq": "h",
        },
        {
            "name": "smartbuilding",
            "data": "custom",
            "root": resolve_existing_path([dataset_root / "smartbuilding", smart_csv.parent]),
            "path": "smart.csv",
            "enc_in": 37,
            "d_model": 64,
            "d_ff": 128,
            "wv": "db4",
            "m": 3,
            "alpha": 1.0,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "train_epochs": 10,
            "patience": 3,
            "use_norm": 1,
            "lradj": "type1",
            "freq": "h",
            "target": "Floor_Total(kW)",
        },
    ]

    filtered = []
    for cfg in configs:
        if cfg["root"] is None:
            continue
        file_path = Path(cfg["root"]) / str(cfg["path"])
        if file_path.exists():
            filtered.append(cfg)
    return filtered


def maybe_filter_datasets(configs: List[Dict[str, object]], dataset_filter: str) -> List[Dict[str, object]]:
    if not dataset_filter:
        return configs
    wanted = {item.strip() for item in dataset_filter.split(",") if item.strip()}
    return [cfg for cfg in configs if cfg["name"] in wanted]


def prepare_output_dirs(output_dir: Path, keep_existing: bool) -> Dict[str, Path]:
    checkpoints_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    results_file = output_dir / "result_long_term_forecast.txt"

    if not keep_existing and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not keep_existing and results_file.exists():
        results_file.unlink()

    legacy_result_file = Path("result_long_term_forecast.txt")
    if not keep_existing and legacy_result_file.exists():
        legacy_result_file.unlink()

    return {
        "output_dir": output_dir,
        "checkpoints_dir": checkpoints_dir,
        "plots_dir": plots_dir,
        "results_file": results_file,
    }


def build_command(cfg: Dict[str, object], exp: Dict[str, str], args: argparse.Namespace, checkpoints_dir: Path) -> List[str]:
    train_epochs = args.train_epochs or int(cfg["train_epochs"])
    patience = args.patience or int(cfg["patience"])
    unique_model_id = f"{cfg['name']}_{exp['model']}_{exp['attention_mode']}"

    cmd = [
        "python",
        "-u",
        "run.py",
        "--is_training",
        "1",
        "--model",
        exp["model"],
        "--attention_mode",
        exp["attention_mode"],
        "--model_id",
        unique_model_id,
        "--data",
        str(cfg["data"]),
        "--root_path",
        str(cfg["root"]),
        "--data_path",
        str(cfg["path"]),
        "--features",
        "M",
        "--freq",
        str(cfg["freq"]),
        "--seq_len",
        "96",
        "--pred_len",
        "96",
        "--e_layers",
        "1",
        "--d_model",
        str(cfg["d_model"]),
        "--d_ff",
        str(cfg["d_ff"]),
        "--enc_in",
        str(cfg["enc_in"]),
        "--dec_in",
        str(cfg["enc_in"]),
        "--c_out",
        str(cfg["enc_in"]),
        "--wv",
        str(cfg["wv"]),
        "--m",
        str(cfg["m"]),
        "--alpha",
        str(cfg["alpha"]),
        "--learning_rate",
        str(cfg["learning_rate"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--train_epochs",
        str(train_epochs),
        "--patience",
        str(patience),
        "--num_workers",
        "2",
        "--checkpoints",
        str(checkpoints_dir),
        "--fix_seed",
        "2025",
        "--use_norm",
        str(cfg["use_norm"]),
        "--lradj",
        str(cfg["lradj"]),
    ]

    if "target" in cfg:
        cmd.extend(["--target", str(cfg["target"])])

    return cmd


def run_experiments(configs: List[Dict[str, object]], args: argparse.Namespace, checkpoints_dir: Path) -> None:
    for cfg in configs:
        print("\n" + "=" * 100)
        print(f"DATASET: {cfg['name']} | file={Path(cfg['root']) / cfg['path']}")
        print("=" * 100 + "\n")

        for exp in EXPERIMENTS:
            cmd = build_command(cfg, exp, args, checkpoints_dir)
            print(f">>> {cfg['name']} | {exp['tag']}")
            print(" ".join(cmd))

            if args.dry_run:
                print()
                continue

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(
                    f"Run failed for dataset={cfg['name']} variant={exp['tag']} with exit code {process.returncode}"
                )


def copy_outputs(checkpoints_dir: Path, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    legacy_result_file = Path("result_long_term_forecast.txt")

    if legacy_result_file.exists():
        shutil.copy(legacy_result_file, output_dir / "result_long_term_forecast.txt")

    for root_dir, _, files in os.walk(checkpoints_dir):
        for file_name in files:
            if not file_name.endswith(".pdf"):
                continue
            root_path = Path(root_dir)
            target_plot_dir = plots_dir / root_path.name
            target_plot_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(root_path / file_name, target_plot_dir / file_name)


def parse_metrics_file(results_file: Path) -> pd.DataFrame:
    if not results_file.exists():
        return pd.DataFrame()

    content = results_file.read_text(encoding="utf-8").strip()
    if not content:
        return pd.DataFrame()

    entries = content.split("\n\n")
    pattern = re.compile(
        r"^(?P<dataset>[^_]+)_(?P<model>SimpleTM_(?:SWT|FFT))_(?P<attention>original|dual)_"
    )
    metric_pattern = re.compile(r"([a-zA-Z0-9_]+):([^,]+)")

    rows: List[Dict[str, object]] = []
    for entry in entries:
        lines = [line.strip() for line in entry.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        setting = lines[0]
        metrics_line = lines[-1]
        match = pattern.search(setting)
        if not match:
            continue

        model_type = match.group("model")
        attention_mode = match.group("attention")
        tokenization = "SWT" if model_type.endswith("SWT") else "FFT"
        row = {
            "Dataset": match.group("dataset"),
            "Model": model_type,
            "AttentionMode": attention_mode,
            "Variant": f"{tokenization}_{attention_mode}",
            "Setting": setting,
        }
        for metric_name, value in metric_pattern.findall(metrics_line):
            try:
                row[metric_name.upper()] = float(value.strip())
            except ValueError:
                continue
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    sort_cols = ["Dataset", "Model", "AttentionMode"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def save_metric_tables(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        print("No parseable metrics found.")
        return

    raw_path = output_dir / "raw_metrics.csv"
    df.to_csv(raw_path, index=False)

    value_columns = [metric.upper() for metric in METRIC_ORDER if metric.upper() in df.columns]
    pivot = df.pivot(index="Dataset", columns="Variant", values=value_columns)
    variant_order = ["SWT_original", "SWT_dual", "FFT_original", "FFT_dual"]
    pivot = pivot.reindex(columns=variant_order, level=1)
    final_path = output_dir / "final_metrics.csv"
    pivot.to_csv(final_path)

    print("\nRaw metrics:")
    print(df.to_string(index=False))
    print(f"\nSaved raw metrics to {raw_path}")
    print(f"Saved final metrics table to {final_path}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    smart_csv = Path(args.smart_csv)
    output_dir = Path(args.output_dir)

    dirs = prepare_output_dirs(output_dir, args.keep_existing)
    configs = make_dataset_configs(dataset_root, smart_csv)
    configs = maybe_filter_datasets(configs, args.datasets)

    if not configs:
        raise FileNotFoundError(
            "No datasets were found. Check --dataset-root, --smart-csv, or --datasets."
        )

    run_plan = pd.DataFrame(
        [
            {
                "Dataset": cfg["name"],
                "DataType": cfg["data"],
                "Path": str(Path(cfg["root"]) / str(cfg["path"])),
                "enc_in": cfg["enc_in"],
                "d_model": cfg["d_model"],
                "d_ff": cfg["d_ff"],
                "wv": cfg["wv"],
                "m": cfg["m"],
                "alpha": cfg["alpha"],
                "learning_rate": cfg["learning_rate"],
                "batch_size": cfg["batch_size"],
                "train_epochs": args.train_epochs or cfg["train_epochs"],
                "patience": args.patience or cfg["patience"],
            }
            for cfg in configs
        ]
    )
    run_plan.to_csv(output_dir / "run_plan.csv", index=False)

    print(f"Datasets found: {len(configs)}")
    print(f"Variants per dataset: {len(EXPERIMENTS)}")
    print(f"Total planned runs: {len(configs) * len(EXPERIMENTS)}")

    run_experiments(configs, args, dirs["checkpoints_dir"])

    if args.dry_run:
        return

    copy_outputs(dirs["checkpoints_dir"], dirs["output_dir"])
    df = parse_metrics_file(dirs["results_file"])
    save_metric_tables(df, dirs["output_dir"])

    archive_base = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir,
    )
    print(f"\nPacked outputs to {archive_base}")


if __name__ == "__main__":
    main()
