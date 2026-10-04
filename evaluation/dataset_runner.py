"""
CLI Entry Point for Dataset Replay & Runtime Evaluation
CASS-EW SIH Problem Statement 26055 - Phase 5

Usage:
    python -m evaluation.dataset_runner --dataset path/to/dataset.csv [options]
    python -m evaluation.dataset_runner --dataset path/to/dataset.csv --config path/to/dataset_config.json --steps 100 --seed 42 --output results/run.json

Options:
    --dataset PATH          Path to dataset file (.csv, .json, .h5, .hdf5) [required]
    --config PATH           Path to JSON config defining field_mapping and units
    --steps INT             Number of scan steps to execute (default: 100)
    --dwell FLOAT           Dwell time in seconds (default: 0.005)
    --seed INT              Random seed for receiver noise and scheduler (default: 42)
    --bands INT             Number of discrete receiver bands (default: 10)
    --freq-min FLOAT        Minimum coverage frequency in MHz (optional)
    --freq-max FLOAT        Maximum coverage frequency in MHz (optional)
    --output PATH           Path to write JSON result report (optional)
"""

import argparse
import json
import sys
from pathlib import Path

from data.dataset_adapter import DatasetConfig
from data.dataset_runtime import DatasetRunConfig, DatasetRuntimeRunner, DatasetRunResult


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="CASS-EW User Dataset Runtime Bridge & Replay Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file (.csv, .json, .h5, .hdf5)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON file containing DatasetConfig (field_mapping, units)")
    parser.add_argument("--steps", type=int, default=100, help="Number of scan steps to execute")
    parser.add_argument("--dwell", type=float, default=0.005, help="Receiver dwell duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--bands", type=int, default=10, help="Number of discrete physical bands")
    parser.add_argument("--freq-min", type=float, default=None, help="Optional lower frequency coverage boundary in MHz")
    parser.add_argument("--freq-max", type=float, default=None, help="Optional upper frequency coverage boundary in MHz")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for result JSON")
    return parser.parse_args(args)


def main(cli_args=None):
    args = parse_args(cli_args)

    dataset_config = None
    if args.config is not None:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"[ERROR] Config file does not exist: {args.config}", file=sys.stderr)
            sys.exit(1)
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg_dict = json.load(f)

        cov = None
        if "coverage_freq_range_mhz" in cfg_dict and cfg_dict["coverage_freq_range_mhz"]:
            cov = tuple(cfg_dict["coverage_freq_range_mhz"])
        elif args.freq_min is not None and args.freq_max is not None:
            cov = (args.freq_min, args.freq_max)

        dataset_config = DatasetConfig(
            field_mapping=cfg_dict.get("field_mapping", {}),
            units=cfg_dict.get("units", {}),
            coverage_freq_range_mhz=cov
        )
    elif args.freq_min is not None and args.freq_max is not None:
        dataset_config = DatasetConfig(
            coverage_freq_range_mhz=(args.freq_min, args.freq_max)
        )

    cov_tuple = (args.freq_min, args.freq_max) if (args.freq_min is not None and args.freq_max is not None) else None

    run_config = DatasetRunConfig(
        dataset_path=args.dataset,
        dataset_config=dataset_config,
        seed=args.seed,
        steps=args.steps,
        dwell_time_s=args.dwell,
        num_bands=args.bands,
        output_path=args.output,
        coverage_freq_range_mhz=cov_tuple
    )

    runner = DatasetRuntimeRunner(run_config)
    result = runner.run()

    # Print user-friendly output
    print(result.to_text_summary())

    if args.output:
        print(f"[INFO] Detailed result saved to: {args.output}")

    if result.validation_status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
