#!/usr/bin/env python3
"""Build Thermo Xcalibur MS queue CSVs from an LMD replicate manifest.

YAML-driven wrapper around :func:`xldvp_seg.lmd.ms_queue.build_thermo_queues`.

Usage:
    python scripts/build_ms_queue.py \
        --samples path/to/mk_replicates.csv \
        --config  configs/ms_queue/bm_mk.yaml \
        --output-dir path/to/ms_queues/ \
        [--combined] [--well-col well] [--plate-col plate]

YAML schema (all keys optional except file_name_template + autosampler_slots):

    file_name_template: "{date}_OA1_EdRo_SA_E990_{slide}_{bone}_rep{replicate}_{well_384}_{well_96}"
    autosampler_slots:
      B2: 2
      B3: 3
    ms_method: null                # or "C:\\Xcalibur\\methods\\..."
    path: "D:\\\\"
    date: null                     # defaults to today's YYYYMMDD
    empty_marker:
      column: slide
      value: EMPTY
    empty_file_name_template: null
    shuffle: true
    shuffle_seed: 42
    bracket_type: 4

Outputs:
    <output_dir>/<prefix>_<box_key>.csv   one per (plate, quadrant) group
    <output_dir>/<prefix>_key.csv         sample key (always)
    <output_dir>/<prefix>_key.json        sample key (JSON)
    <output_dir>/<prefix>_combined.csv    if --combined
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from xldvp_seg.exceptions import ConfigError
from xldvp_seg.lmd.ms_queue import ThermoQueueConfig, build_thermo_queues

_VALID_CONFIG_KEYS = {
    "file_name_template",
    "autosampler_slots",
    "ms_method",
    "path",
    "date",
    "empty_marker",
    "empty_file_name_template",
    "shuffle",
    "shuffle_seed",
    "bracket_type",
    "bracketing_blanks",
    "bracketing_blank_template",
    "interspersed_blanks",
    "interspersed_blank_template",
    "group_by_column",
    "group_by_ascending",
    "group_separator_blanks",
    "group_separator_blank_template",
    "column_substitutions",
}


def _load_config(config_path: Path) -> ThermoQueueConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    unknown = sorted(set(raw) - _VALID_CONFIG_KEYS)
    if unknown:
        raise ConfigError(
            f"Unknown keys in {config_path.name}: {unknown}. "
            f"Valid keys: {sorted(_VALID_CONFIG_KEYS)}"
        )

    if "file_name_template" not in raw:
        raise ConfigError(f"{config_path.name} missing required key 'file_name_template'")
    if "autosampler_slots" not in raw or not raw["autosampler_slots"]:
        raise ConfigError(f"{config_path.name} missing required key 'autosampler_slots'")

    empty_marker = None
    if raw.get("empty_marker"):
        em = raw["empty_marker"]
        if "column" not in em or "value" not in em:
            raise ConfigError("empty_marker must be a mapping with 'column' and 'value' keys")
        empty_marker = (em["column"], em["value"])

    column_subs = None
    if raw.get("column_substitutions"):
        column_subs = {}
        for col, sub in raw["column_substitutions"].items():
            if "pattern" not in sub or "replacement" not in sub:
                raise ConfigError(
                    f"column_substitutions[{col!r}] must have 'pattern' and 'replacement' keys"
                )
            column_subs[col] = (sub["pattern"], sub["replacement"])

    return ThermoQueueConfig(
        file_name_template=raw["file_name_template"],
        autosampler_slots=dict(raw["autosampler_slots"]),
        ms_method=raw.get("ms_method"),
        path=raw.get("path", "D:\\"),
        date=raw.get("date"),
        empty_marker=empty_marker,
        empty_file_name_template=raw.get("empty_file_name_template"),
        shuffle=raw.get("shuffle", True),
        shuffle_seed=raw.get("shuffle_seed", 42),
        bracket_type=raw.get("bracket_type", 4),
        bracketing_blanks=raw.get("bracketing_blanks", 0),
        bracketing_blank_template=raw.get("bracketing_blank_template"),
        interspersed_blanks=raw.get("interspersed_blanks", 0),
        interspersed_blank_template=raw.get("interspersed_blank_template"),
        group_by_column=raw.get("group_by_column"),
        group_by_ascending=raw.get("group_by_ascending", True),
        group_separator_blanks=raw.get("group_separator_blanks", False),
        group_separator_blank_template=raw.get("group_separator_blank_template"),
        column_substitutions=column_subs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="build_ms_queue",
        description="Build Thermo Xcalibur MS queue CSVs from LMD replicates.",
    )
    parser.add_argument("--samples", required=True, type=Path, help="Input replicates CSV")
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output CSVs")
    parser.add_argument("--well-col", default="well", help="Input column with 384-well addresses")
    parser.add_argument(
        "--plate-col",
        default="plate",
        help="Input column with plate number (use '' for single-plate)",
    )
    parser.add_argument("--out-prefix", default="ms_queue", help="Filename prefix for outputs")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write a combined CSV concatenating all boxes",
    )
    args = parser.parse_args()

    if not args.samples.exists():
        parser.error(f"samples file not found: {args.samples}")
    if not args.config.exists():
        parser.error(f"config file not found: {args.config}")

    plate_col = args.plate_col or None
    config = _load_config(args.config)
    outputs = build_thermo_queues(
        samples=args.samples,
        config=config,
        out_dir=args.output_dir,
        well_col=args.well_col,
        plate_col=plate_col,
        out_prefix=args.out_prefix,
        combined=args.combined,
    )

    print(f"\nWrote {len(outputs)} files to {args.output_dir}:")
    for key, path in outputs.items():
        print(f"  {key:20s} -> {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
