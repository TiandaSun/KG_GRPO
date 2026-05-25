"""Analyze GRPO speed benchmark results from verl file logger JSONL output.

Reads per-config JSONL files produced by run_benchmark_speed.job, discards
warmup steps, computes timing statistics, and outputs a comparison table.

Usage:
    python scripts/analyze_benchmark.py results/benchmark_<JOB_ID>/ --warmup=3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from statistics import mean, stdev

logger = logging.getLogger(__name__)

# Timing keys to extract and display (in order)
TIMING_KEYS = [
    "timing_s/step",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/old_log_prob",
    "timing_s/RefPolicy",
    "timing_s/reward",
]

PERF_KEYS = [
    "perf/time_per_step",
    "perf/throughput",
    "perf/total_num_tokens",
]

# Short display names for table columns
DISPLAY_NAMES: dict[str, str] = {
    "timing_s/step": "step (s)",
    "timing_s/gen": "gen (s)",
    "timing_s/update_actor": "actor (s)",
    "timing_s/old_log_prob": "logprob (s)",
    "timing_s/RefPolicy": "ref (s)",
    "timing_s/reward": "reward (s)",
    "perf/time_per_step": "total (s)",
    "perf/throughput": "tok/s/gpu",
}

# Configs in expected order
CONFIG_ORDER = [
    "baseline",
    "opt_gpu_mem",
    "opt_micro_batch",
    "opt_ref_batch",
    "opt_kg_workers",
    "all_combined",
]

CONFIG_LABELS: dict[str, str] = {
    "baseline": "baseline",
    "opt_gpu_mem": "gpu_mem=0.6",
    "opt_micro_batch": "micro_batch=4",
    "opt_ref_batch": "ref_batch=8",
    "opt_kg_workers": "kg_workers=4",
    "all_combined": "ALL COMBINED",
}


def parse_jsonl(path: Path) -> list[dict[str, float]]:
    """Parse a verl FileLogger JSONL file into list of metric dicts per step."""
    records: list[dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s:%d", path, line_num)
                continue

            step = obj.get("step")
            data = obj.get("data", {})
            if step is None or not data:
                continue

            # Extract numeric metrics
            metrics: dict[str, float] = {"step": float(step)}
            for key in TIMING_KEYS + PERF_KEYS:
                if key in data and isinstance(data[key], (int, float)):
                    metrics[key] = float(data[key])

            records.append(metrics)

    return records


def detect_errors(results_dir: Path, config_name: str) -> list[str]:
    """Check for OOM or crash indicators in stderr log."""
    errors: list[str] = []
    stderr_path = results_dir / f"{config_name}_stderr.log"
    if not stderr_path.exists():
        errors.append("stderr log missing")
        return errors

    content = stderr_path.read_text(errors="replace")
    if "CUDA out of memory" in content or "OutOfMemoryError" in content:
        errors.append("CUDA OOM")
    if "RuntimeError: NCCL" in content:
        errors.append("NCCL error")
    if "Killed" in content:
        errors.append("process killed")
    if "TimeoutExpired" in content or "timeout: sending signal" in content:
        errors.append("timeout")

    return errors


def load_wall_times(results_dir: Path) -> dict[str, tuple[int, int]]:
    """Load wall times CSV: config -> (exit_code, wall_time_s)."""
    wall_times: dict[str, tuple[int, int]] = {}
    csv_path = results_dir / "wall_times.csv"
    if not csv_path.exists():
        return wall_times
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row["config"]
            wall_times[config] = (int(row["exit_code"]), int(row["wall_time_s"]))
    return wall_times


def compute_stats(
    records: list[dict[str, float]], warmup: int
) -> dict[str, tuple[float, float]]:
    """Compute mean and stdev for each metric, discarding warmup steps.

    Returns dict: metric_key -> (mean, stdev).
    """
    measured = records[warmup:]
    if not measured:
        return {}

    stats: dict[str, tuple[float, float]] = {}
    for key in TIMING_KEYS + PERF_KEYS:
        values = [r[key] for r in measured if key in r]
        if len(values) >= 2:
            stats[key] = (mean(values), stdev(values))
        elif len(values) == 1:
            stats[key] = (values[0], 0.0)

    return stats


def format_stat(m: float, s: float, width: int = 12) -> str:
    """Format mean ± std into a fixed-width string."""
    if s > 0.05:
        text = f"{m:.1f} +/- {s:.1f}"
    else:
        text = f"{m:.1f}"
    return text.center(width)


def generate_report(
    all_stats: dict[str, dict[str, tuple[float, float]]],
    wall_times: dict[str, tuple[int, int]],
    all_errors: dict[str, list[str]],
    warmup: int,
    measure: int,
    results_dir: Path,
) -> str:
    """Generate the full benchmark report as a string."""
    lines: list[str] = []

    lines.append("=" * 100)
    lines.append("GRPO Speed Benchmark Results")
    lines.append(f"Results dir: {results_dir}")
    lines.append(f"Steps measured: {measure} (after {warmup} warmup)")
    lines.append("=" * 100)
    lines.append("")

    # --- Main comparison table ---
    # Columns: config, step, gen, actor, logprob, ref, throughput, status
    display_keys = [
        "timing_s/step",
        "timing_s/gen",
        "timing_s/update_actor",
        "timing_s/old_log_prob",
        "timing_s/RefPolicy",
        "perf/throughput",
    ]
    col_width = 14
    header_names = ["Config"] + [DISPLAY_NAMES.get(k, k) for k in display_keys] + ["Status"]
    header = f"{'Config':<18}" + "".join(
        DISPLAY_NAMES.get(k, k).center(col_width) for k in display_keys
    ) + "  Status"
    lines.append(header)
    lines.append("-" * len(header))

    for config_name in CONFIG_ORDER:
        label = CONFIG_LABELS.get(config_name, config_name)
        stats = all_stats.get(config_name)
        errors = all_errors.get(config_name, [])
        wt = wall_times.get(config_name)

        if stats is None or not stats:
            status = "FAILED" if errors else "NO DATA"
            if errors:
                status += f" ({', '.join(errors)})"
            lines.append(f"{label:<18}" + " " * (col_width * len(display_keys)) + f"  {status}")
            continue

        row = f"{label:<18}"
        for key in display_keys:
            if key in stats:
                m, s = stats[key]
                row += format_stat(m, s, col_width)
            else:
                row += "-".center(col_width)

        status = "OK"
        if errors:
            status = ", ".join(errors)
        elif wt and wt[0] != 0:
            status = f"exit={wt[0]}"
        row += f"  {status}"
        lines.append(row)

    lines.append("")

    # --- Speedup table ---
    baseline_stats = all_stats.get("baseline")
    if baseline_stats and "timing_s/step" in baseline_stats:
        baseline_step = baseline_stats["timing_s/step"][0]
        lines.append("--- Speedup vs Baseline ---")
        for config_name in CONFIG_ORDER[1:]:
            label = CONFIG_LABELS.get(config_name, config_name)
            stats = all_stats.get(config_name)
            if stats and "timing_s/step" in stats:
                config_step = stats["timing_s/step"][0]
                if config_step > 0:
                    speedup = baseline_step / config_step
                    pct = (1 - config_step / baseline_step) * 100
                    lines.append(f"  {label:<18} {speedup:.2f}x ({pct:+.1f}%)")
                else:
                    lines.append(f"  {label:<18} N/A")
            else:
                lines.append(f"  {label:<18} FAILED / NO DATA")
        lines.append("")

    # --- Wall time summary ---
    if wall_times:
        lines.append("--- Wall Time (including model load + Ray init) ---")
        for config_name in CONFIG_ORDER:
            label = CONFIG_LABELS.get(config_name, config_name)
            wt = wall_times.get(config_name)
            if wt:
                exit_code, secs = wt
                lines.append(f"  {label:<18} {secs // 60}m {secs % 60}s (exit={exit_code})")
            else:
                lines.append(f"  {label:<18} N/A")
        lines.append("")

    # --- Errors ---
    has_errors = any(errs for errs in all_errors.values())
    if has_errors:
        lines.append("--- Errors ---")
        for config_name in CONFIG_ORDER:
            errors = all_errors.get(config_name, [])
            if errors:
                label = CONFIG_LABELS.get(config_name, config_name)
                lines.append(f"  {label}: {', '.join(errors)}")
        lines.append("")
    else:
        lines.append("--- Errors: none ---")
        lines.append("")

    # --- Per-config detailed breakdown ---
    lines.append("--- Detailed Breakdown (all timing keys) ---")
    for config_name in CONFIG_ORDER:
        label = CONFIG_LABELS.get(config_name, config_name)
        stats = all_stats.get(config_name)
        if not stats:
            lines.append(f"\n  [{label}] NO DATA")
            continue
        lines.append(f"\n  [{label}]")
        for key in TIMING_KEYS + PERF_KEYS:
            if key in stats:
                m, s = stats[key]
                lines.append(f"    {key:<30} {m:>8.2f} +/- {s:>6.2f}")

    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Analyze GRPO speed benchmark results")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing per-config JSONL files",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup steps to discard (default: 3)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    warmup: int = args.warmup

    if not results_dir.is_dir():
        logger.error("Results directory not found: %s", results_dir)
        sys.exit(1)

    # Load wall times
    wall_times = load_wall_times(results_dir)

    # Parse each config's JSONL and compute stats
    all_stats: dict[str, dict[str, tuple[float, float]]] = {}
    all_errors: dict[str, list[str]] = {}
    total_measured = 0

    for config_name in CONFIG_ORDER:
        jsonl_path = results_dir / f"{config_name}.jsonl"
        errors = detect_errors(results_dir, config_name)

        if not jsonl_path.exists():
            logger.warning("JSONL not found for %s: %s", config_name, jsonl_path)
            all_errors[config_name] = errors or ["JSONL missing"]
            continue

        records = parse_jsonl(jsonl_path)
        if len(records) <= warmup:
            logger.warning(
                "%s: only %d steps recorded (need >%d for measurement)",
                config_name, len(records), warmup,
            )
            errors.append(f"only {len(records)} steps")

        stats = compute_stats(records, warmup)
        all_stats[config_name] = stats
        all_errors[config_name] = errors

        measured = max(0, len(records) - warmup)
        total_measured += measured
        logger.info(
            "%s: %d total steps, %d measured, step_time=%.1f +/- %.1f",
            config_name,
            len(records),
            measured,
            stats.get("timing_s/step", (0, 0))[0],
            stats.get("timing_s/step", (0, 0))[1],
        )

    if not all_stats:
        logger.error("No data found in %s", results_dir)
        sys.exit(1)

    # Compute measure count from first successful config
    first_stats = next(iter(all_stats.values()), {})
    measure_count = 0
    for config_name in CONFIG_ORDER:
        jsonl_path = results_dir / f"{config_name}.jsonl"
        if jsonl_path.exists():
            records = parse_jsonl(jsonl_path)
            measure_count = max(measure_count, len(records) - warmup)

    # Generate and output report
    report = generate_report(
        all_stats=all_stats,
        wall_times=wall_times,
        all_errors=all_errors,
        warmup=warmup,
        measure=measure_count,
        results_dir=results_dir,
    )

    # Print to stdout
    print(report)

    # Save to file
    summary_path = results_dir / "benchmark_summary.txt"
    summary_path.write_text(report, encoding="utf-8")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
