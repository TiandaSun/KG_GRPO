#!/usr/bin/env python3
"""Analyze GRPO training log and produce a comprehensive diagnostic report."""

import ast
import re
import sys
import statistics
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PhaseInfo:
    name: str
    start_step: int
    end_step: int
    samples: int
    steps: list = field(default_factory=list)


def parse_log(log_path: str) -> dict:
    """Parse the GRPO training log file and extract all metrics."""
    text = Path(log_path).read_text()
    lines = text.strip().split("\n")

    # --- Header info ---
    header = {}
    for line in lines[:15]:
        if "Date:" in line:
            header["start_date"] = line.split("Date:")[-1].strip()
        if "Node:" in line:
            header["node"] = line.split("Node:")[-1].strip()
        if "GPUs detected:" in line:
            header["gpus"] = line.split(":")[-1].strip()
        if "NVIDIA" in line:
            header["gpu_type"] = line.strip()
        if "trainable params:" in line:
            header["params_info"] = line.strip()

    # --- Phase info from .err file (try) ---
    err_path = log_path.replace(".log", ".err")
    phases = []
    if Path(err_path).exists():
        err_text = Path(err_path).read_text()
        phase_pattern = re.compile(
            r"=== Starting (\S+) \(max_hops=(\d+), max_steps=(\d+)\) ==="
        )
        samples_pattern = re.compile(
            r"Loaded (\d+) QA pairs.*\(max_hops=(\d+)\)"
        )
        phase_matches = phase_pattern.findall(err_text)
        sample_matches = samples_pattern.findall(err_text)

        cumulative_steps = 0
        for i, (pname, max_hops, max_steps) in enumerate(phase_matches):
            n_samples = int(sample_matches[i][0]) if i < len(sample_matches) else 0
            start = cumulative_steps
            end = cumulative_steps + int(max_steps)
            phases.append(PhaseInfo(
                name=pname,
                start_step=start,
                end_step=end,
                samples=n_samples,
            ))
            cumulative_steps = end

    # --- Parse metric dicts ---
    all_steps = []
    for line in lines:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                d = ast.literal_eval(line)
                # Convert all values to float
                step = {k: float(v) for k, v in d.items()}
                all_steps.append(step)
            except (ValueError, SyntaxError):
                continue

    # --- Assign steps to phases ---
    if phases:
        step_idx = 0
        for phase in phases:
            phase_step_count = phase.end_step - phase.start_step
            # Each log line = 1 logging interval. Figure out steps per log from epoch progression
            # Simpler: just divide evenly based on known max_steps per phase
            pass

        # Better approach: use the log line index. Phase boundaries are at:
        # Phase1: 350 steps, Phase2: 350 steps, Phase3: 300 steps = 1000 total
        # Logging every 10 steps => 35 + 35 + 30 = 100 log lines
        steps_per_phase = []
        for phase in phases:
            n_log_entries = (phase.end_step - phase.start_step) // 10  # logging_steps=10
            steps_per_phase.append(n_log_entries)

        idx = 0
        for pi, phase in enumerate(phases):
            n = steps_per_phase[pi]
            phase.steps = all_steps[idx:idx + n]
            idx += n

    # --- Footer info ---
    footer = {}
    for line in lines[-30:]:
        if "Training complete:" in line:
            footer["end_date"] = line.split("Training complete:")[-1].strip()
        if "State:" in line:
            footer["state"] = line.split("State:")[-1].strip()
        if "Job Wall-clock time:" in line:
            footer["wall_time"] = line.split(":")[-1].strip().lstrip()
            # re-parse properly
            m = re.search(r"Job Wall-clock time:\s+(.+)", line)
            if m:
                footer["wall_time"] = m.group(1).strip()
        if "GPU Utilized" in line or "CPU Utilized" in line:
            pass
        if "Memory Utilized:" in line:
            m = re.search(r"Memory Utilized:\s+(.+)", line)
            if m:
                footer["memory_used"] = m.group(1).strip()

    return {
        "header": header,
        "phases": phases,
        "all_steps": all_steps,
        "footer": footer,
    }


def compute_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of values."""
    if not values:
        return {}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
        "first_5_mean": statistics.mean(values[:5]) if len(values) >= 5 else statistics.mean(values),
        "last_5_mean": statistics.mean(values[-5:]) if len(values) >= 5 else statistics.mean(values),
    }


def linear_trend(values: list[float]) -> float:
    """Compute simple linear regression slope (per step)."""
    if len(values) < 2:
        return 0.0
    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(values)
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def print_report(data: dict) -> None:
    """Print a comprehensive analysis report."""
    header = data["header"]
    phases = data["phases"]
    all_steps = data["all_steps"]
    footer = data["footer"]

    print("=" * 80)
    print("         GRPO TRAINING LOG ANALYSIS REPORT")
    print("=" * 80)

    # --- Job Summary ---
    print("\n## Job Summary")
    print(f"  Node:           {header.get('node', 'N/A')}")
    print(f"  GPU:            {header.get('gpu_type', 'N/A')}")
    print(f"  Start:          {header.get('start_date', 'N/A')}")
    print(f"  End:            {footer.get('end_date', 'N/A')}")
    print(f"  Wall time:      {footer.get('wall_time', 'N/A')}")
    print(f"  Status:         {footer.get('state', 'N/A')}")
    print(f"  Memory used:    {footer.get('memory_used', 'N/A')}")
    print(f"  Parameters:     {header.get('params_info', 'N/A')}")
    print(f"  Total log entries: {len(all_steps)}")

    # --- Overall Metrics ---
    print("\n" + "=" * 80)
    print("## Overall Training Trajectory (all 1000 steps)")
    print("=" * 80)

    key_metrics = [
        ("reward", "Total Reward (kg + format)"),
        ("rewards/kg_reward_func/mean", "KG Reward"),
        ("rewards/format_reward_func/mean", "Format Reward"),
        ("entropy", "Policy Entropy"),
        ("completions/mean_length", "Mean Completion Length"),
        ("completions/clipped_ratio", "Clipped Ratio (hit max_len)"),
        ("reward_std", "Reward Std (within group)"),
        ("frac_reward_zero_std", "Frac Zero-Std Batches"),
        ("grad_norm", "Gradient Norm"),
        ("loss", "Policy Loss"),
    ]

    print(f"\n  {'Metric':<38} {'First5':>8} {'Last5':>8} {'Mean':>8} {'Min':>8} {'Max':>8} {'Trend':>10}")
    print(f"  {'-'*36}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")

    for key, label in key_metrics:
        values = [s[key] for s in all_steps if key in s]
        if not values:
            continue
        st = compute_stats(values)
        trend = linear_trend(values)
        trend_dir = "+" if trend > 0 else ""
        print(f"  {label:<38} {st['first_5_mean']:>8.4f} {st['last_5_mean']:>8.4f} {st['mean']:>8.4f} {st['min']:>8.4f} {st['max']:>8.4f} {trend_dir}{trend:>9.6f}")

    # --- Per-Phase Analysis ---
    if phases:
        print("\n" + "=" * 80)
        print("## Per-Phase Breakdown")
        print("=" * 80)

        for phase in phases:
            steps = phase.steps
            if not steps:
                print(f"\n  ### {phase.name}: NO DATA")
                continue

            print(f"\n  ### {phase.name}")
            print(f"      Steps: {phase.start_step}–{phase.end_step} | Samples: {phase.samples} | Log entries: {len(steps)}")

            phase_metrics = [
                ("reward", "Total Reward"),
                ("rewards/kg_reward_func/mean", "KG Reward"),
                ("rewards/format_reward_func/mean", "Format Reward"),
                ("entropy", "Entropy"),
                ("completions/mean_length", "Mean Comp. Length"),
                ("completions/clipped_ratio", "Clipped Ratio"),
                ("frac_reward_zero_std", "Frac Zero-Std"),
                ("reward_std", "Reward Std"),
                ("grad_norm", "Grad Norm"),
            ]

            print(f"      {'Metric':<25} {'First5':>8} {'Last5':>8} {'Mean':>8} {'Trend':>10}")
            print(f"      {'-'*23}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")

            for key, label in phase_metrics:
                values = [s[key] for s in steps if key in s]
                if not values:
                    continue
                st = compute_stats(values)
                trend = linear_trend(values)
                trend_dir = "+" if trend > 0 else ""
                print(f"      {label:<25} {st['first_5_mean']:>8.4f} {st['last_5_mean']:>8.4f} {st['mean']:>8.4f} {trend_dir}{trend:>9.6f}")

    # --- Critical Diagnostic Checks ---
    print("\n" + "=" * 80)
    print("## Diagnostic Checks")
    print("=" * 80)

    # 1. EOS / completion length
    mean_lengths = [s["completions/mean_length"] for s in all_steps if "completions/mean_length" in s]
    clipped = [s["completions/clipped_ratio"] for s in all_steps if "completions/clipped_ratio" in s]
    terminated = [s.get("completions/mean_terminated_length", 0) for s in all_steps if "completions/mean_length" in s]
    avg_clipped = statistics.mean(clipped)
    avg_terminated = statistics.mean(terminated)
    pct_fully_clipped = sum(1 for c in clipped if c >= 0.99) / len(clipped) * 100

    print(f"\n  1. EOS / Completion Length")
    print(f"     Mean completion length:          {statistics.mean(mean_lengths):.1f} / 512")
    print(f"     Average clipped ratio:           {avg_clipped:.4f}")
    print(f"     % of steps fully clipped (>=99%): {pct_fully_clipped:.1f}%")
    print(f"     Avg terminated length:           {avg_terminated:.1f}")

    if avg_clipped > 0.95:
        print(f"     ⚠️  PROBLEM: {avg_clipped*100:.1f}% of completions hit max_length=512.")
        print(f"        The model is NOT learning to terminate. The EOS fix may not be working,")
        print(f"        or max_completion_length=512 is too short for the reasoning required.")
    else:
        print(f"     ✅ Model is learning to terminate before max_length.")

    # 2. Loss near zero
    losses = [s["loss"] for s in all_steps if "loss" in s]
    avg_loss = statistics.mean([abs(l) for l in losses])
    print(f"\n  2. Policy Loss")
    print(f"     Mean |loss|:                     {avg_loss:.2e}")
    if avg_loss < 1e-6:
        print(f"     ⚠️  PROBLEM: Loss is essentially zero (~{avg_loss:.1e}).")
        print(f"        This means the policy is barely updating. Possible causes:")
        print(f"        - All completions in each group get near-identical rewards (no gradient signal)")
        print(f"        - Clip ratio is killing all gradients")
        print(f"        - Learning rate too low")
    else:
        print(f"     ✅ Loss magnitude looks reasonable.")

    # 3. Entropy
    entropies = [s["entropy"] for s in all_steps if "entropy" in s]
    ent_start = statistics.mean(entropies[:5])
    ent_end = statistics.mean(entropies[-5:])
    ent_drop_pct = (ent_start - ent_end) / ent_start * 100

    print(f"\n  3. Entropy")
    print(f"     Start (first 5):                 {ent_start:.4f}")
    print(f"     End (last 5):                    {ent_end:.4f}")
    print(f"     Drop:                            {ent_drop_pct:.1f}%")
    if ent_end < 0.05:
        print(f"     ⚠️  CRITICAL: Entropy collapsed to near-zero. Model has degenerated.")
    elif ent_drop_pct > 70:
        print(f"     ⚠️  WARNING: Entropy dropped by {ent_drop_pct:.0f}%. Risk of mode collapse.")
    elif ent_end > 0.1:
        print(f"     ✅ Entropy is decreasing but still healthy (>{ent_end:.3f}).")
    else:
        print(f"     ⚠️  Entropy is getting low ({ent_end:.4f}). Monitor for collapse.")

    # 4. Reward trend
    rewards = [s["reward"] for s in all_steps if "reward" in s]
    reward_trend = linear_trend(rewards)
    reward_start = statistics.mean(rewards[:5])
    reward_end = statistics.mean(rewards[-5:])
    reward_improvement = reward_end - reward_start

    print(f"\n  4. Reward Trajectory")
    print(f"     Start (first 5):                 {reward_start:.4f}")
    print(f"     End (last 5):                    {reward_end:.4f}")
    print(f"     Improvement:                     {'+' if reward_improvement > 0 else ''}{reward_improvement:.4f}")
    print(f"     Linear trend per log step:       {'+' if reward_trend > 0 else ''}{reward_trend:.6f}")
    if reward_improvement > 0.1:
        print(f"     ✅ GOOD: Meaningful reward improvement (+{reward_improvement:.3f}).")
    elif reward_improvement > 0:
        print(f"     ⚠️  Reward improved slightly (+{reward_improvement:.3f}), but may not be significant.")
    else:
        print(f"     ❌ Reward did not improve. Training may not be producing useful signal.")

    # 5. KG reward specifically
    kg_rewards = [s["rewards/kg_reward_func/mean"] for s in all_steps if "rewards/kg_reward_func/mean" in s]
    kg_start = statistics.mean(kg_rewards[:5])
    kg_end = statistics.mean(kg_rewards[-5:])
    kg_improvement = kg_end - kg_start

    print(f"\n  5. KG Reward (Path Alignment)")
    print(f"     Start (first 5):                 {kg_start:.4f}")
    print(f"     End (last 5):                    {kg_end:.4f}")
    print(f"     Improvement:                     {'+' if kg_improvement > 0 else ''}{kg_improvement:.4f}")
    if kg_improvement > 0.1:
        print(f"     ✅ GOOD: Model is learning KG path alignment.")
    elif kg_improvement > 0:
        print(f"     ⚠️  Slight KG reward improvement ({kg_improvement:+.3f}), marginal.")
    else:
        print(f"     ❌ KG reward not improving.")

    # 6. Format reward
    fmt_rewards = [s["rewards/format_reward_func/mean"] for s in all_steps if "rewards/format_reward_func/mean" in s]
    fmt_mean = statistics.mean(fmt_rewards)
    fmt_min = min(fmt_rewards)

    print(f"\n  6. Format Reward (<think> tags)")
    print(f"     Mean:                            {fmt_mean:.4f}")
    print(f"     Min:                             {fmt_min:.4f}")
    if fmt_mean > 0.95:
        print(f"     ✅ Model consistently produces <think> format.")
    else:
        print(f"     ⚠️  Format reward dipped — model occasionally loses <think> tags.")

    # 7. Frac zero-std (wasted batches)
    zero_std = [s["frac_reward_zero_std"] for s in all_steps if "frac_reward_zero_std" in s]
    avg_zero_std = statistics.mean(zero_std)
    max_zero_std = max(zero_std)

    print(f"\n  7. Zero-Variance Batches (wasted compute)")
    print(f"     Mean frac_zero_std:              {avg_zero_std:.4f}")
    print(f"     Max frac_zero_std:               {max_zero_std:.4f}")
    if avg_zero_std > 0.3:
        print(f"     ⚠️  WARNING: {avg_zero_std*100:.0f}% of prompts give identical rewards across generations.")
        print(f"        These produce zero gradient. Consider dynamic prompt filtering.")
    else:
        print(f"     ✅ Most batches have reward variance (productive gradient signal).")

    # 8. Clip ratio
    clip_high = [s.get("clip_ratio/high_mean", 0) for s in all_steps]
    clip_low = [s.get("clip_ratio/low_mean", 0) for s in all_steps]
    avg_clip_high = statistics.mean(clip_high)
    avg_clip_low = statistics.mean(clip_low)

    print(f"\n  8. PPO Clip Ratio")
    print(f"     Mean high clip:                  {avg_clip_high:.6f}")
    print(f"     Mean low clip:                   {avg_clip_low:.6f}")
    if avg_clip_high < 0.01 and avg_clip_low < 0.01:
        print(f"     ⚠️  NOTE: Clip ratios are all zero. Policy ratio stays within clip range.")
        print(f"        Combined with ~zero loss, this suggests very small policy updates.")

    # --- Reward distribution analysis ---
    print(f"\n  9. Reward Distribution Analysis")
    # Estimate accuracy from KG reward. R_bin = +0.1 if correct, -1.0 if wrong.
    # reward = r_path + r_bin. If r_bin = -1.0 (wrong), kg_reward is strongly negative.
    # If r_bin = +0.1 (correct), kg_reward is around +0.1 to +1.5
    # Rough split: kg_reward > 0 => likely correct answer
    kg_positive = sum(1 for r in kg_rewards if r > 0)
    kg_negative = sum(1 for r in kg_rewards if r <= 0)
    print(f"     Steps with kg_reward > 0:        {kg_positive}/{len(kg_rewards)} ({kg_positive/len(kg_rewards)*100:.1f}%)")
    print(f"     Steps with kg_reward <= 0:       {kg_negative}/{len(kg_rewards)} ({kg_negative/len(kg_rewards)*100:.1f}%)")
    print(f"     (kg_reward > 0 ≈ more correct answers than wrong in the batch)")

    # --- Timing ---
    step_times = [s.get("step_time", 0) for s in all_steps if "step_time" in s]
    if step_times:
        print(f"\n  10. Timing")
        total_step_time = sum(step_times)
        print(f"     Avg step time:                   {statistics.mean(step_times):.1f}s")
        print(f"     Total training time:             {total_step_time/3600:.1f}h")
        print(f"     Steps/hour:                      {len(step_times) * 10 / (total_step_time / 3600):.0f}")

    # --- Overall Verdict ---
    print("\n" + "=" * 80)
    print("## OVERALL VERDICT")
    print("=" * 80)

    issues = []
    positives = []

    if avg_clipped > 0.95:
        issues.append("EOS not working — nearly all completions hit max_length (512 tokens)")
    if avg_loss < 1e-6:
        issues.append(f"Loss is ~zero ({avg_loss:.1e}) — policy is barely updating")
    if ent_end < 0.05:
        issues.append("Entropy collapsed — model degenerated")
    if reward_improvement <= 0:
        issues.append("Total reward did not improve across training")
    if avg_clip_high < 0.01 and avg_clip_low < 0.01 and avg_loss < 1e-6:
        issues.append("Zero clip + zero loss = GRPO is not producing meaningful policy updates")

    if reward_improvement > 0.1:
        positives.append(f"Total reward improved significantly: {reward_start:.3f} → {reward_end:.3f} (+{reward_improvement:.3f})")
    if kg_improvement > 0.1:
        positives.append(f"KG reward improved: {kg_start:.3f} → {kg_end:.3f} (+{kg_improvement:.3f})")
    if fmt_mean > 0.95:
        positives.append(f"Format reward stable at {fmt_mean:.3f} — <think> tags preserved")
    if ent_end > 0.1:
        positives.append(f"Entropy still healthy at {ent_end:.3f} — no mode collapse")
    if reward_improvement > 0 and reward_improvement <= 0.1:
        positives.append(f"Total reward improved slightly: {reward_start:.3f} → {reward_end:.3f} (+{reward_improvement:.3f})")

    if positives:
        print("\n  Positives:")
        for p in positives:
            print(f"    ✅ {p}")

    if issues:
        print("\n  Issues:")
        for i in issues:
            print(f"    ⚠️  {i}")

    if not issues:
        print("\n  🎉 Training looks healthy! All checks passed.")
    elif len(issues) <= 1 and reward_improvement > 0:
        print(f"\n  Training shows promise but has {'an issue' if len(issues) == 1 else 'issues'} to address.")
    else:
        print(f"\n  ⚠️  Training has significant issues ({len(issues)} problems detected).")

    print()


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "kg_grpo_train-30865081.log"
    data = parse_log(log_path)
    print_report(data)
