#!/usr/bin/env python3
"""Extract training metrics from wandb for Tasks 11, 12, and 13.

Task 11 (Goodhart): training reward vs eval EM divergence
Task 12 (KG verification): E3 step-reward components over training
Task 13 (Entropy/KL): entropy and KL for all experiments

Usage:
    python scripts/task11_13_metrics.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
RESULTS_DIR = PROJECT_DIR / "results"

WANDB_ENTITY = "tianda-sun-University of York"
WANDB_PROJECT = "kg-align-verl"

RUNS = {
    "E1_outcome": "grpo-cwq-7b-outcome-20260321",
    "E2_heuristic": "grpo-cwq-7b-heuristic-20260325",
    "E3_verifiable": "grpo-cwq-7b-verifiable-20260321",
}

EVAL_FILES = {
    "E1_outcome": RESULTS_DIR / "eval_e1_goodhart_curve.json",
    "E2_heuristic": RESULTS_DIR / "eval_e2_goodhart_curve.json",
    "E3_verifiable": RESULTS_DIR / "eval_e3_goodhart_curve.json",
}

# Steps to sample for Task 13 table
KEY_STEPS = [0, 250, 500, 750, 1000, 1250]


def load_eval_data() -> Dict[str, Dict[str, Any]]:
    """Load pre-computed eval results from JSON files."""
    eval_data: Dict[str, Dict[str, Any]] = {}
    for label, path in EVAL_FILES.items():
        if path.exists():
            with open(path) as f:
                eval_data[label] = json.load(f)
            logger.info("Loaded eval data for %s: %d checkpoints", label, len(eval_data[label]))
        else:
            logger.warning("Eval file not found: %s", path)
            eval_data[label] = {}
    return eval_data


def discover_and_extract_wandb() -> Dict[str, Any]:
    """Connect to wandb API, discover keys, and extract metrics."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed")
        sys.exit(1)

    api = wandb.Api()
    all_run_data: Dict[str, Any] = {}

    for label, run_name in RUNS.items():
        logger.info("=" * 60)
        logger.info("Processing run: %s (%s)", label, run_name)

        # Find the run
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"display_name": run_name},
        )
        run_list = list(runs)
        if not run_list:
            logger.warning("Run not found: %s", run_name)
            all_run_data[label] = {"keys": [], "history": []}
            continue

        run = run_list[0]
        logger.info("Found run: %s (id=%s, state=%s)", run.name, run.id, run.state)

        # Discover all available keys from summary
        summary_keys = sorted(run.summary.keys())
        logger.info("Summary keys (%d):", len(summary_keys))
        for k in summary_keys:
            val = run.summary.get(k)
            if isinstance(val, (int, float)):
                logger.info("  %s = %s", k, val)
            else:
                logger.info("  %s (type=%s)", k, type(val).__name__)

        # Pull history - get a small sample to discover history keys
        sample = run.history(samples=5, pandas=False)
        if sample:
            history_keys = sorted(sample[0].keys())
            logger.info("\nHistory keys (%d):", len(history_keys))
            for k in history_keys:
                logger.info("  %s", k)
        else:
            history_keys = []
            logger.warning("No history rows found")

        # Now pull full history
        # Filter to relevant keys to reduce data transfer
        relevant_patterns = ["reward", "score", "entropy", "kl", "step", "r_step", "r_outcome",
                             "r_valid", "r_on_path", "r_progress", "r_coherence", "critic",
                             "loss", "lr", "_step", "global_step", "val-", "ppo_kl"]
        relevant_keys = [
            k for k in history_keys
            if any(p in k.lower() for p in relevant_patterns)
        ]
        # Always include _step and training/global_step
        if "_step" not in relevant_keys:
            relevant_keys.append("_step")
        if "training/global_step" not in relevant_keys and "training/global_step" in history_keys:
            relevant_keys.append("training/global_step")

        logger.info("\nRelevant keys to extract: %s", relevant_keys)

        # Pull dense training history (without val-aux keys which are sparse)
        full_history = run.history(samples=5000, keys=relevant_keys, pandas=False)
        logger.info("Pulled %d training history rows", len(full_history))

        # Separately pull val-aux keys (sparse, only at test intervals)
        val_aux_keys = [k for k in summary_keys if k.startswith("val-")]
        val_history: List[Dict[str, Any]] = []
        if val_aux_keys:
            logger.info("\nVal-aux keys from summary: %s", val_aux_keys)
            val_fetch_keys = ["_step", "training/global_step"] + val_aux_keys
            val_history = run.history(samples=5000, keys=val_fetch_keys, pandas=False)
            logger.info("Pulled %d val-aux history rows", len(val_history))

            # Merge val-aux data into full_history by step
            val_by_step: Dict[int, Dict[str, Any]] = {}
            for vrow in val_history:
                vs = vrow.get("training/global_step", vrow.get("_step"))
                if vs is not None:
                    val_by_step[int(vs)] = {k: v for k, v in vrow.items() if v is not None}

            for row in full_history:
                rs = row.get("training/global_step", row.get("_step"))
                if rs is not None and int(rs) in val_by_step:
                    row.update(val_by_step[int(rs)])

            # Add val-aux keys to relevant_keys for downstream discovery
            for vk in val_aux_keys:
                if vk not in relevant_keys:
                    relevant_keys.append(vk)

        all_run_data[label] = {
            "keys": history_keys,
            "summary_keys": summary_keys,
            "relevant_keys": relevant_keys,
            "history": full_history,
        }

    return all_run_data


def find_best_key(keys: List[str], patterns: List[str]) -> Optional[str]:
    """Find the best matching key from a list of candidate patterns."""
    for p in patterns:
        for k in keys:
            if p == k:
                return k
    # Fallback: substring match
    for p in patterns:
        for k in keys:
            if p in k.lower():
                return k
    return None


def task11_goodhart(
    run_data: Dict[str, Any], eval_data: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Task 11: Goodhart analysis - training reward vs eval EM."""
    logger.info("\n" + "=" * 60)
    logger.info("TASK 11: GOODHART ANALYSIS")
    logger.info("=" * 60)

    result: Dict[str, Any] = {}

    for label in RUNS:
        data = run_data.get(label, {})
        history = data.get("history", [])
        keys = data.get("relevant_keys", [])

        # Find reward key
        reward_key = find_best_key(
            keys,
            [
                "critic/score/mean",
                "train-core/cwq/reward/mean",
                "reward/mean",
                "reward",
                "score",
            ],
        )
        logger.info("\n%s - reward key: %s", label, reward_key)

        # Extract step -> reward mapping
        # Use training/global_step if available, otherwise _step
        step_reward: Dict[int, float] = {}
        if reward_key and history:
            for row in history:
                step = row.get("training/global_step", row.get("_step"))
                val = row.get(reward_key)
                if step is not None and val is not None:
                    step_reward[int(step)] = float(val)

        # Merge with eval EM
        edata = eval_data.get(label, {})
        merged = []
        # Get all steps from both sources
        all_steps = sorted(
            set(step_reward.keys()) | set(int(s) for s in edata.keys())
        )

        for s in all_steps:
            entry: Dict[str, Any] = {"step": s}
            if s in step_reward:
                entry["train_reward"] = round(step_reward[s], 4)
            es = str(s)
            if es in edata:
                entry["eval_em"] = edata[es].get("em")
                entry["eval_contains_em"] = edata[es].get("contains_em")
                entry["eval_f1"] = edata[es].get("f1")
                entry["eval_avg_tool_calls"] = edata[es].get("avg_tool_calls")
            merged.append(entry)

        result[label] = {
            "reward_key": reward_key,
            "n_reward_points": len(step_reward),
            "n_eval_points": len(edata),
            "merged": merged,
        }

        # Print table
        print(f"\n{'='*70}")
        print(f"  {label}: Training Reward vs Eval EM")
        print(f"{'='*70}")
        print(f"  {'Step':>6}  {'TrainReward':>12}  {'EvalEM':>8}  {'ContainsEM':>10}  {'F1':>8}  {'ToolCalls':>10}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
        # Subsample for display: show eval steps + every 50th training step
        eval_step_set = set(int(s) for s in edata.keys())
        display_steps = set(range(0, 1300, 50)) | eval_step_set | {1, 1259}
        for entry in merged:
            step = entry["step"]
            if step not in display_steps:
                continue
            tr = f"{entry['train_reward']:.4f}" if "train_reward" in entry else "---"
            em = f"{entry['eval_em']:.3f}" if "eval_em" in entry else "---"
            cem = f"{entry['eval_contains_em']:.3f}" if "eval_contains_em" in entry else "---"
            f1 = f"{entry['eval_f1']:.3f}" if "eval_f1" in entry else "---"
            tc = f"{entry['eval_avg_tool_calls']:.2f}" if "eval_avg_tool_calls" in entry else "---"
            if "train_reward" in entry or "eval_em" in entry:
                print(f"  {step:>6}  {tr:>12}  {em:>8}  {cem:>10}  {f1:>8}  {tc:>10}")

        # Identify Goodhart divergence
        if step_reward and edata:
            eval_steps = sorted(int(s) for s in edata.keys())
            if len(eval_steps) >= 2:
                peak_em_step = max(eval_steps, key=lambda s: edata[str(s)].get("em", 0))
                peak_em = edata[str(peak_em_step)].get("em", 0)
                last_em = edata[str(eval_steps[-1])].get("em", 0)
                print(f"\n  Peak eval EM: {peak_em:.3f} at step {peak_em_step}")
                print(f"  Final eval EM: {last_em:.3f} at step {eval_steps[-1]}")
                if last_em < peak_em:
                    print(f"  ** GOODHART DIVERGENCE detected: EM dropped after step {peak_em_step}")
                else:
                    print(f"  No Goodhart divergence detected (EM monotonically increasing or flat)")

    return result


def task12_kg_verification(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """Task 12: KG verification reward components for E3."""
    logger.info("\n" + "=" * 60)
    logger.info("TASK 12: KG VERIFICATION REWARD (E3)")
    logger.info("=" * 60)

    label = "E3_verifiable"
    data = run_data.get(label, {})
    history = data.get("history", [])
    keys = data.get("relevant_keys", [])

    # Find step reward keys — check both history keys and val-aux keys
    all_keys = keys + data.get("summary_keys", [])
    component_patterns = {
        "r_step_avg": ["val-aux/cwq/r_step_avg/mean@1", "r_step_avg", "step_reward", "r_step"],
        "r_outcome": ["val-aux/cwq/r_outcome/mean@1", "r_outcome", "outcome_reward"],
        "r_valid": ["val-aux/cwq/r_valid/mean@1", "r_valid", "valid_reward"],
        "r_on_path": ["val-aux/cwq/r_on_path/mean@1", "r_on_path", "on_path"],
        "r_progress": ["val-aux/cwq/r_progress/mean@1", "r_progress", "progress"],
        "r_coherence": ["val-aux/cwq/r_coherence/mean@1", "r_coherence", "coherence"],
        "score": ["val-aux/cwq/score/mean@1", "val-core/cwq/reward/mean@1"],
        "em": ["val-aux/cwq/em/mean@1"],
        "f1": ["val-aux/cwq/f1/mean@1"],
    }

    found_keys: Dict[str, Optional[str]] = {}
    for comp, patterns in component_patterns.items():
        found_keys[comp] = find_best_key(all_keys, patterns)
        logger.info("  %s -> %s", comp, found_keys[comp])

    # Extract data — use training/global_step as the step index
    result_rows: List[Dict[str, Any]] = []
    if history:
        for row in history:
            step = row.get("training/global_step", row.get("_step"))
            if step is None:
                continue
            entry: Dict[str, Any] = {"step": int(step)}
            for comp, key in found_keys.items():
                if key and key in row and row[key] is not None:
                    entry[comp] = float(row[key])
            if len(entry) > 1:  # has at least one metric beyond step
                result_rows.append(entry)

    print(f"\n{'='*70}")
    print(f"  TASK 12: E3 KG Verification Reward Components")
    print(f"{'='*70}")

    if result_rows:
        active_comps = [c for c, k in found_keys.items() if k is not None]
        header = f"  {'Step':>6}"
        for c in active_comps:
            header += f"  {c:>12}"
        print(header)
        print(f"  {'-'*6}" + f"  {'-'*12}" * len(active_comps))

        for row in result_rows:
            step = row["step"]
            if step % 50 == 0 or step in KEY_STEPS:
                line = f"  {step:>6}"
                for c in active_comps:
                    val = row.get(c)
                    line += f"  {val:>12.4f}" if val is not None else f"  {'---':>12}"
                print(line)
    else:
        print("  No step-reward component data found in E3 history.")
        print(f"  Available keys: {keys}")

    result = {
        "found_keys": {k: v for k, v in found_keys.items()},
        "n_rows": len(result_rows),
        "data": result_rows,
    }
    return result


def task13_entropy_kl(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """Task 13: Entropy and KL for all experiments."""
    logger.info("\n" + "=" * 60)
    logger.info("TASK 13: ENTROPY & KL DIVERGENCE")
    logger.info("=" * 60)

    result: Dict[str, Any] = {}

    for label in RUNS:
        data = run_data.get(label, {})
        history = data.get("history", [])
        keys = data.get("relevant_keys", [])

        entropy_key = find_best_key(keys, ["actor/entropy", "entropy", "policy_entropy"])
        kl_key = find_best_key(keys, ["actor/ppo_kl", "actor/kl_divergence", "approx_kl"])

        logger.info("%s: entropy_key=%s, kl_key=%s", label, entropy_key, kl_key)

        # Extract step -> metrics (use training/global_step)
        step_data: Dict[int, Dict[str, float]] = {}
        if history:
            for row in history:
                step = row.get("training/global_step", row.get("_step"))
                if step is None:
                    continue
                s = int(step)
                entry: Dict[str, float] = {}
                if entropy_key and entropy_key in row and row[entropy_key] is not None:
                    entry["entropy"] = float(row[entropy_key])
                if kl_key and kl_key in row and row[kl_key] is not None:
                    entry["kl"] = float(row[kl_key])
                if entry:
                    step_data[s] = entry

        result[label] = {
            "entropy_key": entropy_key,
            "kl_key": kl_key,
            "n_points": len(step_data),
            "data": {str(s): v for s, v in sorted(step_data.items())},
        }

    # Print table
    print(f"\n{'='*70}")
    print(f"  TASK 13: Entropy & KL at Key Steps")
    print(f"{'='*70}")

    for label in RUNS:
        r = result[label]
        print(f"\n  {label} (entropy_key={r['entropy_key']}, kl_key={r['kl_key']})")
        print(f"  {'Step':>6}  {'Entropy':>12}  {'KL':>12}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}")

        data = r["data"]
        for step in KEY_STEPS:
            # Find closest step
            if str(step) in data:
                d = data[str(step)]
                actual_step = step
            else:
                available = sorted(int(s) for s in data.keys())
                if not available:
                    continue
                closest = min(available, key=lambda x: abs(x - step))
                if abs(closest - step) > 25:
                    continue
                d = data[str(closest)]
                actual_step = closest

            ent = f"{d['entropy']:.4f}" if "entropy" in d else "---"
            kl = f"{d['kl']:.6f}" if "kl" in d else "---"
            print(f"  {actual_step:>6}  {ent:>12}  {kl:>12}")

    return result


def main() -> None:
    """Main entry point."""
    logger.info("Loading eval data...")
    eval_data = load_eval_data()

    logger.info("\nConnecting to wandb API...")
    run_data = discover_and_extract_wandb()

    # Print all discovered keys summary
    print("\n" + "=" * 70)
    print("  ALL DISCOVERED METRIC KEYS PER RUN")
    print("=" * 70)
    for label in RUNS:
        data = run_data.get(label, {})
        keys = data.get("keys", [])
        print(f"\n  {label} ({len(keys)} keys):")
        for k in keys:
            print(f"    {k}")

    # Run analyses
    t11_result = task11_goodhart(run_data, eval_data)
    t12_result = task12_kg_verification(run_data)
    t13_result = task13_entropy_kl(run_data)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_t11 = RESULTS_DIR / "task11_goodhart_data.json"
    with open(out_t11, "w") as f:
        json.dump(t11_result, f, indent=2, default=str)
    logger.info("Saved Task 11 data to %s", out_t11)

    out_t12 = RESULTS_DIR / "task12_kg_reward_data.json"
    with open(out_t12, "w") as f:
        json.dump(t12_result, f, indent=2, default=str)
    logger.info("Saved Task 12 data to %s", out_t12)

    out_t13 = RESULTS_DIR / "task13_entropy_kl_data.json"
    with open(out_t13, "w") as f:
        json.dump(t13_result, f, indent=2, default=str)
    logger.info("Saved Task 13 data to %s", out_t13)

    print(f"\n{'='*70}")
    print(f"  Results saved to:")
    print(f"    {out_t11}")
    print(f"    {out_t12}")
    print(f"    {out_t13}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
