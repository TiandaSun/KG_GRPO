"""S1.2 — Oracle L3 upper bound: gold-relation inference-time intervention.

Runs G2@500 inference on the full CWQ test set (3531 questions). At every
``<search>get_tail_entities(entity, relation)</search>`` (or any other 2-arg
KG action) the policy issues, the relation argument is REPLACED with the
gold relation (selected from ``extra_info.kg_path``) before the tool call
is dispatched to the KG server. The entity argument the policy chose is
kept untouched. This measures the upper-bound EM achievable if the model
only had to generate correct entities and the correct relations were
oracled in.

Oracle policy:
- Per-question gold relation set = unique relations in ``extra_info.kg_path``
  (matches V14-B2's reference set).
- For each tool call, pick the SINGLE gold relation that is most edit-close
  to whatever the policy emitted (V14-B2's normalised Levenshtein on the
  full and dotted-tail forms). Ties broken by ``difflib.SequenceMatcher``.
- If the question has no gold relations (should not happen on CWQ test, but
  fall back gracefully), or the policy emitted an unparseable tool call,
  the policy's emission is forwarded unchanged.

Outputs match ``scripts/eval_with_tools.py`` so downstream Phase 7 stratifiers
work without changes:

* ``results/phase7/oracle_l3_upper_bound_full_test.json``
* ``results/phase7/oracle_l3_upper_bound_per_sample/step_0_per_sample.json``
* ``results/trajectories/phase7/oracle_l3_full/step_0/trajectories.json``
  (each tool call is annotated with ``policy_relation`` and
  ``oracle_chosen_relation``).

Usage::

    # Start KG server first:
    python -m src_verl.kg_server.server --kg freebase \
        --freebase_dir data/freebase/kg --port 18901 &

    # Then run eval:
    python scripts/task_s12_oracle_l3.py \
        --merged_checkpoint outputs/verl-sft-cwq-39g2_step500-merged \
        --eval_data data/freebase/verl_cwq/test.parquet \
        --kg_server_url http://localhost:18901 \
        --output results/phase7/oracle_l3_upper_bound_full_test.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers (kept identical to scripts/eval_with_tools.py).
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(predicted: str, gold: str, aliases: list | None = None) -> float:
    pred_norm = normalize(predicted)
    if not pred_norm:
        return 0.0
    if pred_norm == normalize(gold):
        return 1.0
    if aliases:
        for a in aliases:
            if pred_norm == normalize(str(a)):
                return 1.0
    return 0.0


def contains_match(output: str, gold: str, aliases: list | None = None) -> float:
    out_norm = normalize(output)
    if normalize(gold) in out_norm:
        return 1.0
    if aliases:
        for a in aliases:
            if normalize(str(a)) in out_norm:
                return 1.0
    return 0.0


def token_f1(predicted: str, gold: str) -> float:
    p = normalize(predicted).split()
    g = normalize(gold).split()
    if not g:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)


def extract_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[1].split("</answer>")[0].strip()
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Edit-closeness oracle (V14-B2 style).
# ---------------------------------------------------------------------------


def _norm_relation(rel: Any) -> str:
    if rel is None:
        return ""
    s = str(rel).lower().strip()
    return s.replace("_", " ").replace(".", " ").strip()


def _rel_tail(rel: Any) -> str:
    if rel is None:
        return ""
    s = str(rel).strip()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s.lower().replace("_", " ").strip()


def _edit_distance(a: str, b: str) -> int:
    """Stdlib Levenshtein distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb]


def _closeness(queried: str, gold: str) -> tuple[int, float]:
    """Return (min_edit_dist, max_seq_ratio) on full + dotted-tail forms.

    Lower distance, higher ratio = closer.
    """
    q_full = _norm_relation(queried)
    g_full = _norm_relation(gold)
    q_tail = _rel_tail(queried)
    g_tail = _rel_tail(gold)
    if not q_full or not g_full:
        return max(len(q_full), len(g_full)), 0.0

    dists = [_edit_distance(q_full, g_full), _edit_distance(q_tail, g_tail)]
    ratios = [
        difflib.SequenceMatcher(None, q_full, g_full).ratio(),
        difflib.SequenceMatcher(None, q_tail, g_tail).ratio(),
    ]
    if q_tail and g_tail and (q_tail == g_tail or q_tail in g_full or g_tail in q_full):
        dists.append(1)
        ratios.append(0.9)
    return min(dists), max(ratios)


def pick_oracle_relation(policy_relation: str | None, gold_relations: list[str]) -> str | None:
    """Return the single gold relation most edit-close to the policy's emission.

    - If the policy didn't emit a relation, return the first gold relation
      (deterministic fallback so the tool call still has 2 args).
    - If gold_relations is empty, return None (caller forwards the policy
      emission unchanged).
    """
    if not gold_relations:
        return None
    if policy_relation is None or not str(policy_relation).strip():
        return gold_relations[0]

    best: tuple[int, float, int, str] | None = None  # (dist, -ratio, index, rel)
    for idx, gold in enumerate(gold_relations):
        dist, ratio = _closeness(policy_relation, gold)
        key = (dist, -ratio, idx, gold)
        if best is None or key < best:
            best = key
    assert best is not None
    return best[3]


def collect_gold_relations(kg_path: Any) -> list[str]:
    """Return ordered, deduplicated relations from the question's kg_path.

    Order is preserved (first occurrence wins) so the deterministic fallback
    in ``pick_oracle_relation`` is reproducible.
    """
    rels: list[str] = []
    seen: set[str] = set()
    if kg_path is None:
        return rels
    try:
        iterator = list(kg_path)
    except TypeError:
        return rels
    for triple in iterator:
        try:
            triple_list = list(triple)
        except TypeError:
            continue
        if len(triple_list) < 3:
            continue
        rel = str(triple_list[1]).strip()
        if not rel or rel in seen:
            continue
        seen.add(rel)
        rels.append(rel)
    return rels


# ---------------------------------------------------------------------------
# Tool-call parsing + KG dispatch (mirrors scripts/eval_with_tools.py with a
# relation-rewrite hook).
# ---------------------------------------------------------------------------


SEARCH_RE = re.compile(r"<search>\s*(\w+)\(([^)]*)\)\s*</search>", re.DOTALL)


def parse_search_call(text: str) -> tuple[str, str, str | None] | None:
    """Parse the FIRST ``<search>action(args)</search>`` call.

    Returns ``(action, entity, relation)`` or ``None`` if no call is found.
    """
    match = SEARCH_RE.search(text)
    if not match:
        return None
    action = match.group(1)
    args = [a.strip().strip("'\"") for a in match.group(2).split(",") if a.strip()]
    entity = args[0] if args else ""
    relation = args[1] if len(args) > 1 else None
    return action, entity, relation


def call_kg_server(action: str, entity: str, relation: str | None, server_url: str) -> str:
    payload: dict[str, Any] = {"action": action, "entity": entity}
    if relation:
        payload["relation"] = relation
    try:
        resp = requests.post(f"{server_url}/retrieve", json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", data.get("result", []))
            if isinstance(results, list) and len(results) > 10:
                results = results[:10]
            return json.dumps(results, ensure_ascii=False)
        return f"Error: {resp.status_code}"
    except Exception as e:  # noqa: BLE001 — surface server errors as text
        return f"Error: {e}"


def generate_with_oracle_tools(
    model: Any,
    tokenizer: Any,
    question: str,
    system_prompt: str,
    gold_relations: list[str],
    kg_server_url: str,
    max_turns: int = 5,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.0,
    device: str = "cuda",
) -> tuple[str, int, list[dict[str, Any]]]:
    """Multi-turn generation with relation-oracle KG tool calls.

    Returns ``(full_response, num_tool_calls, tool_call_records)``. Each record
    captures both the policy's original tool call and the oracle's rewrite so
    we can audit the intervention from the trajectory file.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    num_tool_calls = 0
    full_response = ""
    tool_records: list[dict[str, Any]] = []

    for turn in range(max_turns):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        full_response += response

        # Done — model emitted an answer.
        if "<answer>" in response and "</answer>" in response:
            messages.append({"role": "assistant", "content": response})
            break

        call = parse_search_call(response)
        if call is None:
            # Plain reasoning, no tool call; stop.
            messages.append({"role": "assistant", "content": response})
            break

        action, entity, policy_relation = call
        num_tool_calls += 1

        # ---- Oracle relation rewrite -------------------------------------
        # Only rewrite when the action takes a relation arg AND we have gold
        # relations to choose from. Single-arg actions (e.g. get_relations)
        # are passed through unchanged.
        rewritten_relation = policy_relation
        oracle_chosen: str | None = None
        if policy_relation is not None and gold_relations:
            oracle_chosen = pick_oracle_relation(policy_relation, gold_relations)
            if oracle_chosen is not None:
                rewritten_relation = oracle_chosen

        kg_result = call_kg_server(action, entity, rewritten_relation, kg_server_url)
        # ------------------------------------------------------------------

        tool_records.append({
            "turn": turn,
            "action": action,
            "entity": entity,
            "policy_relation": policy_relation,
            "oracle_chosen_relation": oracle_chosen,
            "relation_used": rewritten_relation,
            "rewritten": (oracle_chosen is not None and oracle_chosen != policy_relation),
        })

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"<tool_response>{kg_result}</tool_response>"})
        full_response += f"\n<tool_response>{kg_result}</tool_response>\n"

    return full_response, num_tool_calls, tool_records


# ---------------------------------------------------------------------------
# Main eval loop.
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="S1.2 oracle L3 upper-bound eval.")
    parser.add_argument(
        "--merged_checkpoint",
        type=Path,
        default=Path("outputs/verl-sft-cwq-39g2_step500-merged"),
        help="Path to the merged G2@500 HF checkpoint (used as both base_model and weights).",
    )
    parser.add_argument(
        "--eval_data",
        type=Path,
        default=Path("data/freebase/verl_cwq/test.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase7/oracle_l3_upper_bound_full_test.json"),
    )
    parser.add_argument(
        "--save_per_sample",
        type=Path,
        default=Path("results/phase7/oracle_l3_upper_bound_per_sample"),
    )
    parser.add_argument(
        "--save_trajectories",
        type=Path,
        default=Path("results/trajectories/phase7/oracle_l3_full"),
    )
    parser.add_argument("--kg_server_url", type=str, default="http://localhost:18901")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 = all samples (default: all 3531 CWQ test).")
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Persist per-sample + trajectories every N samples.")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--resume", action="store_true",
                        help="Skip sample_ids already in the per-sample output.")
    parser.add_argument(
        "--step_label", type=str, default="0",
        help="Step folder label inside trajectories/per-sample dirs (default '0' for merged).",
    )
    args = parser.parse_args()

    # ---- KG server health --------------------------------------------------
    try:
        r = requests.get(f"{args.kg_server_url}/health", timeout=5)
        logger.info("KG server healthy: %s", r.status_code)
    except Exception as e:  # noqa: BLE001
        logger.error("KG server not reachable at %s: %s", args.kg_server_url, e)
        return

    # ---- Load checkpoint + tokenizer --------------------------------------
    df = pd.read_parquet(args.eval_data)
    tokenizer = AutoTokenizer.from_pretrained(args.merged_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_prompt = df.iloc[0]["prompt"][0]["content"]

    logger.info("Loading merged checkpoint from %s ...", args.merged_checkpoint)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.merged_checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to(args.device)
        .eval()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.save_per_sample.mkdir(parents=True, exist_ok=True)
    traj_dir = args.save_trajectories / f"step_{args.step_label}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_file = traj_dir / "trajectories.json"

    end_idx = args.end_idx if args.end_idx > 0 else len(df)
    max_samples = args.max_samples if args.max_samples > 0 else len(df)

    if args.start_idx > 0 or args.end_idx > 0:
        ps_file = args.save_per_sample / f"step_{args.step_label}_chunk_{args.start_idx}_{end_idx}.json"
    else:
        ps_file = args.save_per_sample / f"step_{args.step_label}_per_sample.json"

    # ---- Resume ------------------------------------------------------------
    per_sample: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    ems: list[float] = []
    cms: list[float] = []
    f1s: list[float] = []
    tool_counts: list[int] = []
    rewrite_counts: list[int] = []
    already_done: set[str] = set()

    if args.resume and ps_file.exists():
        try:
            prior = json.load(open(ps_file))
            for rec in prior:
                already_done.add(str(rec["sample_id"]))
                per_sample.append(rec)
                ems.append(float(rec.get("em", 0)))
                cms.append(float(rec.get("contains_em", 0)))
                f1s.append(float(rec.get("f1", 0)))
                tool_counts.append(int(rec.get("num_tool_calls", 0)))
                rewrite_counts.append(int(rec.get("num_relation_rewrites", 0)))
            logger.info("Resumed: %d samples already processed", len(already_done))
        except Exception as e:  # noqa: BLE001
            logger.warning("Resume failed (%s) — starting fresh", e)

    if args.resume and traj_file.exists():
        try:
            trajectories = json.load(open(traj_file))
        except Exception:  # noqa: BLE001
            trajectories = []

    # ---- Main loop --------------------------------------------------------
    n_evaluated = len(per_sample)
    start = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        if i < args.start_idx:
            continue
        if i >= end_idx:
            break
        if n_evaluated >= max_samples:
            break

        sample_id = row["extra_info"].get("sample_id", str(i))
        if str(sample_id) in already_done:
            continue

        question = row["prompt"][1]["content"]
        gold = row["reward_model"]["ground_truth"]
        all_answers = row["extra_info"].get("all_answers", [gold])
        if hasattr(all_answers, "tolist"):
            all_answers = all_answers.tolist()
        hops = int(row["extra_info"].get("hops", 0))
        gold_relations = collect_gold_relations(row["extra_info"].get("kg_path"))
        n_evaluated += 1

        full_response, n_tools, tool_records = generate_with_oracle_tools(
            model,
            tokenizer,
            question,
            system_prompt,
            gold_relations,
            args.kg_server_url,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
        )

        predicted = extract_answer(full_response)
        em_score = exact_match(predicted, gold, all_answers)
        cm_score = contains_match(full_response, gold, all_answers)
        f1_score = token_f1(predicted, gold)
        n_rewrites = sum(1 for r in tool_records if r.get("rewritten"))

        ems.append(em_score)
        cms.append(cm_score)
        f1s.append(f1_score)
        tool_counts.append(n_tools)
        rewrite_counts.append(n_rewrites)

        per_sample.append({
            "sample_id": str(sample_id),
            "em": float(em_score),
            "contains_em": float(cm_score),
            "f1": float(f1_score),
            "num_tool_calls": int(n_tools),
            "num_relation_rewrites": int(n_rewrites),
            "num_gold_relations": len(gold_relations),
            "hops": hops,
        })

        trajectories.append({
            "sample_id": str(sample_id),
            "question": question,
            "gold_answer": gold,
            "all_answers": [str(a) for a in all_answers],
            "hops": hops,
            "gold_relations": gold_relations,
            "predicted": predicted,
            "full_response": full_response,
            "num_tool_calls": int(n_tools),
            "num_relation_rewrites": int(n_rewrites),
            "tool_calls": tool_records,
            "em": em_score,
            "f1": f1_score,
        })

        if n_evaluated % 50 == 0:
            logger.info(
                "  %d/%d EM=%.3f ContEM=%.3f F1=%.3f Tools=%.2f Rewrites=%.2f",
                n_evaluated, max_samples,
                sum(ems) / len(ems), sum(cms) / len(cms), sum(f1s) / len(f1s),
                sum(tool_counts) / len(tool_counts),
                sum(rewrite_counts) / len(rewrite_counts),
            )

        if n_evaluated % args.save_every == 0:
            with open(ps_file, "w") as f:
                json.dump(per_sample, f, indent=2)
            with open(traj_file, "w") as f:
                json.dump(trajectories, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start

    summary = {
        args.step_label: {
            "em": sum(ems) / len(ems) if ems else 0.0,
            "contains_em": sum(cms) / len(cms) if cms else 0.0,
            "f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "avg_tool_calls": sum(tool_counts) / len(tool_counts) if tool_counts else 0.0,
            "avg_relation_rewrites": (
                sum(rewrite_counts) / len(rewrite_counts) if rewrite_counts else 0.0
            ),
            "n_samples": len(ems),
            "elapsed_s": elapsed,
            "merged_checkpoint": str(args.merged_checkpoint),
            "intervention": "gold_relation_oracle_L3",
        }
    }

    with open(ps_file, "w") as f:
        json.dump(per_sample, f, indent=2)
    with open(traj_file, "w") as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved %d trajectories to %s", len(trajectories), traj_file)
    logger.info("Saved %d per-sample records to %s", len(per_sample), ps_file)
    logger.info("Saved summary to %s", args.output)

    print("\n=== Oracle L3 upper bound (G2@500, gold-relation rewrite) ===")
    print(f"{'Step':>6} {'EM':>8} {'ContEM':>8} {'F1':>8} {'Tools':>8} {'Rewrt':>8}")
    print("-" * 52)
    r = summary[args.step_label]
    print(
        f"{args.step_label:>6} {r['em']:>8.4f} {r['contains_em']:>8.4f} "
        f"{r['f1']:>8.4f} {r['avg_tool_calls']:>8.2f} {r['avg_relation_rewrites']:>8.2f}"
    )


# ---------------------------------------------------------------------------
# Smoke-test for the rewrite logic (no GPU, no KG server). Run with::
#     python scripts/task_s12_oracle_l3.py --selftest
# ---------------------------------------------------------------------------


def _selftest() -> None:
    cases = [
        # (policy_relation, gold_relations, expected_oracle_choice)
        ("sports.team.mascot",                         # mistyped — gold is mascot.team
         ["sports.mascot.team", "sports.sports_team.championships"],
         "sports.mascot.team"),
        ("sports.sports_team.champions",               # close to championships
         ["sports.mascot.team", "sports.sports_team.championships"],
         "sports.sports_team.championships"),
        ("",                                           # policy emitted no relation
         ["people.person.place_of_birth"],
         "people.person.place_of_birth"),
        ("anything",                                   # no gold relations -> None
         [],
         None),
    ]
    for q, golds, expected in cases:
        got = pick_oracle_relation(q, golds)
        assert got == expected, f"pick_oracle_relation({q!r}, {golds!r}) -> {got!r}, expected {expected!r}"
        print(f"OK: ({q!r}, {golds!r}) -> {got!r}")

    # End-to-end parse + rewrite on a fake response string.
    fake_resp = "<think>x</think><search>get_tail_entities(Lou Seal, sports.team.mascot)</search>"
    parsed = parse_search_call(fake_resp)
    assert parsed is not None, "parse_search_call returned None on a valid call"
    action, entity, policy_relation = parsed
    assert action == "get_tail_entities" and entity == "Lou Seal"
    assert policy_relation == "sports.team.mascot"
    chosen = pick_oracle_relation(policy_relation, ["sports.mascot.team",
                                                    "sports.sports_team.championships"])
    assert chosen == "sports.mascot.team", f"unexpected oracle pick: {chosen!r}"
    print(f"OK: end-to-end parse+rewrite on fake response -> {chosen!r}")

    # collect_gold_relations dedup + order preservation
    rels = collect_gold_relations([
        ["A", "rel.x", "B"],
        ["B", "rel.y", "C"],
        ["C", "rel.x", "D"],  # duplicate
    ])
    assert rels == ["rel.x", "rel.y"], f"unexpected gold relations: {rels!r}"
    print(f"OK: collect_gold_relations dedup -> {rels!r}")

    print("\nAll selftests passed.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
