"""V15-Q3: Token-entropy time series across I-Self merged checkpoints.

Differentiates from Yu et al. 2025 (uniform action-distribution collapse) by
showing SELECTIVE collapse on `<search>...</search>` tokens (INSIDE) vs the
rest of the assistant text (OUTSIDE).

For each checkpoint at training step N in {200, 250, 300, 400}:
  1. Load merged 7B model (`outputs/verl-sft-cwq-39i_self_step{N}-merged`).
  2. Run greedy multi-turn agent loop (max_turns=5, max_new_tokens=512) on the
     SHARED 500-Q subset (file passed via --subset_ids), with live KG tool calls.
  3. During each generation step, capture per-token logits, compute Shannon
     entropy in NATS, and classify each generated token as INSIDE or OUTSIDE
     based on the running text seen so far (whether the cursor is currently
     between an unmatched `<search>` and the next `</search>`).
  4. Aggregate per-step mean entropies.

Outputs (under --out_root, default `results/phase7/v15_q3_entropy/`):
  step_{N}/inside_entropy.json        # mean + per-token list
  step_{N}/outside_entropy.json
  entropy_curve.png                   # x = step, two lines (inside/outside)
  summary.json                        # one record per step

ENTROPY UNIT: nats (natural log). Documented here so downstream papers cite it
correctly. Conversion: bits = nats / ln(2).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402  headless backend
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

logger = logging.getLogger(__name__)


SEARCH_OPEN = "<search>"
SEARCH_CLOSE = "</search>"


# ---------------------------------------------------------------------------
# KG tool plumbing — mirrors scripts/eval_with_tools.py
# ---------------------------------------------------------------------------


def call_kg_server(action: str, entity: str, relation: str | None, server_url: str) -> str:
    payload: dict[str, Any] = {"action": action, "entity": entity}
    if relation:
        payload["relation"] = relation
    try:
        resp = requests.post(f"{server_url}/retrieve", json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", data.get("result", []))
            if isinstance(results, list) and len(results) > 10:
                results = results[:10]
            return json.dumps(results, ensure_ascii=False)
        return f"Error: {resp.status_code}"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def parse_search_call(text: str) -> tuple[str, str, str | None] | None:
    match = re.search(r"<search>\s*(\w+)\(([^)]*)\)\s*</search>", text, re.DOTALL)
    if not match:
        return None
    action = match.group(1)
    args = [a.strip().strip("'\"") for a in match.group(2).split(",") if a.strip()]
    entity = args[0] if args else ""
    relation = args[1] if len(args) > 1 else None
    return action, entity, relation


# ---------------------------------------------------------------------------
# Inside/outside classification helper
# ---------------------------------------------------------------------------


def classify_tokens_inside_outside(
    so_far_text_before_step: str,
    new_token_strs: list[str],
) -> list[bool]:
    """For each newly emitted token, decide if it falls INSIDE a `<search>`
    block (True) or OUTSIDE (False), based on the running text up to and
    INCLUDING that token.

    Rule: a token is INSIDE iff at the moment it was emitted, the running text
    contains an unmatched `<search>` (i.e. last `<search>` is at a higher
    position than the last `</search>`).
    """
    cursor = so_far_text_before_step
    flags: list[bool] = []
    for tok in new_token_strs:
        cursor += tok
        last_open = cursor.rfind(SEARCH_OPEN)
        last_close = cursor.rfind(SEARCH_CLOSE)
        # INSIDE if there is an unmatched open. We treat the closing tag itself
        # as still INSIDE because it's part of the search call structure.
        if last_open == -1:
            flags.append(False)
        elif last_close == -1:
            flags.append(True)
        else:
            flags.append(last_open > last_close)
    return flags


# ---------------------------------------------------------------------------
# Greedy generation that records per-token logits + token strings
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_generate_record_entropy(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, list[float], list[str]]:
    """Run greedy decode step-by-step, returning the generated ids, per-step
    Shannon entropy in NATS over the full vocab logits, and decoded per-token
    strings.
    """
    device = input_ids.device
    generated = input_ids.clone()
    past = None
    entropies: list[float] = []
    new_token_ids: list[int] = []

    cur = generated
    for _ in range(max_new_tokens):
        out = model(
            input_ids=cur if past is None else cur[:, -1:],
            past_key_values=past,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]  # (1, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # Shannon entropy in nats: H = -sum p * log p
        ent = -(probs * log_probs).sum(dim=-1).item()
        entropies.append(ent)
        next_id = int(torch.argmax(logits, dim=-1).item())
        new_token_ids.append(next_id)
        past = out.past_key_values
        cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos_token_id:
            break
    generated = cur
    # Decode each new token individually so we can map per-token-string to flags
    new_token_strs = [
        tokenizer.decode([tid], skip_special_tokens=False) for tid in new_token_ids
    ]
    return generated, entropies, new_token_strs


# ---------------------------------------------------------------------------
# Multi-turn agent loop with entropy capture
# ---------------------------------------------------------------------------


def run_one_question_with_entropy(
    model: Any,
    tokenizer: Any,
    question: str,
    system_prompt: str,
    kg_server_url: str,
    device: str,
    max_turns: int = 5,
    max_new_tokens: int = 512,
) -> tuple[list[float], list[float], int]:
    """Returns (inside_entropies, outside_entropies, num_tool_calls)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    inside_acc: list[float] = []
    outside_acc: list[float] = []
    num_tool_calls = 0
    full_assistant_text = ""

    for _turn in range(max_turns):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=4096
        ).to(device)

        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        _gen_ids, ents, tok_strs = greedy_generate_record_entropy(
            model, tokenizer, enc["input_ids"], max_new_tokens, eos
        )

        # Classify each new token as inside/outside relative to the running
        # assistant text of THIS turn (search tags don't span turns in our
        # template, so reset to "" each turn).
        flags = classify_tokens_inside_outside("", tok_strs)
        for ent, is_inside in zip(ents, flags):
            (inside_acc if is_inside else outside_acc).append(ent)

        response_text = "".join(tok_strs)
        # Strip trailing eos visualisation if present
        if tokenizer.eos_token and response_text.endswith(tokenizer.eos_token):
            response_text = response_text[: -len(tokenizer.eos_token)]
        full_assistant_text += response_text

        if "<answer>" in response_text and "</answer>" in response_text:
            messages.append({"role": "assistant", "content": response_text})
            break

        call = parse_search_call(response_text)
        if call is None:
            messages.append({"role": "assistant", "content": response_text})
            break

        action, entity, relation = call
        num_tool_calls += 1
        kg_result = call_kg_server(action, entity, relation, kg_server_url)
        messages.append({"role": "assistant", "content": response_text})
        messages.append(
            {"role": "user", "content": f"<tool_response>{kg_result}</tool_response>"}
        )

    return inside_acc, outside_acc, num_tool_calls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_subset_ids(subset_ids_path: Path) -> set[str]:
    with open(subset_ids_path) as f:
        d = json.load(f)
    if isinstance(d, list):
        return set(map(str, d))
    if "subset_ids" in d:
        return set(map(str, d["subset_ids"]))
    if "filter_ids" in d:
        return set(map(str, d["filter_ids"]))
    raise ValueError(f"Cannot extract IDs from {subset_ids_path}")


def select_rows_by_id(rows: list[dict], id_set: set[str]) -> list[dict]:
    out = []
    for r in rows:
        sid = str(r.get("extra_info", {}).get("sample_id", ""))
        if sid in id_set:
            out.append(r)
    return out


def evaluate_step(
    step: int,
    ckpt_dir: Path,
    selected_rows: list[dict],
    kg_server_url: str,
    out_step_dir: Path,
    max_turns: int,
    max_new_tokens: int,
) -> dict:
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir missing: {ckpt_dir}")
    logger.info("=== step %d: loading %s ===", step, ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()

    system_prompt = selected_rows[0]["prompt"][0]["content"]

    all_inside: list[float] = []
    all_outside: list[float] = []
    t0 = time.time()
    for i, row in enumerate(selected_rows):
        prompt_msgs = row["prompt"]
        user_msg = next((m for m in prompt_msgs if m.get("role") == "user"), None)
        question = user_msg["content"] if user_msg else ""
        try:
            ins, outs, _ntools = run_one_question_with_entropy(
                model, tokenizer, question, system_prompt,
                kg_server_url, device="cuda",
                max_turns=max_turns, max_new_tokens=max_new_tokens,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("step %d q %d failed: %s", step, i, e)
            ins, outs = [], []
        all_inside.extend(ins)
        all_outside.extend(outs)
        if (i + 1) % 25 == 0:
            mean_in = sum(all_inside) / max(1, len(all_inside))
            mean_out = sum(all_outside) / max(1, len(all_outside))
            logger.info(
                "  step %d  %d/%d  mean_in=%.4f mean_out=%.4f  elapsed=%ds",
                step, i + 1, len(selected_rows), mean_in, mean_out,
                int(time.time() - t0),
            )

    out_step_dir.mkdir(parents=True, exist_ok=True)
    mean_inside = sum(all_inside) / max(1, len(all_inside))
    mean_outside = sum(all_outside) / max(1, len(all_outside))
    with open(out_step_dir / "inside_entropy.json", "w") as f:
        json.dump(
            {"unit": "nats", "count": len(all_inside), "mean": mean_inside,
             "values": all_inside},
            f,
        )
    with open(out_step_dir / "outside_entropy.json", "w") as f:
        json.dump(
            {"unit": "nats", "count": len(all_outside), "mean": mean_outside,
             "values": all_outside},
            f,
        )

    # Free GPU before next step
    del model
    torch.cuda.empty_cache()

    return {
        "step": step,
        "n_inside": len(all_inside),
        "n_outside": len(all_outside),
        "mean_inside": mean_inside,
        "mean_outside": mean_outside,
        "elapsed_s": time.time() - t0,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--subset_ids", type=Path, required=True)
    p.add_argument("--kg_server_url", type=str, required=True)
    p.add_argument(
        "--checkpoints_root", type=Path, default=Path("outputs"),
        help="Parent dir holding verl-sft-cwq-39i_self_step{N}-merged dirs.",
    )
    p.add_argument(
        "--ckpt_pattern", type=str,
        default="verl-sft-cwq-39i_self_step{step}-merged",
    )
    p.add_argument("--steps", type=str, default="200,250,300,400")
    p.add_argument(
        "--eval_data", type=Path,
        default=Path("data/freebase/verl_cwq/test.parquet"),
    )
    p.add_argument(
        "--out_root", type=Path,
        default=Path("results/phase7/v15_q3_entropy"),
    )
    p.add_argument("--max_turns", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=512)
    args = p.parse_args()

    # KG server health check
    try:
        r = requests.get(f"{args.kg_server_url}/health", timeout=10)
        logger.info("KG server: %s %s", r.status_code, r.text[:80])
    except Exception as e:  # noqa: BLE001
        logger.error("KG server unreachable at %s: %s", args.kg_server_url, e)
        sys.exit(1)

    id_set = load_subset_ids(args.subset_ids)
    logger.info("Loaded %d subset IDs", len(id_set))

    rows = pq.read_table(args.eval_data).to_pylist()
    selected = select_rows_by_id(rows, id_set)
    logger.info("Selected %d / %d rows", len(selected), len(rows))
    if not selected:
        raise SystemExit("No rows matched subset IDs.")

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)

    summary_records: list[dict] = []
    for step in steps:
        ckpt_dir = args.checkpoints_root / args.ckpt_pattern.format(step=step)
        out_step_dir = args.out_root / f"step_{step}"
        rec = evaluate_step(
            step=step,
            ckpt_dir=ckpt_dir,
            selected_rows=selected,
            kg_server_url=args.kg_server_url,
            out_step_dir=out_step_dir,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
        )
        summary_records.append(rec)
        # Incremental save so a crash mid-sweep keeps earlier results
        with open(args.out_root / "summary.json", "w") as f:
            json.dump({"unit": "nats", "records": summary_records}, f, indent=2)

    # Plot curve
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xs = [r["step"] for r in summary_records]
    ax.plot(xs, [r["mean_inside"] for r in summary_records],
            marker="o", label="inside <search>")
    ax.plot(xs, [r["mean_outside"] for r in summary_records],
            marker="s", label="outside <search>")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean per-token entropy (nats)")
    ax.set_title("V15-Q3: Inside vs outside <search> entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_root / "entropy_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", args.out_root / "entropy_curve.png")
    logger.info("Done. Summary:\n%s", json.dumps(summary_records, indent=2))


if __name__ == "__main__":
    main()
