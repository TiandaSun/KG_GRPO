"""Task 33 L3: Wikidata Probe + Anthropic LLM Rubric Audit.

For 300 sampled CWQ questions:
1. Extract the Freebase MID from the gold answer
2. Map MID -> Wikidata QID via P646 (Freebase identifier)
3. Query Wikidata to reconstruct what the answer was around 2015 (when Freebase was frozen)
4. Use Anthropic claude-haiku-4-5 with a rubric prompt to classify each item:
   - T  (temporal drift): answer was correct in 2015 but changed by 2025
   - I1 (mis-grounded): annotation error at creation time
   - I2 (incomplete gold): multiple valid answers, some missing
   - I3 (ambiguous): question is inherently unclear
   - R  (residual): can't classify reliably

Cost: ~$5-15 with claude-haiku-4-5.

Prerequisites:
    export ANTHROPIC_API_KEY="sk-ant-..."
    pip install anthropic SPARQLWrapper

Usage:
    python scripts/task33_l3_wikidata_probe.py \
        --input data/freebase/cwq_original_train.json \
        --l1 results/task33_l1_lexical.json \
        --l2 results/task33_l2_volatility.json \
        --output results/task33_l3_classifications.json \
        --n_sample 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "KG-GRPO-Research/1.0"

RUBRIC_PROMPT = """You are an expert annotator auditing a knowledge-graph QA dataset (ComplexWebQuestions, CWQ).
CWQ was created in 2018 from Freebase, which was frozen in 2015. The Wikidata reference is from 2025.

You will receive:
- A natural-language question
- The gold answer (from CWQ/Freebase)
- The current Wikidata answer (from 2025), if found
- The historical Wikidata answer (around 2015), if reconstructable
- Lexical/volatility flags from prior analysis

Your task: classify the *root cause* of any disagreement between the gold answer and the current Wikidata answer.

Return EXACTLY one of these category codes:
- T   = Temporal drift: gold answer was correct in 2015 but the fact has changed by 2025 (e.g., president changed, player traded). The annotation is correct relative to the frozen 2015 snapshot.
- I1  = Mis-grounded: annotation error at creation time. The gold answer was wrong even in 2015.
- I2  = Incomplete gold: multiple valid answers exist; gold lists only one. Other answers are also correct.
- I3  = Ambiguous: question is inherently unclear (multiple interpretations possible).
- R   = Residual / can't classify reliably.

Respond with ONLY the category code (T, I1, I2, I3, or R), followed by a one-sentence reason.
Example: "T - Donald Trump was president in 2017-2021, gold reflects this."
"""


def query_wikidata_by_freebase_mid(mid: str) -> dict:
    """Map a Freebase MID (e.g., 'm.0d05w3') to Wikidata QID via P646.

    Returns: {"qid": str, "label": str, "found": bool}
    """
    # Wikidata stores Freebase IDs in P646
    query = f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item wdt:P646 "/{mid.replace('.', '/')}" .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT 1
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    try:
        resp = requests.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            bindings = data.get("results", {}).get("bindings", [])
            if bindings:
                item_uri = bindings[0].get("item", {}).get("value", "")
                qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
                label = bindings[0].get("itemLabel", {}).get("value", "")
                return {"qid": qid, "label": label, "found": True}
    except Exception as e:
        logger.warning("MID->QID lookup failed for %s: %s", mid, e)
    return {"qid": "", "label": "", "found": False}


def call_anthropic_classify(client, question: str, gold: str, current_wikidata: str,
                              historical_wikidata: str, l1_flag: bool, l2_score: float) -> dict:
    """Call claude-haiku-4-5 with the rubric to classify a sample.

    Returns: {"category": str, "reason": str, "raw": str}
    """
    user_msg = (
        f"Question: {question}\n"
        f"Gold answer (CWQ/Freebase 2015): {gold}\n"
        f"Current Wikidata answer (2025): {current_wikidata or 'NOT FOUND'}\n"
        f"Historical Wikidata answer (~2015): {historical_wikidata or 'NOT RECONSTRUCTABLE'}\n"
        f"Lexical temporal flag: {'YES' if l1_flag else 'no'}\n"
        f"Predicate volatility (max): {l2_score:.2f}\n"
        f"\nClassify the disagreement (or absence of one) using the rubric. Reply with one category code + one-sentence reason."
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            system=RUBRIC_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text.strip()
        # Extract first token-like code
        first_word = raw.split()[0].rstrip(".,:-")
        category = "R"
        for code in ("T", "I1", "I2", "I3", "R"):
            if first_word.upper() == code:
                category = code
                break
        reason = raw[len(first_word):].lstrip(" -:,").strip()
        return {"category": category, "reason": reason, "raw": raw}
    except Exception as e:
        logger.error("Anthropic call failed: %s", e)
        return {"category": "ERROR", "reason": str(e), "raw": ""}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="L3: Wikidata + Anthropic LLM rubric audit")
    parser.add_argument("--input", type=Path, default=Path("data/freebase/cwq_original_train.json"))
    parser.add_argument("--l1", type=Path, default=Path("results/task33_l1_lexical.json"))
    parser.add_argument("--l2", type=Path, default=Path("results/task33_l2_volatility.json"))
    parser.add_argument("--output", type=Path, default=Path("results/task33_l3_classifications.json"))
    parser.add_argument("--n_sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep_s", type=float, default=1.0)
    parser.add_argument("--stratified", action="store_true",
                        help="Stratify sample across L1/L2 buckets (default uniform random)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY env var not set")
        logger.error("Run: export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Load data
    with open(args.input) as f:
        data = json.load(f)
    logger.info("Loaded %d CWQ questions", len(data))

    # Load L1/L2 flags into ID-keyed dicts
    with open(args.l1) as f:
        l1_data = json.load(f)
    l1_by_id = {x["ID"]: x for x in l1_data["per_question"]}

    with open(args.l2) as f:
        l2_data = json.load(f)
    l2_by_id = {x["ID"]: x for x in l2_data["per_question"]}

    # Sample
    rng = random.Random(args.seed)
    if args.stratified:
        # Stratify: 100 high-volatility, 100 medium, 100 low (or 100 lexical-temporal, etc.)
        high = [d for d in data if l2_by_id.get(d["ID"], {}).get("max_volatility", 0) >= 0.7]
        medium = [d for d in data if 0.3 <= l2_by_id.get(d["ID"], {}).get("max_volatility", 0) < 0.7]
        low = [d for d in data if l2_by_id.get(d["ID"], {}).get("max_volatility", 0) < 0.3]
        rng.shuffle(high); rng.shuffle(medium); rng.shuffle(low)
        per_bucket = args.n_sample // 3
        sample = high[:per_bucket] + medium[:per_bucket] + low[:per_bucket]
        rng.shuffle(sample)
    else:
        sample = rng.sample(data, min(args.n_sample, len(data)))

    logger.info("Sampling %d questions for L3 audit", len(sample))

    classifications = []
    t0 = time.time()
    counts = {"T": 0, "I1": 0, "I2": 0, "I3": 0, "R": 0, "ERROR": 0}

    for i, item in enumerate(sample):
        qid_lookup = {"qid": "", "label": "", "found": False}
        gold_label = ""
        if item.get("answers") and isinstance(item["answers"], list) and len(item["answers"]) > 0:
            ans = item["answers"][0]
            mid = ans.get("answer_id", "")
            gold_label = ans.get("answer", "")
            if mid:
                qid_lookup = query_wikidata_by_freebase_mid(mid)
                time.sleep(args.sleep_s)

        # For now, "current Wikidata answer" = the QID's English label (best effort)
        # "Historical Wikidata answer" = blank (would need temporal SPARQL — defer)
        current_wd = qid_lookup.get("label", "") if qid_lookup.get("found") else ""

        l1_flag = l1_by_id.get(item["ID"], {}).get("is_temporal", False)
        l2_score = l2_by_id.get(item["ID"], {}).get("max_volatility", 0.0)

        result = call_anthropic_classify(
            client,
            question=item["question"],
            gold=gold_label,
            current_wikidata=current_wd,
            historical_wikidata="",
            l1_flag=l1_flag,
            l2_score=l2_score,
        )

        classifications.append({
            "ID": item["ID"],
            "question": item["question"],
            "gold": gold_label,
            "freebase_mid": item.get("answers", [{}])[0].get("answer_id", ""),
            "wikidata_qid": qid_lookup.get("qid", ""),
            "current_wikidata_label": current_wd,
            "l1_temporal": l1_flag,
            "l2_volatility": l2_score,
            "category": result["category"],
            "reason": result["reason"],
            "raw_response": result["raw"],
        })

        counts[result["category"]] = counts.get(result["category"], 0) + 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            logger.info(
                "  %d/%d  T=%d I1=%d I2=%d I3=%d R=%d ERR=%d  (%.1fs/sample)",
                i + 1, len(sample), counts["T"], counts["I1"], counts["I2"],
                counts["I3"], counts["R"], counts["ERROR"], elapsed / (i + 1),
            )

        time.sleep(0.5)

    summary = {
        "n_total": len(sample),
        "counts": counts,
        "pct": {k: v / len(sample) for k, v in counts.items()},
        "elapsed_s": time.time() - t0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "classifications": classifications}, f, indent=2)
    logger.info("Saved to %s", args.output)

    print("\n=== L3 Classification Distribution ===")
    print(f"Total sampled: {len(sample)}")
    for cat in ("T", "I1", "I2", "I3", "R", "ERROR"):
        c = counts.get(cat, 0)
        print(f"  {cat:>5}  {c:4d}  ({100*c/len(sample):5.1f}%)")
    print(f"\nTemporal (T): {100*counts['T']/len(sample):.1f}%")
    print(f"Intrinsic (I1+I2+I3): {100*(counts['I1']+counts['I2']+counts['I3'])/len(sample):.1f}%")


if __name__ == "__main__":
    main()
