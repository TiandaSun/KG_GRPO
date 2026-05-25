"""Task 35 Step 2: Wikidata Pre-Materialization for KGQAGen-10k.

For each question in dev+test splits, extract all QIDs from proof triples and
the seed entity. Query Wikidata SPARQL for the 1-hop neighborhood of each QID
and cache to local JSON files.

Cache format (per QID):
    data/wikidata_cache/{qid}.json
    {
        "qid": "Q12345",
        "label": "Some Entity",
        "outgoing": {  // qid -> {prop_id: [(target_qid, target_label), ...]}
            "P31": [["Q5", "human"], ...],
            ...
        },
        "incoming": {
            "P57": [["Q8888", "Some Film"], ...],
            ...
        }
    }

Resilient: skip QIDs already cached, save progress after every batch.

Usage:
    python scripts/task35_prematerialize_wikidata.py \
        --output_dir data/wikidata_cache \
        --max_qids 0  # 0 = all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "KG-GRPO-Research/1.0 (research project)"

# Pattern to extract Q-numbers from proof labels like "Fei Mu (Q705561)"
QID_PATTERN = re.compile(r"\(Q(\d+)\)")
PID_PATTERN = re.compile(r"\(P(\d+)\)")


def extract_qids_from_kgqagen(dev_data: list[dict], test_data: list[dict]) -> set[str]:
    """Extract all unique QIDs from proof triples + seed entities."""
    qids: set[str] = set()
    for split in (dev_data, test_data):
        for sample in split:
            seed = sample.get("seed", "")
            if seed.startswith("Q"):
                qids.add(seed)
            proof = sample.get("proof", [])
            if isinstance(proof, str):
                # Re-parse from string representation if needed
                try:
                    proof = json.loads(proof.replace("'", '"'))
                except Exception:
                    proof = []
            if isinstance(proof, list):
                for triple in proof:
                    if isinstance(triple, list) and len(triple) == 3:
                        for elem in triple:
                            for m in QID_PATTERN.findall(str(elem)):
                                qids.add(f"Q{m}")
    return qids


def query_wikidata_neighborhood(qid: str, max_retries: int = 3) -> dict:
    """Query 1-hop outgoing + incoming neighborhood for a QID.

    Returns:
        {"qid": qid, "label": str, "outgoing": {prop: [...]}, "incoming": {prop: [...]}}
    """
    # Outgoing: ?prop ?target where wd:qid ?prop ?target
    outgoing_query = f"""
    SELECT ?prop ?propLabel ?target ?targetLabel WHERE {{
      wd:{qid} ?p ?target .
      ?prop wikibase:directClaim ?p .
      FILTER(STRSTARTS(STR(?target), "http://www.wikidata.org/entity/Q"))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT 500
    """

    # Incoming: ?source ?prop wd:qid
    incoming_query = f"""
    SELECT ?prop ?propLabel ?source ?sourceLabel WHERE {{
      ?source ?p wd:{qid} .
      ?prop wikibase:directClaim ?p .
      FILTER(STRSTARTS(STR(?source), "http://www.wikidata.org/entity/Q"))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT 500
    """

    # Label query
    label_query = f"""
    SELECT ?qidLabel WHERE {{
      wd:{qid} rdfs:label ?qidLabel .
      FILTER(LANG(?qidLabel) = "en")
    }}
    LIMIT 1
    """

    result = {"qid": qid, "label": "", "outgoing": {}, "incoming": {}}

    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}

    for query_name, query in [("label", label_query), ("outgoing", outgoing_query), ("incoming", incoming_query)]:
        for attempt in range(max_retries):
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

                    if query_name == "label":
                        if bindings:
                            result["label"] = bindings[0].get("qidLabel", {}).get("value", "")
                    elif query_name == "outgoing":
                        for b in bindings:
                            prop_uri = b.get("prop", {}).get("value", "")
                            prop_id = prop_uri.rsplit("/", 1)[-1] if prop_uri else ""
                            target_uri = b.get("target", {}).get("value", "")
                            target_qid = target_uri.rsplit("/", 1)[-1] if target_uri else ""
                            target_label = b.get("targetLabel", {}).get("value", "")
                            if prop_id and target_qid:
                                result["outgoing"].setdefault(prop_id, []).append([target_qid, target_label])
                    elif query_name == "incoming":
                        for b in bindings:
                            prop_uri = b.get("prop", {}).get("value", "")
                            prop_id = prop_uri.rsplit("/", 1)[-1] if prop_uri else ""
                            source_uri = b.get("source", {}).get("value", "")
                            source_qid = source_uri.rsplit("/", 1)[-1] if source_uri else ""
                            source_label = b.get("sourceLabel", {}).get("value", "")
                            if prop_id and source_qid:
                                result["incoming"].setdefault(prop_id, []).append([source_qid, source_label])
                    break  # success, next query
                elif resp.status_code == 429:
                    # Rate limited
                    wait = 30 * (attempt + 1)
                    logger.warning("Rate limited (429) on %s for %s, sleeping %ds", query_name, qid, wait)
                    time.sleep(wait)
                else:
                    logger.warning("HTTP %d on %s for %s", resp.status_code, query_name, qid)
                    time.sleep(5)
            except requests.exceptions.RequestException as e:
                logger.warning("Network error on %s for %s: %s", query_name, qid, e)
                time.sleep(10)

    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Pre-materialize Wikidata neighborhoods for KGQAGen-10k")
    parser.add_argument("--output_dir", type=Path, default=Path("data/wikidata_cache"))
    parser.add_argument("--max_qids", type=int, default=0, help="0 = all")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--sleep_s", type=float, default=1.0, help="Sleep between QID queries")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load KGQAGen-10k
    import os
    os.environ.setdefault("HF_HOME", "/scratch/u6gg/ts1201.u6gg/hf_cache")
    from datasets import load_dataset

    logger.info("Loading KGQAGen-10k from HuggingFace")
    ds = load_dataset("lianglz/KGQAGen-10k")
    dev_data = [dict(s) for s in ds["dev"]]
    test_data = [dict(s) for s in ds["test"]]
    logger.info("Dev: %d, Test: %d", len(dev_data), len(test_data))

    qids = sorted(extract_qids_from_kgqagen(dev_data, test_data))
    logger.info("Extracted %d unique QIDs from dev+test", len(qids))

    if args.max_qids > 0:
        qids = qids[args.start_idx:args.start_idx + args.max_qids]
    elif args.start_idx > 0:
        qids = qids[args.start_idx:]
    logger.info("Will process %d QIDs (start_idx=%d)", len(qids), args.start_idx)

    n_already_cached = 0
    n_fetched = 0
    n_failed = 0
    t0 = time.time()

    for i, qid in enumerate(qids):
        cache_file = args.output_dir / f"{qid}.json"
        if cache_file.exists():
            n_already_cached += 1
            continue

        try:
            result = query_wikidata_neighborhood(qid)
            n_neighbors = sum(len(v) for v in result["outgoing"].values()) + sum(len(v) for v in result["incoming"].values())
            if n_neighbors == 0 and not result["label"]:
                logger.warning("Empty result for %s", qid)
                n_failed += 1
            with open(cache_file, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=1)
            n_fetched += 1
        except Exception as e:
            logger.error("Failed for %s: %s", qid, e)
            n_failed += 1

        time.sleep(args.sleep_s)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(qids) - i - 1) / max(rate, 0.001) / 60
            logger.info(
                "  %d/%d  cached=%d fetched=%d failed=%d  %.2f q/s  ETA=%.0fm",
                i + 1, len(qids), n_already_cached, n_fetched, n_failed, rate, eta,
            )

    elapsed = time.time() - t0
    logger.info(
        "Done. Processed %d QIDs in %.0fs. cached=%d fetched=%d failed=%d",
        len(qids), elapsed, n_already_cached, n_fetched, n_failed,
    )

    # Write summary
    summary_file = args.output_dir / "_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "n_total_qids_in_dataset": len(extract_qids_from_kgqagen(dev_data, test_data)),
            "n_processed_this_run": len(qids),
            "n_already_cached": n_already_cached,
            "n_fetched": n_fetched,
            "n_failed": n_failed,
            "elapsed_s": elapsed,
        }, f, indent=2)


if __name__ == "__main__":
    main()
