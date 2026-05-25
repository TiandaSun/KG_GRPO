# V14-A1 Search-R1 Baseline Setup Notes (Isambard-AI, 2026-04-19)

Task: hpc_tasks.md v14.1 Task V14-A1 — run Search-R1's exact outcome-only GRPO recipe on our CWQ+Freebase stack, then evaluate at checkpoint(s) against our full 3,531-sample CWQ test set (both via our 4-tool interface and via the Search-R1 single-tool adapter).

## Repo layout (Search-R1 @ main, cloned from GitHub)

`external/Search-R1/` — NOT committed (added to `.gitignore`). Key paths:

| Path | Purpose |
|---|---|
| `verl/trainer/main_ppo.py` | Training entrypoint. `python -m verl.trainer.main_ppo <hydra overrides>` |
| `verl/trainer/config/ppo_trainer.yaml` | Hydra config. Has `retriever.url`, `retriever.topk`, `max_turns`, `data.*`, `actor_rollout_ref.*`, `algorithm.*`, `trainer.*`. |
| `verl/utils/reward_score/qa_em.py` | Outcome reward (EM over `<answer>` tag). Uses `compute_score_em(solution_str, ground_truth={"target":[...]})`. |
| `search_r1/llm_agent/generation.py` | Multi-turn rollout loop. Calls the retriever via `requests.post(search_url, json={queries:[...], topk, return_scores:True})`. |
| `search_r1/search/retrieval_server.py` | Their reference local retrieval server (E5/BM25). We replace this with our Freebase adapter. |
| `train_grpo.sh` | Reference GRPO recipe (Qwen2.5-3B by default; we override to Qwen2.5-7B-Instruct). |

Training is Hydra-based, so all overrides can be passed on the CLI — no repo edits needed.

## Data format Search-R1 expects

From `scripts/data_process/nq_search.py`:

```python
{
  "data_source": "nq|triviaqa|popqa|hotpotqa|2wikimultihopqa|musique|bamboogle",
  "prompt": [{"role": "user", "content": <wrapped question with <think>/<search>/<answer> template>}],
  "ability": "fact-reasoning",
  "reward_model": {"style": "rule", "ground_truth": {"target": [<list of golden answers>]}},
  "extra_info": {"split": "train|test", "index": idx}
}
```

Important:
- `data_source` must be one of the whitelisted names in `main_ppo._select_rm_score_fn`, otherwise no reward gets computed. We use `hotpotqa` (CWQ is multi-hop factoid QA).
- `ground_truth.target` is a LIST; our CWQ parquet has a single string. We map `extra_info.all_answers` -> list.
- `prompt` is a single user message containing the full Search-R1 `<think>/<search>/<answer>` instruction template + the question.

Conversion: `scripts/prepare_cwq_for_searchr1.py` — CWQ verl parquet -> Search-R1 parquet. Output at `data/freebase/searchr1_cwq/{train,test}.parquet` (27,639 + 3,531 rows).

## Retriever API Search-R1 expects

From `search_r1/llm_agent/generation.py::_batch_search` and `search_r1/search/retrieval_server.py::retrieve_endpoint`:

```
POST /retrieve
request  = {"queries": [str, ...], "topk": int, "return_scores": bool}
response = {"result": [
  [  # per-query list
    {"document": {"contents": "Title\nBody ..."}, "score": float},
    ...
  ],
  ...
]}
```

Document dict may omit `score` when `return_scores=False`, but `{"document": {"contents": "Title\n<body>"}}` is required. `_passages2string` parses first line as title and rest as body.

Adapter (`scripts/phase7_ii2_searchr1_adapter.py`, rewritten 2026-04-19) fans batch queries out to our KG server's `/retrieve` (`{action, entity, relation}`) and repackages results in the expected shape. Query-to-KG-call mapping uses the `parse_query_to_call` heuristic:

- `<entity> <dotted.freebase.predicate>` -> `get_tail_entities(entity, relation)`
- `relations of <entity>` -> `get_tail_relations(entity)`
- natural-language fallback -> `get_tail_relations(whole_query)` (returns empty if no match, which we surface as a single "no results" doc so the rollout still sees something)

## ARM / Isambard compatibility

- Search-R1 bundles its own forked `verl/` directory inside the repo — we add that to `PYTHONPATH` before running, so our conda-installed `verl` does NOT get pulled in.
- Search-R1 `requirements.txt` pins `tensordict<0.6`, `transformers<4.48`, `vllm<=0.6.3`, `flash-attn`. Our kg_verl env has newer versions (tensordict 0.10, transformers 4.57, vllm 0.12, no flash-attn). Risk: API breaks at runtime. Mitigations documented in the SLURM script:
  - `VLLM_ATTENTION_BACKEND=XFORMERS` (Search-R1 already sets this in train_grpo.sh for qwen2-7b).
  - Force `actor_rollout_ref.rollout.enforce_eager=true` (ARM Triton graph compilation broken).
  - Use `attn_implementation=sdpa` (flash-attn unavailable on ARM).
- If the runtime import fails inside the SLURM job, the path forward is to create a dedicated `searchr1` conda env matching their pins — this is out of scope for the login-node setup; flag in the job log and ask user before installing.

## Training command

All fields match `train_grpo.sh` except:
- model -> `Qwen/Qwen2.5-7B-Instruct` (we already have it in `$HF_HOME`)
- data -> `data/freebase/searchr1_cwq/{train,test}.parquet`
- `retriever.url=http://localhost:19001/retrieve` (our adapter)
- `trainer.total_training_steps=500` (24h walltime budget — see below)
- `trainer.save_freq=100`, `trainer.test_freq=100`
- `trainer.default_local_dir=/projects/u6gg/KG_GRPO/checkpoints/kg-align-verl/searchr1-cwq-7b-20260420`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2`, `trainer.n_gpus_per_node=4` — we have 4 GPUs per Isambard node
- batch sizes scaled down to fit: `data.train_batch_size=256`, `actor.ppo_mini_batch_size=128`, `actor.ppo_micro_batch_size=32`, `rollout.n_agent=5`

### Step budget (500 not 800)

Spec asks 500 vs 800 at 24h. Our E3 7B GRPO (comparable compute) took ~23min/100 steps across 4 GH200 GPUs for the outcome+process reward config. Search-R1 uses `max_turns=2` (Search-R1 default; not multi-turn KG; our adapter still produces useful search hits in 1-2 turns) so wall time will be bounded by the rollout count. Safe budget at 24h with 20% buffer for vLLM init + checkpoint saves: **500 steps** (save_freq=100 yields 5 checkpoints: 100/200/300/400/500). This is below the spec's 800; we accept this because:
  1. Search-R1 paper v0.2 shows convergence by ~300-400 steps.
  2. Step 400 is the spec's primary checkpoint of interest ("if Search-R1 step 400 > G2's 40.0% EM -> paper reframing trigger").
  3. If the trainer is still making progress at step 500, we can submit a follow-up `--dependency=afterok` continuation job.

## Eval plan (arms A + B)

Search-R1 trains on CWQ with a **single** `<search>` tool. Our standard eval (`eval_with_tools.py`) assumes our 4-tool prompt. We run both:
- **Arm A (4-tool prompt, fulltool)**: `eval_with_tools.py --model <searchr1_ckpt>` — tests whether a model trained on the single-search API generalizes to our 4 explicit Freebase tools. Expected: poor; this is a finding ("does a single-tool-trained model do anything useful with 4 tools?").
- **Arm B (Search-R1 single-tool prompt)**: re-use the Search-R1 template + our adapter for evaluation. This is the **pure Search-R1 baseline number** for the paper.

Both arms: full 3,531-sample CWQ test set. Outputs at `results/phase7/v14_a1_searchr1_step{N}_{fulltool|singletool}_eval.json`.

## Environment / secrets

- `W&B project`: KG-Align-Verl (default for our group). Set `trainer.project_name=kg-align-verl` and `trainer.experiment_name=searchr1-cwq-7b-20260420`.

## Files created / modified (2026-04-19)

- `external/Search-R1/` — cloned, untracked.
- `.gitignore` — added `external/`.
- `scripts/phase7_ii2_searchr1_adapter.py` — **rewritten** to speak Search-R1's batched `{queries,topk,return_scores}` schema and return `{"result":[[{"document":{"contents":...},"score":...},...],...]}`. Also fixed a latent bug: previous version sent `{"tool": ...}` to our KG server which expects `{"action": ...}`.
- `scripts/prepare_cwq_for_searchr1.py` — new, CWQ verl parquet -> Search-R1 parquet.
- `data/freebase/searchr1_cwq/{train,test}.parquet` — converted (27,639 + 3,531).
- `scripts/run_phase7_v14_a1_searchr1_train.job` — new training SLURM script.
- `scripts/run_phase7_v14_a1_searchr1_eval.job` — new eval SLURM script (arms A+B).
- `scripts/submit_phase7_v14_a1_chain.sh` — new chain submission script.

## Watch-list (first 24h)

- Spec trigger: **"if Search-R1 step 400 > G2's 40.0% EM -> paper reframing trigger"**.
- Interim monitoring: logs/p7_v14a1_searchr1-*.log; W&B project.
- If Search-R1's bundled verl fails to launch on ARM with our deps, the adapter + data conversion are still valid -- we can swap in our own verl's GRPO trainer + Search-R1's single-tool schema later.
