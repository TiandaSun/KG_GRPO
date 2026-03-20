"""Stage 5: GRPO training with curriculum learning.

Trains the SFT-warmed model with GRPO using KG path-alignment and format
reward functions. Uses a 3-phase curriculum: 1-hop → 2-hop → 3-hop.

Usage:
    python src/training/train_grpo.py --config configs/grpo_validation.yaml

    # Override SFT adapter path:
    python src/training/train_grpo.py --config configs/grpo_validation.yaml \
        --sft_adapter_path outputs/sft-warmup-v2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.rewards.kg_reward import format_reward_func, kg_reward_func

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CurriculumPhase:
    """Configuration for a single curriculum phase."""

    name: str
    max_hops: int
    max_steps: int
    run_name: str


@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO curriculum training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: str = "all-linear"
    lora_dropout: float = 0.05
    use_dora: bool = True
    use_rslora: bool = True

    # GRPO
    num_generations: int = 8
    max_completion_length: int = 512
    max_prompt_length: int = 256
    temperature: float = 1.0
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    save_steps: int = 200
    logging_steps: int = 10
    bf16: bool = True
    seed: int = 42
    optim: str = "adamw_8bit"
    report_to: str = "wandb"

    # Dr. GRPO / DAPO
    scale_rewards: bool = False
    beta: float = 0.02
    loss_type: str = "dr_grpo"
    num_iterations: int = 2
    mask_truncated_completions: bool = True

    # vLLM
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.4
    vllm_tensor_parallel_size: int = 1

    # Data
    train_file: str = "data/processed/conceptnet_qa_train.jsonl"
    val_file: str = "data/processed/conceptnet_qa_val.jsonl"

    # Curriculum
    phases: list[CurriculumPhase] = field(default_factory=lambda: [
        CurriculumPhase("phase1_1hop", 1, 350, "kg-align-grpo-phase1"),
        CurriculumPhase("phase2_2hop", 2, 350, "kg-align-grpo-phase2"),
        CurriculumPhase("phase3_all", 3, 300, "kg-align-grpo-phase3"),
    ])

    # Output
    output_dir: str = "outputs/grpo"
    sft_adapter_path: str = "outputs/sft-warmup"

    @classmethod
    def from_yaml(cls, path: Path) -> GRPOTrainingConfig:
        """Load config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        model_cfg = raw.get("model", {})
        lora_cfg = raw.get("lora", {})
        grpo_cfg = raw.get("grpo", {})
        data_cfg = raw.get("data", {})
        curriculum_cfg = raw.get("curriculum", {})
        output_cfg = raw.get("output", {})

        phases = []
        for p in curriculum_cfg.get("phases", []):
            phases.append(CurriculumPhase(
                name=p["name"],
                max_hops=p["max_hops"],
                max_steps=p["max_steps"],
                run_name=p.get("run_name", p["name"]),
            ))

        config = cls(
            model_name=model_cfg.get("name", cls.model_name),
            torch_dtype=model_cfg.get("torch_dtype", cls.torch_dtype),
            attn_implementation=model_cfg.get("attn_implementation", cls.attn_implementation),
            lora_r=lora_cfg.get("r", cls.lora_r),
            lora_alpha=lora_cfg.get("lora_alpha", cls.lora_alpha),
            target_modules=lora_cfg.get("target_modules", cls.target_modules),
            lora_dropout=lora_cfg.get("lora_dropout", cls.lora_dropout),
            use_dora=lora_cfg.get("use_dora", cls.use_dora),
            use_rslora=lora_cfg.get("use_rslora", cls.use_rslora),
            num_generations=grpo_cfg.get("num_generations", cls.num_generations),
            max_completion_length=grpo_cfg.get("max_completion_length", cls.max_completion_length),
            max_prompt_length=grpo_cfg.get("max_prompt_length", cls.max_prompt_length),
            temperature=grpo_cfg.get("temperature", cls.temperature),
            learning_rate=grpo_cfg.get("learning_rate", cls.learning_rate),
            per_device_train_batch_size=grpo_cfg.get("per_device_train_batch_size", cls.per_device_train_batch_size),
            gradient_accumulation_steps=grpo_cfg.get("gradient_accumulation_steps", cls.gradient_accumulation_steps),
            gradient_checkpointing=grpo_cfg.get("gradient_checkpointing", cls.gradient_checkpointing),
            save_steps=grpo_cfg.get("save_steps", cls.save_steps),
            logging_steps=grpo_cfg.get("logging_steps", cls.logging_steps),
            bf16=grpo_cfg.get("bf16", cls.bf16),
            seed=grpo_cfg.get("seed", cls.seed),
            optim=grpo_cfg.get("optim", cls.optim),
            report_to=grpo_cfg.get("report_to", cls.report_to),
            scale_rewards=grpo_cfg.get("scale_rewards", cls.scale_rewards),
            beta=grpo_cfg.get("beta", cls.beta),
            loss_type=grpo_cfg.get("loss_type", cls.loss_type),
            num_iterations=grpo_cfg.get("num_iterations", cls.num_iterations),
            mask_truncated_completions=grpo_cfg.get("mask_truncated_completions", cls.mask_truncated_completions),
            use_vllm=grpo_cfg.get("use_vllm", cls.use_vllm),
            vllm_mode=grpo_cfg.get("vllm_mode", cls.vllm_mode),
            vllm_gpu_memory_utilization=grpo_cfg.get("vllm_gpu_memory_utilization", cls.vllm_gpu_memory_utilization),
            vllm_tensor_parallel_size=grpo_cfg.get("vllm_tensor_parallel_size", cls.vllm_tensor_parallel_size),
            train_file=data_cfg.get("train_file", cls.train_file),
            val_file=data_cfg.get("val_file", cls.val_file),
            output_dir=output_cfg.get("dir", cls.output_dir),
            sft_adapter_path=output_cfg.get("sft_adapter_path", cls.sft_adapter_path),
        )
        if phases:
            config.phases = phases
        return config


def load_grpo_dataset(path: Path, max_hops: int) -> Dataset:
    """Load QA pairs and filter by hop count for curriculum.

    Formats as conversational prompts with metadata columns preserved
    for reward function kwargs.

    Args:
        path: Path to JSONL file.
        max_hops: Maximum number of hops to include (1, 2, or 3).

    Returns:
        Dataset with 'prompt', 'kg_path', 'gold_answer_short', 'hops' columns.
    """
    prompts: list[list[dict[str, str]]] = []
    kg_paths: list[list[list[str]]] = []
    gold_answers: list[str] = []
    hops_list: list[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if record.get("is_negative", False):
                continue

            record_hops = record.get("hops", 1)
            if record_hops > max_hops:
                continue

            prompts.append([{"role": "user", "content": record["question"]}])
            kg_paths.append(record["kg_path"])
            gold_answers.append(record["gold_answer_short"])
            hops_list.append(record_hops)

    logger.info(
        "Loaded %d QA pairs from %s (max_hops=%d)",
        len(prompts), path, max_hops,
    )
    return Dataset.from_dict({
        "prompt": prompts,
        "kg_path": kg_paths,
        "gold_answer_short": gold_answers,
        "hops": hops_list,
    })


def _build_lora_config(config: GRPOTrainingConfig) -> LoraConfig:
    """Build LoRA config with DoRA/rsLoRA fallbacks."""
    lora_kwargs: dict[str, Any] = {
        "r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "target_modules": config.target_modules,
        "lora_dropout": config.lora_dropout,
        "task_type": TaskType.CAUSAL_LM,
        "bias": "none",
    }

    if config.use_dora:
        try:
            LoraConfig(use_dora=True, r=4, target_modules=["q_proj"])
            lora_kwargs["use_dora"] = True
            logger.info("DoRA enabled")
        except (TypeError, ValueError):
            logger.warning("DoRA not supported, using standard LoRA")

    if config.use_rslora:
        try:
            LoraConfig(use_rslora=True, r=4, target_modules=["q_proj"])
            lora_kwargs["use_rslora"] = True
            logger.info("rsLoRA enabled")
        except (TypeError, ValueError):
            logger.warning("rsLoRA not supported, skipping")

    return LoraConfig(**lora_kwargs)


def _build_grpo_config(
    config: GRPOTrainingConfig,
    phase: CurriculumPhase,
    output_dir: str,
) -> GRPOConfig:
    """Build GRPOConfig with graceful fallbacks for optional features."""
    grpo_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "max_steps": phase.max_steps,
        "num_generations": config.num_generations,
        "max_completion_length": config.max_completion_length,
        "max_prompt_length": config.max_prompt_length,
        "temperature": config.temperature,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "save_steps": config.save_steps,
        "logging_steps": config.logging_steps,
        "bf16": config.bf16,
        "seed": config.seed,
        "optim": config.optim,
        "report_to": config.report_to,
        "run_name": phase.run_name,
        "beta": config.beta,
        "remove_unused_columns": False,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if config.gradient_checkpointing else None,
    }

    # Optional GRPOConfig params — added to dict now, validated at GRPOConfig() call.
    # If the installed TRL version doesn't support a param, GRPOConfig(**grpo_kwargs)
    # will raise TypeError. We catch that below and retry without unsupported params.
    optional_params = {
        "scale_rewards": config.scale_rewards,
        "loss_type": config.loss_type,
        "num_iterations": config.num_iterations,
        "mask_truncated_completions": config.mask_truncated_completions,
    }
    grpo_kwargs.update(optional_params)

    # vLLM colocate mode — check availability before setting flags
    if config.use_vllm:
        try:
            import vllm  # noqa: F401
            import torch
            num_gpus = torch.cuda.device_count()
            tp_size = config.vllm_tensor_parallel_size
            # TRL requires vllm_tensor_parallel_size to divide world_size evenly.
            # world_size = number of training processes (set by torchrun/accelerate).
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if tp_size > num_gpus:
                logger.warning(
                    "Config requests TP=%d but only %d GPU(s) available. "
                    "Falling back to TP=%d.",
                    tp_size, num_gpus, num_gpus,
                )
                tp_size = num_gpus
            if world_size > 0 and world_size % tp_size != 0:
                fallback_tp = max(
                    t for t in range(1, tp_size + 1)
                    if world_size % t == 0 and t <= num_gpus
                )
                logger.warning(
                    "TP=%d doesn't divide world_size=%d evenly. "
                    "Falling back to TP=%d.",
                    tp_size, world_size, fallback_tp,
                )
                tp_size = fallback_tp
            grpo_kwargs["use_vllm"] = True
            grpo_kwargs["vllm_mode"] = config.vllm_mode
            grpo_kwargs["vllm_gpu_memory_utilization"] = config.vllm_gpu_memory_utilization
            if tp_size > 1:
                grpo_kwargs["vllm_tensor_parallel_size"] = tp_size
            logger.info(
                "vLLM colocate mode configured (TP=%d, mem_util=%.1f, world_size=%d)",
                tp_size,
                config.vllm_gpu_memory_utilization,
                world_size,
            )
        except ImportError:
            logger.warning(
                "vLLM not installed but use_vllm=True in config. "
                "Falling back to HF generate. Install with: pip install vllm"
            )
        except Exception:
            logger.warning("vLLM configuration failed, will use HF generate")

    # EOS configuration: ensure generation stops at both <|endoftext|> (151643)
    # and <|im_end|> (151645). The Instruct model's tokenizer already has
    # eos_token_id=151645, but we pass both explicitly for robustness.
    # vLLM uses "stop_token_ids" instead of "eos_token_id".
    if grpo_kwargs.get("use_vllm", False):
        grpo_kwargs["generation_kwargs"] = {"stop_token_ids": [151643, 151645]}
    else:
        grpo_kwargs["generation_kwargs"] = {"eos_token_id": [151643, 151645]}

    # Try creating GRPOConfig; if unsupported params cause TypeError, retry
    # by removing them one at a time.
    try:
        return GRPOConfig(**grpo_kwargs)
    except TypeError as e:
        error_msg = str(e)
        for param_name in list(optional_params.keys()):
            if param_name in error_msg:
                logger.warning(
                    "%s parameter not supported by this TRL version, removing",
                    param_name,
                )
                grpo_kwargs.pop(param_name, None)
        return GRPOConfig(**grpo_kwargs)


def _load_model_for_phase(
    config: GRPOTrainingConfig,
    adapter_path: str,
    is_first_phase: bool,
) -> tuple[Any, Any]:
    """Load model and tokenizer for a curriculum phase.

    For the first phase, loads the SFT adapter. For subsequent phases,
    loads the adapter from the previous phase's output.

    Returns:
        (model, tokenizer) tuple.
    """
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only batch generation

    # Load base model with Flash Attention fallback
    # NOTE: Do NOT use device_map="auto" — it conflicts with accelerate DDP.
    # Let the Trainer / accelerate handle device placement.
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
    }
    try:
        model_kwargs["attn_implementation"] = config.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        logger.info("Loaded base model with %s", config.attn_implementation)
    except (ValueError, ImportError):
        logger.warning("Flash Attention 2 not available, using default attention")
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    # Load LoRA adapter from previous phase or SFT warmup
    from peft import PeftModel

    adapter_dir = Path(adapter_path)
    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        logger.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        # No previous adapter — apply fresh LoRA config
        logger.info("No adapter found at %s, applying fresh LoRA config", adapter_path)
        lora_config = _build_lora_config(config)
        from peft import get_peft_model
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # The Instruct model's tokenizer has eos_token_id=151645 (<|im_end|>),
    # which TRL's GRPOTrainer picks up automatically for completion masking.
    # generation_kwargs in GRPOConfig adds both EOS tokens for generation.

    return model, tokenizer


def _get_adapter_path_for_phase(config: GRPOTrainingConfig, phase_idx: int) -> str:
    """Determine the adapter path for a given phase index.

    Phase 0 loads from SFT adapter. Phase N loads from phase N-1's output.
    """
    if phase_idx == 0:
        return config.sft_adapter_path
    prev_phase = config.phases[phase_idx - 1]
    return str(Path(config.output_dir) / prev_phase.name)


def run_single_phase(config: GRPOTrainingConfig, phase_idx: int) -> None:
    """Run a single curriculum phase.

    Each phase runs as a separate OS process (called from SLURM script) to
    guarantee full GPU memory cleanup between phases. vLLM CUDA graphs and
    memory pools do not release with gc.collect()/torch.cuda.empty_cache().

    Args:
        config: Full training configuration.
        phase_idx: Index into config.phases (0, 1, or 2).
    """
    if phase_idx < 0 or phase_idx >= len(config.phases):
        raise ValueError(
            f"phase_index={phase_idx} out of range [0, {len(config.phases)})"
        )

    phase = config.phases[phase_idx]
    adapter_path = _get_adapter_path_for_phase(config, phase_idx)

    logger.info(
        "=== Starting %s (max_hops=%d, max_steps=%d) ===",
        phase.name, phase.max_hops, phase.max_steps,
    )

    # Phase output directory
    phase_output_dir = str(Path(config.output_dir) / phase.name)
    Path(phase_output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset for this phase
    train_dataset = load_grpo_dataset(Path(config.train_file), phase.max_hops)

    if len(train_dataset) == 0:
        logger.warning("No training data for %s, skipping", phase.name)
        return

    # Load model from previous phase's adapter (or SFT)
    model, tokenizer = _load_model_for_phase(
        config, adapter_path, is_first_phase=(phase_idx == 0)
    )

    # Build GRPO config
    grpo_config = _build_grpo_config(config, phase, phase_output_dir)

    # Create trainer
    reward_funcs = [kg_reward_func, format_reward_func]
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        tokenizer=tokenizer,
    )

    # Verify EOS token setup
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    trainer_eos = getattr(trainer, "eos_token_id", "N/A")
    logger.info(
        "trainer.eos_token_id=%s, tokenizer.eos_token_id=%s, <|im_end|>=%s",
        trainer_eos, tokenizer.eos_token_id, im_end_id,
    )

    # Train
    logger.info("Training %s with %d samples", phase.name, len(train_dataset))
    trainer.train()

    # Save adapter
    trainer.save_model(phase_output_dir)
    tokenizer.save_pretrained(phase_output_dir)
    logger.info("Phase %s adapter saved to %s", phase.name, phase_output_dir)

    logger.info("=== Phase %s complete ===", phase.name)


def run_curriculum_training(config: GRPOTrainingConfig) -> None:
    """Run all GRPO curriculum phases sequentially in-process.

    NOTE: This runs all phases in a single process. For 7B models with vLLM,
    prefer using --phase_index to run each phase as a separate process (called
    from the SLURM script) to avoid vLLM GPU memory leaks between phases.
    """
    for phase_idx in range(len(config.phases)):
        run_single_phase(config, phase_idx)
        # Free GPU memory between phases to avoid OOM with vLLM
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared between phases")
    logger.info("=== GRPO curriculum training complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: GRPO curriculum training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/grpo_validation.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--sft_adapter_path",
        type=str,
        default=None,
        help="Override SFT adapter path from config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--pilot_steps",
        type=int,
        default=None,
        help="Run pilot: override all phases to this many steps and run phase 1 only.",
    )
    parser.add_argument(
        "--phase_index",
        type=int,
        default=None,
        help="Run only this phase (0-indexed). Each phase runs as a separate "
             "process to guarantee GPU memory cleanup between phases.",
    )
    args = parser.parse_args()

    config = GRPOTrainingConfig.from_yaml(args.config)
    if args.sft_adapter_path:
        config.sft_adapter_path = args.sft_adapter_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.pilot_steps is not None:
        logger.info("PILOT MODE: running phase 1 only with %d steps", args.pilot_steps)
        config.phases = [CurriculumPhase(
            name="pilot",
            max_hops=1,
            max_steps=args.pilot_steps,
            run_name=f"{config.phases[0].run_name}-pilot",
        )]
        run_curriculum_training(config)
    elif args.phase_index is not None:
        run_single_phase(config, args.phase_index)
    else:
        run_curriculum_training(config)


if __name__ == "__main__":
    main()
