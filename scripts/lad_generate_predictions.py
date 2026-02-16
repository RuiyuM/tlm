#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LAD diffusion inference and export TLM-style generated_predictions.jsonl."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LAD .pth checkpoint.")
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="HF id or local path for the base tokenizer/model family used by LAD.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/AdaptEval/geosignal_random_5k.json",
        help="Path to AdaptEval json file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="saves/baseline/lad_llama32_3b/geography/generated_predictions.jsonl",
        help="Output jsonl path compatible with scripts/eval/eval_similarity.py.",
    )
    parser.add_argument(
        "--lad_repo",
        type=str,
        default="third_party/lad-code",
        help="Path to cloned LAD repository.",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Use first N samples only.")
    parser.add_argument("--start_index", type=int, default=0, help="Start sample index (inclusive).")
    parser.add_argument("--end_index", type=int, default=None, help="End sample index (exclusive).")
    parser.add_argument(
        "--keep_rank_outputs",
        action="store_true",
        help="Keep per-rank output shards when using torchrun multi-process mode.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_it", type=int, default=16)
    parser.add_argument("--noise_start", type=float, default=0.5)
    parser.add_argument("--noising_sharpness", type=float, default=5.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eos_boost", type=float, default=0.0)
    parser.add_argument("--add_tokens", type=int, default=256)

    parser.add_argument(
        "--keep_visualization",
        action="store_true",
        help="Keep notebook-style per-step visualization output.",
    )
    parser.add_argument("--strict", action="store_true", help="Stop immediately if one sample fails.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dist_env() -> tuple[int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def maybe_init_process_group(world_size: int) -> bool:
    if world_size <= 1 or not dist.is_available():
        return False
    if dist.is_initialized():
        return True
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True


def maybe_barrier(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.barrier()


def get_rank_output_path(output_path: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return output_path
    return output_path.with_name(f"{output_path.stem}.rank{rank}{output_path.suffix}")


def merge_rank_outputs(output_path: Path, world_size: int, keep_rank_outputs: bool) -> None:
    part_paths = [get_rank_output_path(output_path, rank, world_size) for rank in range(world_size)]
    for part_path in part_paths:
        if not part_path.exists():
            raise FileNotFoundError(f"Missing rank output: {part_path}")

    with output_path.open("w", encoding="utf-8") as merged_writer:
        for part_path in part_paths:
            with part_path.open("r", encoding="utf-8") as part_reader:
                for line in part_reader:
                    merged_writer.write(line)

    if not keep_rank_outputs:
        for part_path in part_paths:
            part_path.unlink()


def build_question(sample: dict) -> str:
    instruction = str(sample.get("instruction", "")).strip()
    user_input = str(sample.get("input", "")).strip()
    if instruction:
        if user_input:
            return instruction + "\n" + user_input
        return instruction

    question = str(sample.get("question", "")).strip()
    if question:
        return question

    prompt = str(sample.get("prompt", "")).strip()
    if prompt:
        return prompt

    return ""


def build_label(sample: dict) -> str:
    if "output" in sample:
        return str(sample.get("output", ""))
    if "answers" in sample:
        return str(sample.get("answers", ""))
    if "response" in sample:
        return str(sample.get("response", ""))
    if "label" in sample:
        return str(sample.get("label", ""))
    return ""


def ensure_lad_importable(lad_repo: Path) -> None:
    if not lad_repo.exists():
        raise FileNotFoundError(f"LAD repo not found: {lad_repo}")
    lad_repo_str = str(lad_repo.resolve())
    if lad_repo_str not in sys.path:
        sys.path.insert(0, lad_repo_str)


def install_ipython_display_stub() -> None:
    """
    Provide a minimal IPython.display stub so LAD inference can import in non-notebook envs.
    """
    if "IPython.display" in sys.modules:
        return

    ipython_mod = types.ModuleType("IPython")
    ipython_display_mod = types.ModuleType("IPython.display")

    def _noop(*_args, **_kwargs):
        return None

    ipython_display_mod.display = _noop
    ipython_display_mod.clear_output = _noop
    ipython_display_mod.HTML = lambda x=None, *a, **k: x
    ipython_display_mod.Markdown = lambda x=None, *a, **k: x

    ipython_mod.display = ipython_display_mod
    sys.modules["IPython"] = ipython_mod
    sys.modules["IPython.display"] = ipython_display_mod


def maybe_set_local_hf_token(base_model_name: str) -> None:
    if os.getenv("HUGGINGFACE_TOKEN"):
        return
    if Path(base_model_name).exists():
        # LAD loader prompts interactively if HUGGINGFACE_TOKEN is missing.
        # For local model paths, any non-empty value bypasses the prompt.
        os.environ["HUGGINGFACE_TOKEN"] = "LOCAL_MODEL"


def register_lad_pickle_symbols() -> None:
    """
    Some LAD checkpoints were saved with classes bound to `__main__`.
    Register aliases so torch.load can deserialize them from this entrypoint.
    """
    import __main__  # pylint: disable=import-outside-toplevel
    from configs.model_config import CustomTransformerConfig  # pylint: disable=import-outside-toplevel
    from models.custom_transformer import CustomTransformerModel  # pylint: disable=import-outside-toplevel

    if not hasattr(__main__, "CustomTransformerModel"):
        setattr(__main__, "CustomTransformerModel", CustomTransformerModel)
    if not hasattr(__main__, "CustomTransformerConfig"):
        setattr(__main__, "CustomTransformerConfig", CustomTransformerConfig)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = get_dist_env()
    is_distributed = maybe_init_process_group(world_size)

    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)

    # Make stochastic decoding reproducible yet distinct across ranks.
    set_seed(args.seed + rank)

    repo_root = Path(__file__).resolve().parents[1]
    lad_repo = (repo_root / args.lad_repo).resolve()
    ensure_lad_importable(lad_repo)

    maybe_set_local_hf_token(args.base_model_name)

    try:
        from inference.infer import load_trained_model, generate_answer  # pylint: disable=import-error
        import inference.infer as lad_infer  # pylint: disable=import-error
    except ModuleNotFoundError as err:
        if err.name == "IPython":
            install_ipython_display_stub()
            from inference.infer import load_trained_model, generate_answer  # pylint: disable=import-error
            import inference.infer as lad_infer  # pylint: disable=import-error
        else:
            raise

    register_lad_pickle_symbols()

    if not args.keep_visualization:
        lad_infer.display_diffusion_output = lambda *x, **y: None

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    total_samples = len(samples)
    start_index = max(0, args.start_index)
    end_index = total_samples if args.end_index is None else min(args.end_index, total_samples)
    if start_index >= end_index:
        raise ValueError(
            f"Invalid slice: start_index={start_index}, end_index={end_index}, total={total_samples}"
        )
    samples = samples[start_index:end_index]

    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    selected_total = len(samples)
    shard_start = 0
    if world_size > 1:
        shard_start = (selected_total * rank) // world_size
        shard_end = (selected_total * (rank + 1)) // world_size
        samples = samples[shard_start:shard_end]

    model, tokenizer = load_trained_model(args.checkpoint, base_model_name=args.base_model_name)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rank_output_path = get_rank_output_path(output_path, rank, world_size)

    failed = 0
    idx_offset = start_index + shard_start
    progress_desc = "LAD inference" if world_size == 1 else f"LAD inference (rank {rank}/{world_size})"
    with rank_output_path.open("w", encoding="utf-8") as writer:
        for idx, sample in enumerate(tqdm(samples, desc=progress_desc), start=idx_offset):
            question = build_question(sample)
            label = build_label(sample)
            try:
                prediction = generate_answer(
                    question=question,
                    model=model,
                    tokenizer=tokenizer,
                    max_it=args.max_it,
                    noise_start=args.noise_start,
                    noising_sharpness=args.noising_sharpness,
                    max_length=args.max_length,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    eos_boost=args.eos_boost,
                    add_tokens=args.add_tokens,
                )
            except Exception as err:  # noqa: BLE001
                failed += 1
                if args.strict:
                    raise
                print(f"[warn] sample {idx} failed: {err}")
                prediction = ""

            writer.write(
                json.dumps({"prompt": question, "label": label, "predict": prediction}, ensure_ascii=False) + "\n"
            )

    print(f"[rank {rank}] Saved {len(samples)} predictions to: {rank_output_path}")
    print(f"[rank {rank}] Failed samples: {failed}")

    if world_size > 1:
        maybe_barrier(is_distributed)
        if rank == 0:
            merge_rank_outputs(output_path, world_size, args.keep_rank_outputs)
            print(f"Merged {world_size} rank outputs to: {output_path}")
        maybe_barrier(is_distributed)

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
