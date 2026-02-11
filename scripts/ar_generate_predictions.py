#!/usr/bin/env python
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run autoregressive inference and export TLM-style generated_predictions.jsonl."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF id or local model path.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset json file.")
    parser.add_argument("--output_path", type=str, required=True, help="Output jsonl path.")
    parser.add_argument("--max_samples", type=int, default=None, help="Use first N samples only.")
    parser.add_argument("--start_index", type=int, default=0, help="Start sample index (inclusive).")
    parser.add_argument("--end_index", type=int, default=None, help="End sample index (exclusive).")
    parser.add_argument(
        "--keep_rank_outputs",
        action="store_true",
        help="Keep per-rank output shards when using torchrun multi-process mode.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Per-rank batch size.")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Stop immediately if one batch fails.")
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
    if "instruction" in sample:
        instruction = str(sample.get("instruction", "")).strip()
        user_input = str(sample.get("input", "")).strip()
        if user_input:
            return instruction + "\n" + user_input
        return instruction

    if "question" in sample:
        return str(sample.get("question", "")).strip()

    return str(sample.get("prompt", "")).strip()


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


def build_chat_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return question


def get_torch_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()

    rank, world_size, local_rank = get_dist_env()
    is_distributed = maybe_init_process_group(world_size)

    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)

    set_seed(args.seed + rank)

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

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rank_output_path = get_rank_output_path(output_path, rank, world_size)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=get_torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    failed = 0
    idx_offset = start_index + shard_start
    do_sample = args.temperature > 0
    progress_desc = "AR inference" if world_size == 1 else f"AR inference (rank {rank}/{world_size})"

    with rank_output_path.open("w", encoding="utf-8") as writer:
        for batch_start in tqdm(range(0, len(samples), args.batch_size), desc=progress_desc):
            batch = samples[batch_start: batch_start + args.batch_size]
            questions = [build_question(sample) for sample in batch]
            labels = [build_label(sample) for sample in batch]
            prompts = [build_chat_prompt(tokenizer, question) for question in questions]
            idx_base = idx_offset + batch_start

            try:
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_length,
                ).to(device)

                generate_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                if do_sample:
                    generate_kwargs["temperature"] = args.temperature
                    generate_kwargs["top_p"] = args.top_p
                    if args.top_k > 0:
                        generate_kwargs["top_k"] = args.top_k

                with torch.no_grad():
                    output_ids = model.generate(**inputs, **generate_kwargs)

                input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
                predictions = []
                for i, seq in enumerate(output_ids):
                    gen_tokens = seq[input_lengths[i]:]
                    predictions.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

            except Exception as err:  # noqa: BLE001
                failed += len(batch)
                if args.strict:
                    raise
                print(f"[warn] batch start {idx_base} failed: {err}")
                predictions = [""] * len(batch)

            for question, label, prediction in zip(questions, labels, predictions):
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
