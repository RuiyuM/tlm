#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
import types
import __main__
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LAD diffusion model on geosignal_random_5k for ablation.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/AdaptEval/geosignal_random_5k.json",
        help="Path to geosignal_random_5k json.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        required=True,
        help="Path to initial LAD diffusion checkpoint (.pth).",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="/data/rxm210041/huggingface/Llama-3.2-3B-Instruct",
        help="Base model path used for tokenizer loading.",
    )
    parser.add_argument(
        "--lad_repo",
        type=str,
        default="third_party/lad-code",
        help="Path to LAD repository.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saves/lad-ablation/geosignal_5k_lad_train",
        help="Output directory for checkpoints and final model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--no_eval", action="store_true", help="Disable evaluation during training.")
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


def ensure_lad_importable(lad_repo: Path) -> None:
    if not lad_repo.exists():
        raise FileNotFoundError(f"LAD repo not found: {lad_repo}")
    lad_repo_str = str(lad_repo.resolve())
    if lad_repo_str not in sys.path:
        sys.path.insert(0, lad_repo_str)


def register_lad_pickle_symbols() -> None:
    from configs.model_config import CustomTransformerConfig  # pylint: disable=import-outside-toplevel
    from models.custom_transformer import CustomTransformerModel  # pylint: disable=import-outside-toplevel

    if not hasattr(__main__, "CustomTransformerModel"):
        setattr(__main__, "CustomTransformerModel", CustomTransformerModel)
    if not hasattr(__main__, "CustomTransformerConfig"):
        setattr(__main__, "CustomTransformerConfig", CustomTransformerConfig)


def install_ipython_display_stub() -> None:
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


def _build_prompt(instruction: str, input_text: str, response: str) -> str:
    full_instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        full_instruction = f"{full_instruction}\n{input_text}"
    return f"User: {full_instruction}\nAssistant: {response.strip()}"


def _tokenize_batch(tokenizer, batch: Dict[str, List[Any]], max_length: int) -> Dict[str, List[List[int]]]:
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    for instruction, input_text, response in zip(batch["instruction"], batch["input"], batch["output"]):
        text = _build_prompt(str(instruction), str(input_text), str(response))
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids = token_ids + [pad_token_id] * (max_length - len(token_ids))

        input_ids_list.append(token_ids)
        labels_list.append(token_ids.copy())

    return {"input_ids": input_ids_list, "labels": labels_list}


def load_geosignal_dataset(dataset_path: Path, max_samples: int, seed: int) -> DatasetDict:
    with dataset_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in dataset json: {dataset_path}")

    rows = rows[: max_samples if max_samples > 0 else len(rows)]
    dataset = Dataset.from_list(rows).shuffle(seed=seed)

    split = dataset.train_test_split(test_size=0.04, seed=seed)
    val_test = split["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(
        {
            "train": split["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = get_dist_env()
    set_seed(args.seed + rank)

    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and world_size == 1:
        raise RuntimeError(
            "LAD geosignal training script is currently single-GPU only because the loaded checkpoint "
            "is not compatible with Trainer DataParallel. Please run either with one visible GPU, e.g. "
            "`CUDA_VISIBLE_DEVICES=0 python scripts/lad_train_geosignal.py ...`, or with DDP via torchrun."
        )

    repo_root = Path(__file__).resolve().parents[1]
    lad_repo = (repo_root / args.lad_repo).resolve()
    ensure_lad_importable(lad_repo)
    register_lad_pickle_symbols()

    from data.noise import Noiser  # pylint: disable=import-outside-toplevel
    try:
        from inference.infer import patch_legacy_llama_attention  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as err:
        if err.name == "IPython":
            install_ipython_display_stub()
            from inference.infer import patch_legacy_llama_attention  # pylint: disable=import-outside-toplevel
        else:
            raise
    from training.collator import DiffusionDataCollator  # pylint: disable=import-outside-toplevel

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_geosignal_dataset(Path(args.dataset_path), max_samples=args.max_samples, seed=args.seed)
    tokenized = DatasetDict(
        {
            split: ds.map(
                lambda batch: _tokenize_batch(tokenizer=tokenizer, batch=batch, max_length=args.max_length),
                batched=True,
                batch_size=256,
                num_proc=args.num_proc,
                desc=f"Tokenizing {split}",
            )
            for split, ds in dataset.items()
        }
    )

    noiser = Noiser(tokenizer)
    tokenized = DatasetDict(
        {
            split: ds.map(
                noiser.corrupt_batch,
                batched=True,
                batch_size=256,
                num_proc=args.num_proc,
                desc=f"Noising {split}",
            )
            for split, ds in tokenized.items()
        }
    )

    model = torch.load(args.init_checkpoint, map_location=torch.device("cpu"), weights_only=False)
    model = patch_legacy_llama_attention(model)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")
        model.to(device)
    elif torch.backends.mps.is_available():
        model.to(torch.device("mps"))
    else:
        model.to(torch.device("cpu"))

    eval_strategy = "no" if args.no_eval else "steps"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if not args.no_eval else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        max_grad_norm=0.5,
        fp16=torch.cuda.is_available(),
        local_rank=local_rank if world_size > 1 else -1,
        ddp_find_unused_parameters=False if world_size > 1 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if not args.no_eval else None,
        data_collator=DiffusionDataCollator(),
    )

    train_result = trainer.train()
    trainer.save_state()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if trainer.is_world_process_zero():
        final_ckpt = output_dir / "diffusion-model-geosignal-final.pth"
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save, final_ckpt)
        tokenizer.save_pretrained(str(output_dir / "tokenizer"))

        metrics = dict(train_result.metrics)
        metrics["train_size"] = len(tokenized["train"])
        metrics["valid_size"] = len(tokenized["validation"])
        metrics["test_size"] = len(tokenized["test"])
        with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"[done] Saved final LAD checkpoint to: {final_ckpt}")
        print(f"[done] Saved tokenizer to: {output_dir / 'tokenizer'}")
        print(f"[done] Saved train metrics to: {output_dir / 'train_metrics.json'}")


if __name__ == "__main__":
    main()
