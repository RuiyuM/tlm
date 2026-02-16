import json
import math
import os
import random
import sys
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from ...extras import logging
from ...extras.constants import DATA_CONFIG


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


@dataclass
class PromptRecord:
    question: str
    label: str
    prompt_ids: List[int]
    answer_start: int
    question_start: int
    question_end: int
    mask_token_id: int
    score_ell: Optional[float] = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _build_question(sample: Dict[str, Any], columns: Dict[str, str]) -> str:
    prompt_key = columns.get("prompt")
    query_key = columns.get("query")

    prompt_candidates = [prompt_key, "instruction", "question", "prompt"]
    query_candidates = [query_key, "input", "query"]

    prompt = ""
    for key in prompt_candidates:
        if key and key in sample and sample[key] is not None:
            prompt = str(sample[key]).strip()
            break

    query = ""
    for key in query_candidates:
        if key and key in sample and sample[key] is not None:
            query = str(sample[key]).strip()
            break

    if prompt and query:
        return f"{prompt}\n{query}"
    return prompt or query


def _build_label(sample: Dict[str, Any], columns: Dict[str, str]) -> str:
    response_key = columns.get("response")
    for key in [response_key, "output", "answers", "response", "label"]:
        if key and key in sample and sample[key] is not None:
            return str(sample[key])
    return ""


def _read_json_or_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if file_path.suffix.lower() == ".jsonl":
        rows = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    raise ValueError(f"Unsupported dataset file content in {file_path}.")


def _resolve_dataset_entry(dataset_name: str, dataset_dir: str) -> Tuple[Path, Dict[str, Any]]:
    dataset_info_path = Path(dataset_dir) / DATA_CONFIG
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)
        if dataset_name in dataset_info and "file_name" in dataset_info[dataset_name]:
            entry = dataset_info[dataset_name]
            return Path(dataset_dir) / entry["file_name"], entry

    # Fallback: treat dataset name as relative file path.
    fallback_path = Path(dataset_dir) / dataset_name
    return fallback_path, {}


def _load_prompt_records(
    dataset_names: Optional[List[str]],
    dataset_dir: str,
    max_samples: Optional[int],
    tokenizer,
    max_length: int,
) -> List[PromptRecord]:
    if not dataset_names:
        return []

    records: List[PromptRecord] = []
    for dataset_name in dataset_names:
        data_path, entry = _resolve_dataset_entry(dataset_name, dataset_dir)
        columns = entry.get("columns", {}) if isinstance(entry, dict) else {}
        rows = _read_json_or_jsonl(data_path)
        for row in rows:
            question = _build_question(row, columns)
            if not question:
                continue
            label = _build_label(row, columns)
            prompt_ids, answer_start, question_start, question_end, mask_token_id = _prepare_prompt_tokens(
                tokenizer=tokenizer,
                question=question,
                max_length=max_length,
            )
            records.append(
                PromptRecord(
                    question=question,
                    label=label,
                    prompt_ids=prompt_ids,
                    answer_start=answer_start,
                    question_start=question_start,
                    question_end=question_end,
                    mask_token_id=mask_token_id,
                )
            )
            if max_samples is not None and len(records) >= max_samples:
                return records

    return records


def _ensure_lad_importable(lad_repo: Path) -> None:
    if not lad_repo.exists():
        raise FileNotFoundError(f"LAD repo not found: {lad_repo}")
    lad_repo_str = str(lad_repo.resolve())
    if lad_repo_str not in sys.path:
        sys.path.insert(0, lad_repo_str)


def _install_ipython_display_stub() -> None:
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


def _maybe_set_local_hf_token(base_model_name: str) -> None:
    if os.getenv("HUGGINGFACE_TOKEN"):
        return
    if Path(base_model_name).exists():
        os.environ["HUGGINGFACE_TOKEN"] = "LOCAL_MODEL"


def _register_lad_pickle_symbols() -> None:
    import __main__  # pylint: disable=import-outside-toplevel
    from configs.model_config import CustomTransformerConfig  # pylint: disable=import-outside-toplevel
    from models.custom_transformer import CustomTransformerModel  # pylint: disable=import-outside-toplevel

    if not hasattr(__main__, "CustomTransformerModel"):
        setattr(__main__, "CustomTransformerModel", CustomTransformerModel)
    if not hasattr(__main__, "CustomTransformerConfig"):
        setattr(__main__, "CustomTransformerConfig", CustomTransformerConfig)


def _build_lad_prompt(question: str) -> str:
    return (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{question.strip()}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def _find_subsequence(input_ids: List[int], marker_ids: List[int]) -> Optional[int]:
    for i in range(len(input_ids) - len(marker_ids) + 1):
        if input_ids[i : i + len(marker_ids)] == marker_ids:
            return i
    return None


def _prepare_prompt_tokens(
    tokenizer,
    question: str,
    max_length: int,
) -> Tuple[List[int], int, int, int, int]:
    prompt = _build_lad_prompt(question)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    assistant_marker_ids = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False)
    marker_pos = _find_subsequence(input_ids, assistant_marker_ids)
    if marker_pos is None:
        raise ValueError("Assistant marker not found in prompt.")

    answer_start = marker_pos + len(assistant_marker_ids)
    question_start = 0
    question_end = answer_start
    user_marker_ids = tokenizer.encode("<|start_header_id|>user<|end_header_id|>\n", add_special_tokens=False)
    if len(user_marker_ids) > 0:
        latest_user_pos: Optional[int] = None
        for idx in range(max(0, marker_pos - len(user_marker_ids) + 1)):
            if input_ids[idx : idx + len(user_marker_ids)] == user_marker_ids:
                latest_user_pos = idx
        if latest_user_pos is not None:
            question_start = latest_user_pos + len(user_marker_ids)
            question_end = marker_pos

    input_ids = input_ids[:max_length]
    seq_len = len(input_ids)
    answer_start = min(answer_start, seq_len)
    question_start = min(max(0, question_start), answer_start)
    question_end = min(max(question_start, question_end), answer_start)
    if question_end <= question_start:
        question_start = 0
        question_end = answer_start

    mask_token_ids = tokenizer.encode("MASK", add_special_tokens=False)
    if len(mask_token_ids) == 0:
        raise ValueError("Tokenizer cannot encode MASK token.")
    if len(mask_token_ids) != 1:
        raise ValueError(
            f"MASK token must map to exactly 1 token id, but got {len(mask_token_ids)}: {mask_token_ids}."
        )

    return input_ids, answer_start, question_start, question_end, mask_token_ids[0]


def _sample_mask_positions(
    eligible_positions: List[int],
    mask_ratio_min: float,
    mask_ratio_max: float,
    full_mask_prob: float,
) -> List[int]:
    min_ratio = min(mask_ratio_min, mask_ratio_max)
    max_ratio = max(mask_ratio_min, mask_ratio_max)
    use_full_mask = random.random() < full_mask_prob
    if use_full_mask:
        return eligible_positions

    ratio = random.uniform(min_ratio, max_ratio)
    num_to_mask = max(1, int(round(len(eligible_positions) * ratio)))
    num_to_mask = min(num_to_mask, len(eligible_positions))
    return random.sample(eligible_positions, num_to_mask)


def _apply_lad_prompt_noising(
    prompt_ids: List[int],
    eligible_positions: List[int],
    mask_token_id: int,
    noise_prob: float,
    full_mask_prob: float,
) -> Tuple[List[int], List[int]]:
    corrupted = prompt_ids.copy()
    clipped_noise = float(_clamp(noise_prob, 0.0, 1.0))

    # Keep full-mask branch to preserve compatibility with existing knobs.
    if random.random() < full_mask_prob:
        for pos in eligible_positions:
            corrupted[pos] = mask_token_id
    else:
        # Random masking branch (aligned with LAD structurally_corrupt style).
        if random.random() < 0.5:
            mask_fraction = random.uniform(0.0, 0.5)
            num_to_mask = int(len(eligible_positions) * mask_fraction)
            if num_to_mask > 0:
                for pos in random.sample(eligible_positions, min(num_to_mask, len(eligible_positions))):
                    corrupted[pos] = mask_token_id

        if len(eligible_positions) > 2:
            # Local adjacent swaps across prompt-token order.
            for idx in range(len(eligible_positions) - 1):
                if random.random() < (clipped_noise / 4.0):
                    left = eligible_positions[idx]
                    right = eligible_positions[idx + 1]
                    corrupted[left], corrupted[right] = corrupted[right], corrupted[left]

            # Token duplication from neighboring prompt tokens.
            for idx in range(len(eligible_positions)):
                if random.random() >= (clipped_noise / 4.0):
                    continue

                tgt = eligible_positions[idx]
                if idx > 0 and idx < len(eligible_positions) - 1:
                    src_idx = idx - 1 if random.random() < 0.5 else idx + 1
                elif idx > 0:
                    src_idx = idx - 1
                elif idx < len(eligible_positions) - 1:
                    src_idx = idx + 1
                else:
                    continue
                src = eligible_positions[src_idx]
                corrupted[tgt] = corrupted[src]

            # Span shift with bounded span length.
            if random.random() < (clipped_noise / 4.0):
                span_len = random.randint(1, min(3, len(eligible_positions)))
                shift = random.randint(1, 4)
                direction = random.choice([-1, 1])
                start_idx = random.randint(0, len(eligible_positions) - span_len)
                if direction < 0:
                    target_idx = max(0, start_idx - shift)
                else:
                    target_idx = min(len(eligible_positions) - span_len, start_idx + shift)

                source_positions = eligible_positions[start_idx : start_idx + span_len]
                target_positions = eligible_positions[target_idx : target_idx + span_len]
                span_values = [corrupted[pos] for pos in source_positions]
                for pos, val in zip(target_positions, span_values):
                    corrupted[pos] = val

    changed_positions = [pos for pos in eligible_positions if corrupted[pos] != prompt_ids[pos]]
    if len(changed_positions) == 0:
        fallback_pos = random.choice(eligible_positions)
        corrupted[fallback_pos] = mask_token_id
        changed_positions = [fallback_pos]

    return corrupted, changed_positions


def _compute_prompt_ell(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: List[int],
    answer_start: int,
    question_start: int,
    question_end: int,
    mask_token_id: int,
    num_masks: int,
    mask_ratio_min: float,
    mask_ratio_max: float,
    full_mask_prob: float,
    require_grad: bool,
    question_only_span: bool,
    noising_mode: str = "mask",
) -> torch.Tensor:
    if num_masks <= 0:
        raise ValueError("num_masks must be positive.")

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if question_only_span:
        span_start = min(max(0, question_start), answer_start)
        span_end = min(max(span_start, question_end), answer_start)
    else:
        span_start = 0
        span_end = answer_start

    eligible_positions = [i for i in range(span_start, span_end) if prompt_ids[i] not in special_ids]
    if len(eligible_positions) == 0 and question_only_span:
        # Fallback to full prompt-side span to avoid dropping malformed records.
        eligible_positions = [i for i in range(answer_start) if prompt_ids[i] not in special_ids]
    if len(eligible_positions) == 0:
        eligible_positions = list(range(answer_start))
    if len(eligible_positions) == 0:
        raise ValueError("No eligible prompt tokens for DPPL masking.")

    device = _get_input_device(model)
    losses: List[torch.Tensor] = []
    context = nullcontext() if require_grad else torch.no_grad()
    with context:
        for _ in range(num_masks):
            if noising_mode == "mask":
                mask_positions = _sample_mask_positions(
                    eligible_positions=eligible_positions,
                    mask_ratio_min=mask_ratio_min,
                    mask_ratio_max=mask_ratio_max,
                    full_mask_prob=full_mask_prob,
                )
                corrupted = prompt_ids.copy()
                for pos in mask_positions:
                    corrupted[pos] = mask_token_id
            elif noising_mode == "lad_prompt":
                noise_prob = random.uniform(min(mask_ratio_min, mask_ratio_max), max(mask_ratio_min, mask_ratio_max))
                corrupted, mask_positions = _apply_lad_prompt_noising(
                    prompt_ids=prompt_ids,
                    eligible_positions=eligible_positions,
                    mask_token_id=mask_token_id,
                    noise_prob=noise_prob,
                    full_mask_prob=full_mask_prob,
                )
            else:
                raise ValueError(f"Unknown prompt noising mode: {noising_mode}")

            input_tensor = torch.tensor([corrupted], dtype=torch.long, device=device)
            logits = _forward_diffusion_logits(model, input_tensor)[0]  # [seq_len, vocab]

            pos_tensor = torch.tensor(mask_positions, dtype=torch.long, device=device)
            target_tensor = torch.tensor([prompt_ids[p] for p in mask_positions], dtype=torch.long, device=device)
            masked_logits = logits.index_select(0, pos_tensor)
            nll = F.cross_entropy(masked_logits, target_tensor, reduction="mean")
            losses.append(nll)

    return torch.stack(losses).mean()


def _forward_diffusion_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "llama"):
        raise ValueError("LAD model does not expose `llama` backbone.")

    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    masking_type = getattr(model.config, "masking_type", "bidirectional")

    if masking_type == "bidirectional":
        base_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    elif masking_type == "bidirectional_masked":
        base_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        base_mask.fill_diagonal_(False)
    elif masking_type == "unidirectional":
        base_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    else:
        raise ValueError(f"Unknown masking_type: {masking_type}")

    attention_mask = base_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len).to(dtype=torch.float32)
    amp_context = model._get_autocast_context(device) if hasattr(model, "_get_autocast_context") else nullcontext()
    with amp_context:
        outputs = model.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )

    vocab_size = getattr(model.config, "vocab_size", outputs.logits.size(-1))
    return outputs.logits[:, :, :vocab_size]


def _get_input_device(model: torch.nn.Module) -> torch.device:
    candidates = []
    if hasattr(model, "llama"):
        candidates.append(model.llama)
    dispatch_target = _get_dispatch_target(model.llama) if hasattr(model, "llama") else None
    if dispatch_target is not None:
        candidates.append(dispatch_target)

    for module in candidates:
        if hasattr(module, "hf_device_map"):
            device_map = getattr(module, "hf_device_map")
            if isinstance(device_map, dict):
                for device_name in device_map.values():
                    if isinstance(device_name, int):
                        return torch.device(f"cuda:{device_name}")
                    if isinstance(device_name, str) and device_name.startswith("cuda"):
                        return torch.device(device_name)
    return next(model.parameters()).device


def _get_dispatch_target(llama_module: torch.nn.Module) -> torch.nn.Module:
    # For PEFT, dispatch the wrapped base model instead of the outer wrapper.
    if hasattr(llama_module, "base_model") and hasattr(llama_module.base_model, "model"):
        return llama_module.base_model.model
    return llama_module


def _maybe_dispatch_model_parallel(model: torch.nn.Module) -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return False
    if not hasattr(model, "llama"):
        return False

    try:
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory
    except Exception as err:  # noqa: BLE001
        logger.warning_rank0(f"Cannot import accelerate model dispatch utils: {err}")
        return False

    target = _get_dispatch_target(model.llama)
    no_split_module_classes = getattr(target, "_no_split_modules", None)
    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus))
    if num_gpus < 2:
        return False

    try:
        max_memory = get_balanced_memory(
            target,
            max_memory=None,
            no_split_module_classes=no_split_module_classes,
            dtype=torch.float16,
        )
        device_map = infer_auto_device_map(
            target,
            dtype=torch.float16,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
        )

        used_cuda_devices = set()
        for dev in device_map.values():
            if isinstance(dev, int):
                used_cuda_devices.add(dev)
            elif isinstance(dev, str) and dev.startswith("cuda"):
                try:
                    used_cuda_devices.add(int(dev.split(":")[1]))
                except Exception:
                    pass

        if len(used_cuda_devices) < 2:
            # Force split by capping each GPU memory budget.
            forced_max_memory = {}
            for i in gpu_ids:
                total_gib = int(torch.cuda.get_device_properties(i).total_memory / (1024**3))
                forced_max_memory[i] = f"{max(1, total_gib // 2)}GiB"
            forced_max_memory["cpu"] = "64GiB"
            device_map = infer_auto_device_map(
                target,
                dtype=torch.float16,
                max_memory=forced_max_memory,
                no_split_module_classes=no_split_module_classes,
            )

        dispatch_model(target, device_map=device_map)

        # Mirror map for easier logging/introspection.
        setattr(target, "hf_device_map", device_map)
        setattr(model.llama, "hf_device_map", device_map)
        logger.info_rank0(f"Enabled model parallel for LAD backbone: {device_map}")
        return True
    except Exception as err:  # noqa: BLE001
        logger.warning_rank0(f"Model parallel dispatch failed, fallback to single GPU: {err}")
        return False


def _load_lad_model_for_ttl(
    checkpoint_path: str,
    base_model_name: str,
    lad_infer,
) -> Tuple[torch.nn.Module, Any]:
    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=True,
        token=hf_token,
        torch_dtype=torch.float32,
    )

    model = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    model = lad_infer.disable_dropout(model)
    model = lad_infer.patch_legacy_llama_attention(model)

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        model.to(torch.device("mps"))
    else:
        model.to(torch.device("cpu"))

    model.eval()
    return model, tokenizer


def _merge_and_attach_ttl_lora(model: torch.nn.Module, finetuning_args: "FinetuningArguments") -> List[torch.nn.Parameter]:
    if not hasattr(model, "llama"):
        raise ValueError("LAD model does not expose `llama` backbone.")

    if finetuning_args.merge_lad_lora and hasattr(model.llama, "merge_and_unload"):
        model.llama = model.llama.merge_and_unload()
        logger.info_rank0("Merged LAD LoRA into backbone.")

    from peft import LoraConfig, get_peft_model

    target_modules: Any = finetuning_args.lora_target
    if isinstance(target_modules, list) and len(target_modules) == 1 and target_modules[0] == "all":
        target_modules = "all-linear"

    lora_config = LoraConfig(
        r=finetuning_args.lora_rank,
        lora_alpha=finetuning_args.lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.llama = get_peft_model(model.llama, lora_config, adapter_name="ttl")
    if hasattr(model.llama, "set_adapter"):
        model.llama.set_adapter("ttl")

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable LoRA parameters found for TTL.")

    trainable_count = sum(p.numel() for p in trainable_params)
    logger.info_rank0(f"TTL LoRA trainable parameters: {trainable_count}")
    return trainable_params


def _select_offline_samples(
    records: List[PromptRecord],
    model: torch.nn.Module,
    tokenizer,
    finetuning_args: "FinetuningArguments",
) -> Tuple[List[PromptRecord], float]:
    model.eval()
    logger.info_rank0(f"Scoring {len(records)} prompts with prompt-DPPL ...")
    scores: List[float] = []
    for record in tqdm(records, desc="Scoring prompts", leave=False):
        ell = _compute_prompt_ell(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=record.prompt_ids,
            answer_start=record.answer_start,
            question_start=record.question_start,
            question_end=record.question_end,
            mask_token_id=record.mask_token_id,
            num_masks=finetuning_args.prompt_dppl_num_masks,
            mask_ratio_min=finetuning_args.prompt_dppl_mask_ratio_min,
            mask_ratio_max=finetuning_args.prompt_dppl_mask_ratio_max,
            full_mask_prob=finetuning_args.prompt_dppl_full_mask_prob,
            require_grad=False,
            question_only_span=finetuning_args.prompt_dppl_question_only_span,
            noising_mode=finetuning_args.prompt_dppl_score_noising,
        )
        score = float(ell.item())
        record.score_ell = score
        scores.append(score)

    if len(scores) == 0:
        raise ValueError("No prompt could be scored.")

    if finetuning_args.prompt_dppl_top_percent > 0:
        k = max(1, int(math.ceil(len(scores) * finetuning_args.prompt_dppl_top_percent)))
        threshold = sorted(scores)[-k]
        selected = [record for record in records if (record.score_ell is not None and record.score_ell >= threshold)]
    else:
        threshold = finetuning_args.threshold
        selected = [record for record in records if (record.score_ell is not None and record.score_ell > threshold)]

    if len(selected) == 0:
        best = max(records, key=lambda x: x.score_ell if x.score_ell is not None else -1e9)
        selected = [best]
        logger.warning_rank0("No sample passed threshold. Fallback to top-1 prompt by ell.")

    logger.info_rank0(
        f"Prompt-DPPL selection finished: {len(selected)}/{len(records)} selected. Threshold={threshold:.4f}"
    )
    logger.info_rank0(
        f"Scoring stats: mean={float(sum(scores) / len(scores)):.4f}, "
        f"min={float(min(scores)):.4f}, max={float(max(scores)):.4f}"
    )
    return selected, threshold


def _train_ttl_lora(
    selected_records: List[PromptRecord],
    threshold: float,
    model: torch.nn.Module,
    tokenizer,
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> Dict[str, float]:
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=training_args.learning_rate)

    grad_accum = max(1, training_args.gradient_accumulation_steps)
    max_steps = training_args.max_steps if training_args.max_steps and training_args.max_steps > 0 else None
    num_epochs = max(1, int(math.ceil(training_args.num_train_epochs)))

    global_step = 0
    micro_step = 0
    mean_loss = 0.0
    mean_ell = 0.0
    denom = 0
    threshold_tensor = torch.tensor(threshold, device=_get_input_device(model), dtype=torch.float32)

    rng = random.Random(training_args.seed)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(num_epochs):
        if max_steps is not None and global_step >= max_steps:
            break

        shuffled = selected_records.copy()
        rng.shuffle(shuffled)
        for record in tqdm(shuffled, desc=f"TTL train epoch {epoch + 1}/{num_epochs}", leave=False):
            ell = _compute_prompt_ell(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=record.prompt_ids,
                answer_start=record.answer_start,
                question_start=record.question_start,
                question_end=record.question_end,
                mask_token_id=record.mask_token_id,
                num_masks=finetuning_args.prompt_dppl_train_num_masks,
                mask_ratio_min=finetuning_args.prompt_dppl_mask_ratio_min,
                mask_ratio_max=finetuning_args.prompt_dppl_mask_ratio_max,
                full_mask_prob=finetuning_args.prompt_dppl_full_mask_prob,
                require_grad=True,
                question_only_span=finetuning_args.prompt_dppl_question_only_span,
                noising_mode=finetuning_args.prompt_dppl_train_noising,
            )

            if finetuning_args.diffusion_use_weighting:
                coeff = finetuning_args.lamb * torch.exp(ell.detach() - threshold_tensor)
                loss = coeff * ell
            else:
                loss = ell

            (loss / grad_accum).backward()
            micro_step += 1
            mean_loss += float(loss.detach().item())
            mean_ell += float(ell.detach().item())
            denom += 1

            if micro_step % grad_accum == 0:
                if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, training_args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % max(1, training_args.logging_steps) == 0:
                    logger.info_rank0(
                        f"Diffusion TTL step {global_step}: "
                        f"loss={mean_loss / max(1, denom):.6f}, "
                        f"ell={mean_ell / max(1, denom):.6f}"
                    )
                    mean_loss = 0.0
                    mean_ell = 0.0
                    denom = 0

                if max_steps is not None and global_step >= max_steps:
                    break

        if micro_step % grad_accum != 0:
            if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, training_args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    return {"global_step": float(global_step)}


def _snapshot_trainable_lora_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    snapshot: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            snapshot[name] = param.detach().to(dtype=torch.float32, device="cpu").clone()
    return snapshot


def _compute_lora_delta_l2(
    model: torch.nn.Module, snapshot: Dict[str, torch.Tensor]
) -> Tuple[float, float]:
    sq_sum = 0.0
    tracked_numel = 0
    for name, param in model.named_parameters():
        if name not in snapshot:
            continue
        before = snapshot[name]
        after = param.detach().to(dtype=torch.float32, device="cpu")
        diff = after - before
        sq_sum += float(torch.sum(diff * diff).item())
        tracked_numel += diff.numel()

    if tracked_numel == 0:
        return 0.0, 0.0
    return float(math.sqrt(sq_sum)), float(tracked_numel)


def _compute_mean_ell_on_records(
    records: List[PromptRecord],
    model: torch.nn.Module,
    tokenizer,
    finetuning_args: "FinetuningArguments",
    num_masks: int,
    seed: int,
    desc: str,
    noising_mode: str,
) -> float:
    if len(records) == 0:
        return 0.0

    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    total = 0.0
    try:
        model.eval()
        for record in tqdm(records, desc=desc, leave=False):
            ell = _compute_prompt_ell(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=record.prompt_ids,
                answer_start=record.answer_start,
                question_start=record.question_start,
                question_end=record.question_end,
                mask_token_id=record.mask_token_id,
                num_masks=num_masks,
                mask_ratio_min=finetuning_args.prompt_dppl_mask_ratio_min,
                mask_ratio_max=finetuning_args.prompt_dppl_mask_ratio_max,
                full_mask_prob=finetuning_args.prompt_dppl_full_mask_prob,
                require_grad=False,
                question_only_span=finetuning_args.prompt_dppl_question_only_span,
                noising_mode=noising_mode,
            )
            total += float(ell.item())
    finally:
        if was_training:
            model.train()
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    return total / len(records)


def _maybe_enable_input_require_grads(model: torch.nn.Module) -> None:
    if hasattr(model, "llama") and hasattr(model.llama, "enable_input_require_grads"):
        try:
            model.llama.enable_input_require_grads()
            logger.info_rank0("Enabled input require grads for gradient checkpointing.")
        except Exception as err:  # noqa: BLE001
            logger.warning_rank0(f"Cannot enable input require grads: {err}")


def _save_outputs(
    model: torch.nn.Module,
    tokenizer,
    training_args: "Seq2SeqTrainingArguments",
    stats: Dict[str, float],
) -> None:
    def _sanitize_jsonable(obj):
        if isinstance(obj, dict):
            return {k: _sanitize_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_jsonable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_sanitize_jsonable(v) for v in obj)
        if isinstance(obj, torch.dtype):
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        return obj

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "llama") and hasattr(model.llama, "save_pretrained"):
        model.llama.save_pretrained(str(output_dir))
    try:
        tokenizer.save_pretrained(str(output_dir))
    except TypeError as err:
        if "not JSON serializable" not in str(err):
            raise
        logger.warning_rank0(f"Tokenizer save hit non-serializable object, retrying with sanitized kwargs: {err}")
        if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
            tokenizer.init_kwargs = _sanitize_jsonable(tokenizer.init_kwargs)
        if hasattr(tokenizer, "init_inputs") and isinstance(tokenizer.init_inputs, (list, tuple)):
            tokenizer.init_inputs = _sanitize_jsonable(list(tokenizer.init_inputs))
        tokenizer.save_pretrained(str(output_dir))

    stats_path = output_dir / "diffusion_ttl_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info_rank0(f"Saved diffusion TTL artifacts to {output_dir}")


def _run_prediction(
    eval_records: List[PromptRecord],
    model: torch.nn.Module,
    tokenizer,
    generating_args: "GeneratingArguments",
    finetuning_args: "FinetuningArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    lad_infer,
) -> None:
    if len(eval_records) == 0:
        logger.warning_rank0("No eval records for diffusion prediction.")
        return

    model.eval()
    predict_output_dir = (
        Path(training_args.output_dir)
        / f"predict-temperature_{generating_args.temperature}-max_new_tokens_{generating_args.max_new_tokens}"
    )
    predict_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = predict_output_dir / "generated_predictions.jsonl"

    do_sample = bool(generating_args.do_sample)
    if do_sample:
        temperature = max(float(generating_args.temperature), 1e-5)
        top_k = int(generating_args.top_k)
        top_p = float(generating_args.top_p)
    else:
        # LAD diffusion decode is sampling-based. Use top_k=1 at normal temperature for stable greedy-like behavior.
        temperature = 1.0
        top_k = 1
        top_p = 1.0

    max_length = finetuning_args.diffusion_max_length if finetuning_args.diffusion_max_length > 0 else data_args.cutoff_len
    add_tokens = max(1, int(generating_args.max_new_tokens))
    base_seed = int(getattr(training_args, "seed", 42))

    with output_file.open("w", encoding="utf-8") as writer:
        for sample_idx, record in enumerate(tqdm(eval_records, desc="Diffusion predict", leave=False)):
            sample_seed = base_seed + sample_idx
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2**32 - 1))
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            if hasattr(lad_infer, "rng"):
                lad_infer.rng = np.random.default_rng(sample_seed)

            prediction = lad_infer.generate_answer(
                question=record.question,
                model=model,
                tokenizer=tokenizer,
                max_it=finetuning_args.diffusion_max_it,
                noise_start=finetuning_args.diffusion_noise_start,
                noising_sharpness=finetuning_args.diffusion_noising_sharpness,
                max_length=max_length,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                eos_boost=0.0,
                add_tokens=add_tokens,
            )
            writer.write(
                json.dumps(
                    {"prompt": record.question, "label": record.label, "predict": prediction},
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info_rank0(f"Saved diffusion prediction results to {output_file}")


def _load_ttl_lora_for_infer(model: torch.nn.Module, adapter_dir: str) -> bool:
    if not hasattr(model, "llama"):
        return False
    adapter_path = Path(adapter_dir)
    if not (adapter_path / "adapter_config.json").exists():
        return False

    try:
        from peft import PeftModel

        model.llama = PeftModel.from_pretrained(model.llama, str(adapter_path), is_trainable=False)
        logger.info_rank0(f"Loaded trained TTL LoRA from {adapter_path}")
        return True
    except Exception as err:  # noqa: BLE001
        logger.warning_rank0(f"Cannot load trained TTL LoRA from {adapter_path}: {err}")
        return False


def run_ttl_diffusion(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
) -> None:
    del callbacks  # Not used in manual diffusion loop.

    if training_args.world_size > 1:
        raise ValueError("Diffusion TTL backend currently supports single-process training only.")

    if finetuning_args.setting != "offline_ttl":
        raise ValueError("Diffusion TTL backend currently supports offline_ttl only.")

    repo_root = Path(os.getcwd())
    lad_repo = (repo_root / finetuning_args.lad_repo).resolve()
    _ensure_lad_importable(lad_repo)
    _maybe_set_local_hf_token(model_args.model_name_or_path)

    try:
        import inference.infer as lad_infer  # pylint: disable=import-error
    except ModuleNotFoundError as err:
        if err.name == "IPython":
            _install_ipython_display_stub()
            from inference.infer import load_trained_model  # pylint: disable=import-error
            import inference.infer as lad_infer  # pylint: disable=import-error
        else:
            raise

    _register_lad_pickle_symbols()
    lad_infer.display_diffusion_output = lambda *x, **y: None

    model, tokenizer = _load_lad_model_for_ttl(
        checkpoint_path=finetuning_args.lad_checkpoint,
        base_model_name=model_args.model_name_or_path,
        lad_infer=lad_infer,
    )
    if training_args.do_train:
        _merge_and_attach_ttl_lora(model, finetuning_args)
    else:
        if finetuning_args.merge_lad_lora and hasattr(model, "llama") and hasattr(model.llama, "merge_and_unload"):
            model.llama = model.llama.merge_and_unload()
            logger.info_rank0("Merged LAD LoRA into backbone (infer mode).")
        loaded = _load_ttl_lora_for_infer(model, training_args.output_dir)
        if not loaded:
            raise FileNotFoundError(
                "No trained TTL LoRA found in output_dir for prediction-only run. "
                "Run with do_train=true first or point output_dir to a trained adapter directory."
            )
    if finetuning_args.diffusion_model_parallel:
        _maybe_dispatch_model_parallel(model)
    if hasattr(model, "llama") and getattr(training_args, "gradient_checkpointing", False):
        try:
            model.llama.gradient_checkpointing_enable()
            logger.info_rank0("Enabled gradient checkpointing for diffusion TTL.")
            _maybe_enable_input_require_grads(model)
        except Exception as err:  # noqa: BLE001
            logger.warning_rank0(f"Cannot enable gradient checkpointing: {err}")

    max_length = finetuning_args.diffusion_max_length if finetuning_args.diffusion_max_length > 0 else data_args.cutoff_len
    eval_records = _load_prompt_records(
        dataset_names=data_args.eval_dataset if data_args.eval_dataset is not None else data_args.dataset,
        dataset_dir=data_args.dataset_dir,
        max_samples=data_args.max_samples,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    stats: Dict[str, float] = {}
    if training_args.do_train:
        train_records = _load_prompt_records(
            dataset_names=data_args.dataset,
            dataset_dir=data_args.dataset_dir,
            max_samples=data_args.max_samples,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        if len(train_records) == 0:
            raise ValueError("No valid training prompts found for diffusion TTL.")

        logger.info_rank0(f"Loaded {len(train_records)} train prompts and {len(eval_records)} eval prompts.")
        selected_records, threshold = _select_offline_samples(
            records=train_records,
            model=model,
            tokenizer=tokenizer,
            finetuning_args=finetuning_args,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stats = {"selected_count": float(len(selected_records)), "threshold": float(threshold)}
        ell_eval_seed = int(getattr(training_args, "seed", 42))
        ell_eval_num_masks = max(1, finetuning_args.prompt_dppl_num_masks)
        mean_ell_before = _compute_mean_ell_on_records(
            records=selected_records,
            model=model,
            tokenizer=tokenizer,
            finetuning_args=finetuning_args,
            num_masks=ell_eval_num_masks,
            seed=ell_eval_seed,
            desc="Mean ell(before)",
            noising_mode=finetuning_args.prompt_dppl_score_noising,
        )
        lora_snapshot = _snapshot_trainable_lora_params(model)

        train_stats = _train_ttl_lora(
            selected_records=selected_records,
            threshold=threshold,
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            finetuning_args=finetuning_args,
        )
        mean_ell_after = _compute_mean_ell_on_records(
            records=selected_records,
            model=model,
            tokenizer=tokenizer,
            finetuning_args=finetuning_args,
            num_masks=ell_eval_num_masks,
            seed=ell_eval_seed,
            desc="Mean ell(after)",
            noising_mode=finetuning_args.prompt_dppl_score_noising,
        )
        lora_delta_l2, lora_tracked_numel = _compute_lora_delta_l2(model=model, snapshot=lora_snapshot)

        stats.update(train_stats)
        stats.update(
            {
                "mean_ell_before": float(mean_ell_before),
                "mean_ell_after": float(mean_ell_after),
                "mean_ell_delta": float(mean_ell_after - mean_ell_before),
                "lora_delta_l2": float(lora_delta_l2),
                "lora_delta_l2_per_param": float(lora_delta_l2 / max(1.0, lora_tracked_numel)),
                "lora_tracked_numel": float(lora_tracked_numel),
            }
        )
        logger.info_rank0(
            "TTL diagnostics: "
            f"mean_ell_before={mean_ell_before:.6f}, "
            f"mean_ell_after={mean_ell_after:.6f}, "
            f"delta={mean_ell_after - mean_ell_before:.6f}, "
            f"lora_delta_l2={lora_delta_l2:.6f}"
        )
        _save_outputs(model=model, tokenizer=tokenizer, training_args=training_args, stats=stats)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info_rank0(f"Loaded {len(eval_records)} eval prompts for prediction-only run.")

    if training_args.do_predict:
        _run_prediction(
            eval_records=eval_records,
            model=model,
            tokenizer=tokenizer,
            generating_args=generating_args,
            finetuning_args=finetuning_args,
            data_args=data_args,
            training_args=training_args,
            lad_infer=lad_infer,
        )
