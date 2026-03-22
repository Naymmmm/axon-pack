from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
import struct
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


SAFE_TENSOR_DTYPES = {
    "F16": "fp16",
    "BF16": "bf16",
    "F32": "fp32",
}

GGUF_TENSOR_TYPES = {
    0: "fp32",
    1: "fp16",
    30: "bf16",
}

CORE_ASSET_FILES = {
    "config.json",
    "generation_config.json",
}

TOKENIZER_ASSET_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "chat_template.jinja",
}

PROCESSOR_ASSET_FILES = {
    "processor_config.json",
    "preprocessor_config.json",
    "feature_extractor_config.json",
    "image_processor_config.json",
    "video_preprocessor_config.json",
}

_LAYER_RE = re.compile(r"(?:layers|h|block|blk)\.(\d+)")
_SPLIT_EXPERT_RE = re.compile(r"((?:^|[._])experts?\.)\d+((?:[._]|$))", re.IGNORECASE)


@dataclass
class TensorSlice:
    name: str
    shape: list[int]
    dtype: str
    source_path: Path
    source_offset: int
    length: int


@dataclass
class PackOptions:
    quantization: str
    group_size: int | None
    outlier_sigma: float
    enable_vq: bool
    boot_cutoff_layers: int | None
    expert_dedup_threshold: float | None
    jobs: int | None
    prefer_gpu: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "tensor"


def _format_bytes(value: int) -> str:
    gib = 1024 * 1024 * 1024
    mib = 1024 * 1024
    kib = 1024
    if value >= gib:
        return f"{value / gib:.2f} GiB"
    if value >= mib:
        return f"{value / mib:.1f} MiB"
    if value >= kib:
        return f"{value / kib:.1f} KiB"
    return f"{value} B"


def _default_parallel_jobs(total_items: int) -> int:
    if total_items <= 1:
        return 1
    cpu_count = os.cpu_count() or 4
    return max(1, min(total_items, cpu_count, 8))


def _effective_pack_jobs(total_tensors: int, options: PackOptions) -> int:
    requested = options.jobs or _default_parallel_jobs(total_tensors)
    if requested < 1:
        return 1
    if options.prefer_gpu and _torch_cuda_available():
        return 1
    return min(total_tensors, requested)


def _torch_cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _gpu_quantization_enabled(options: PackOptions, tensor: TensorSlice, weights: np.ndarray) -> bool:
    if not options.prefer_gpu or not _torch_cuda_available():
        return False
    if tensor.dtype not in {"fp16", "bf16", "fp32"}:
        return False
    return weights.size >= 1_000_000


def _is_tokenizer_asset(name: str) -> bool:
    lower = name.lower()
    return (
        lower in TOKENIZER_ASSET_FILES
        or lower.startswith("tokenizer.")
        or lower.endswith(".tiktoken")
    )


def _is_processor_asset(name: str) -> bool:
    return name.lower() in PROCESSOR_ASSET_FILES


def _hf_index_path(model_dir: Path) -> Path | None:
    index_paths = sorted(model_dir.glob("*.safetensors.index.json"))
    if not index_paths:
        return None
    preferred = model_dir / "model.safetensors.index.json"
    if preferred in index_paths:
        return preferred
    return index_paths[0]


def _hf_shard_paths(model_dir: Path) -> list[Path]:
    index_path = _hf_index_path(model_dir)
    if index_path is None:
        shard_paths = sorted(model_dir.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"no .safetensors files found in {model_dir}")
        return shard_paths

    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index in {index_path}")

    shard_paths: list[Path] = []
    for shard_name in sorted({str(value) for value in weight_map.values()}):
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"safetensors index {index_path.name} references missing shard {shard_name}"
            )
        shard_paths.append(shard_path)
    return shard_paths


def _asset_inventory(model_dir: Path) -> tuple[list[dict[str, str]], list[str], str | None, str | None]:
    assets: list[dict[str, str]] = []
    tokenizer_files: list[str] = []
    config_file: str | None = None
    generation_config_file: str | None = None

    for source in sorted(model_dir.iterdir(), key=lambda path: path.name):
        if not source.is_file():
            continue
        name = source.name
        lower = name.lower()
        if lower in CORE_ASSET_FILES:
            assets.append({"source": str(source), "dest": name})
            if lower == "config.json":
                config_file = name
            elif lower == "generation_config.json":
                generation_config_file = name
            continue
        if _is_tokenizer_asset(name):
            assets.append({"source": str(source), "dest": name})
            tokenizer_files.append(name)
            continue
        if _is_processor_asset(name):
            assets.append({"source": str(source), "dest": name})

    return assets, tokenizer_files, config_file, generation_config_file


def _write_blob(workspace: Path, prefix: str, name: str, payload: bytes) -> dict[str, Any]:
    path = workspace / f"{prefix}-{_safe_name(name)}.bin"
    path.write_bytes(payload)
    return {"path": str(path), "offset": 0, "length": len(payload)}


def _read_slice_bytes(tensor: TensorSlice) -> bytes:
    with tensor.source_path.open("rb") as handle:
        handle.seek(tensor.source_offset)
        return handle.read(tensor.length)


def _decode_bfloat16(payload: bytes) -> np.ndarray:
    words = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
    return (words << 16).view("<f4")


def _tensor_array(tensor: TensorSlice) -> np.ndarray:
    payload = _read_slice_bytes(tensor)
    if tensor.dtype == "fp16":
        array = np.frombuffer(payload, dtype="<f2").astype(np.float32)
    elif tensor.dtype == "fp32":
        array = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    elif tensor.dtype == "bf16":
        array = _decode_bfloat16(payload)
    else:
        raise ValueError(f"unsupported source dtype {tensor.dtype} for tensor {tensor.name}")
    return np.array(array.reshape(tensor.shape), copy=True)


def _default_generation(config_dir: Path) -> dict[str, Any]:
    generation_path = config_dir / "generation_config.json"
    if generation_path.exists():
        generation = _read_json(generation_path)
        return {
            "temperature": generation.get("temperature", 0.8),
            "top_p": generation.get("top_p", 0.95),
            "top_k": generation.get("top_k", 40),
            "max_new_tokens": generation.get("max_new_tokens", 256),
        }
    return {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "max_new_tokens": 256,
    }


def _moe_from_hf_config(config: dict[str, Any]) -> dict[str, Any] | None:
    num_experts = _int_or_none(
        _first_present(
            config,
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
            "expert_count",
        )
    )
    experts_per_token = _int_or_none(
        _first_present(
            config,
            "num_experts_per_tok",
            "num_experts_per_token",
            "experts_per_token",
            "num_selected_experts",
            "top_k_experts",
        )
    )
    expert_intermediate_dim = _int_or_none(
        _first_present(
            config,
            "moe_intermediate_size",
            "expert_intermediate_size",
            "ffn_moe_intermediate_size",
        )
    )
    if not (num_experts and experts_per_token and expert_intermediate_dim):
        return None

    moe = {
        "num_experts": num_experts,
        "experts_per_token": experts_per_token,
        "expert_intermediate_dim": expert_intermediate_dim,
    }
    num_shared_experts = _int_or_none(_first_present(config, "num_shared_experts", "n_shared_experts"))
    if num_shared_experts is not None:
        moe["num_shared_experts"] = num_shared_experts
    shared_expert_intermediate_dim = _int_or_none(_first_present(config, "shared_expert_intermediate_size"))
    if shared_expert_intermediate_dim is not None:
        moe["shared_expert_intermediate_dim"] = shared_expert_intermediate_dim
    router_aux_loss_coef = _float_or_none(_first_present(config, "router_aux_loss_coef", "aux_loss_alpha"))
    if router_aux_loss_coef is not None:
        moe["router_aux_loss_coef"] = router_aux_loss_coef
    expert_layer_frequency = _int_or_none(_first_present(config, "moe_layer_freq", "expert_layer_freq"))
    if expert_layer_frequency is not None:
        moe["expert_layer_frequency"] = expert_layer_frequency
    return moe


def _model_from_hf_config(config: dict[str, Any]) -> dict[str, Any]:
    hidden_dim = config.get("hidden_size") or config.get("d_model") or 0
    num_heads = config.get("num_attention_heads") or config.get("n_head") or 0
    num_kv_heads = config.get("num_key_value_heads") or num_heads
    head_dim = config.get("head_dim")
    if not head_dim and hidden_dim and num_heads:
        head_dim = hidden_dim // num_heads
    rope_theta = config.get("rope_theta")
    rope_type = "rope" if rope_theta is not None else "unknown"
    model = {
        "hidden_dim": hidden_dim,
        "intermediate_dim": config.get("intermediate_size") or config.get("ffn_dim") or 0,
        "num_layers": config.get("num_hidden_layers") or config.get("n_layer") or 0,
        "num_attention_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim or 0,
        "context_length": config.get("max_position_embeddings") or config.get("n_positions") or config.get("n_ctx") or 0,
        "vocab_size": config.get("vocab_size") or 0,
        "rope": {
            "type": rope_type,
            "theta": rope_theta,
            "scaling": config.get("rope_scaling"),
        },
    }
    total_parameter_count = _int_or_none(
        _first_present(config, "total_parameter_count", "num_parameters", "parameter_count")
    )
    if total_parameter_count is not None:
        model["total_parameter_count"] = total_parameter_count
    active_parameter_count = _int_or_none(
        _first_present(config, "active_parameter_count", "num_active_parameters", "active_num_parameters")
    )
    if active_parameter_count is not None:
        model["active_parameter_count"] = active_parameter_count
    moe = _moe_from_hf_config(config)
    if moe is not None:
        model["moe"] = moe
    return model


def _runtime_from_hf_config(config_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "bos_token_id": config.get("bos_token_id"),
        "eos_token_id": config.get("eos_token_id"),
        "pad_token_id": config.get("pad_token_id"),
        "default_generation": _default_generation(config_dir),
    }


def _collect_assets(model_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any] | None, str | None, str | None]:
    assets, tokenizer_files, config_file, generation_config_file = _asset_inventory(model_dir)
    tokenizer = None
    if tokenizer_files:
        tokenizer = {"kind": "huggingface", "files": tokenizer_files}
    return assets, tokenizer, config_file, generation_config_file


def _default_tensor_descriptor(tensor: TensorSlice, stream_order: int) -> dict[str, Any]:
    return {
        "shape": tensor.shape,
        "dtype": tensor.dtype,
        "bits": None,
        "group_size": None,
        "source_tensor_name": tensor.name,
        "data_offset": 0,
        "data_bytes": tensor.length,
        "scale_interleaved": False,
        "outlier_indices_offset": None,
        "outlier_count": None,
        "sensitivity_score": None,
        "stream_order": stream_order,
        "per_head_bits": None,
        "nf_scale_fp16": False,
        "smoothquant_scale": None,
        "prefetch_priority": None,
        "codebook_id": None,
        "vq_dim": None,
        "dedup_canonical": None,
        "lora_rank": None,
        "lora_alpha": None,
        "target": None,
        "checksum_xxh64": None,
    }


def _hf_tensors(model_dir: Path) -> list[TensorSlice]:
    tensors: list[TensorSlice] = []
    shard_paths = _hf_shard_paths(model_dir)
    for shard in shard_paths:
        with shard.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len).decode("utf-8"))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            dtype = SAFE_TENSOR_DTYPES.get(entry["dtype"])
            if dtype is None:
                raise ValueError(f"unsupported safetensors dtype {entry['dtype']} in {shard}")
            start, end = entry["data_offsets"]
            tensors.append(
                TensorSlice(
                    name=name,
                    shape=[int(value) for value in entry["shape"]],
                    dtype=dtype,
                    source_path=shard,
                    source_offset=8 + header_len + int(start),
                    length=int(end) - int(start),
                )
            )
    tensors.sort(key=lambda tensor: tensor.name)
    return tensors


class GgufReader:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("rb")

    def close(self) -> None:
        self.handle.close()

    def read_bytes(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise EOFError("unexpected end of GGUF file")
        return data

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read_bytes(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.read_bytes(8))[0]

    def read_i8(self) -> int:
        return struct.unpack("<b", self.read_bytes(1))[0]

    def read_u8(self) -> int:
        return struct.unpack("<B", self.read_bytes(1))[0]

    def read_i16(self) -> int:
        return struct.unpack("<h", self.read_bytes(2))[0]

    def read_u16(self) -> int:
        return struct.unpack("<H", self.read_bytes(2))[0]

    def read_i32(self) -> int:
        return struct.unpack("<i", self.read_bytes(4))[0]

    def read_i64(self) -> int:
        return struct.unpack("<q", self.read_bytes(8))[0]

    def read_f32(self) -> float:
        return struct.unpack("<f", self.read_bytes(4))[0]

    def read_f64(self) -> float:
        return struct.unpack("<d", self.read_bytes(8))[0]

    def read_bool(self) -> bool:
        return bool(self.read_u8())

    def read_string(self) -> str:
        length = self.read_u64()
        return self.read_bytes(length).decode("utf-8")

    def read_value(self, value_type: int) -> Any:
        if value_type == 0:
            return self.read_u8()
        if value_type == 1:
            return self.read_i8()
        if value_type == 2:
            return self.read_u16()
        if value_type == 3:
            return self.read_i16()
        if value_type == 4:
            return self.read_u32()
        if value_type == 5:
            return self.read_i32()
        if value_type == 6:
            return self.read_f32()
        if value_type == 7:
            return self.read_bool()
        if value_type == 8:
            return self.read_string()
        if value_type == 9:
            element_type = self.read_u32()
            count = self.read_u64()
            return [self.read_value(element_type) for _ in range(count)]
        if value_type == 10:
            return self.read_u64()
        if value_type == 11:
            return self.read_i64()
        if value_type == 12:
            return self.read_f64()
        raise ValueError(f"unsupported GGUF metadata value type {value_type}")


def _parse_gguf(path: Path) -> tuple[dict[str, Any], list[TensorSlice]]:
    reader = GgufReader(path)
    try:
        if reader.read_bytes(4) != b"GGUF":
            raise ValueError("invalid GGUF magic")
        version = reader.read_u32()
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count = reader.read_u64()
        metadata_count = reader.read_u64()
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = reader.read_string()
            value_type = reader.read_u32()
            metadata[key] = reader.read_value(value_type)

        tensor_infos: list[tuple[str, list[int], int, int]] = []
        for _ in range(tensor_count):
            name = reader.read_string()
            dimension_count = reader.read_u32()
            dimensions = [reader.read_u64() for _ in range(dimension_count)]
            tensor_type = reader.read_u32()
            offset = reader.read_u64()
            tensor_infos.append((name, [int(value) for value in dimensions], tensor_type, offset))

        alignment = int(metadata.get("general.alignment", 32))
        data_start = reader.handle.tell()
        if data_start % alignment:
            data_start += alignment - (data_start % alignment)

        tensors: list[TensorSlice] = []
        for name, dimensions, tensor_type, offset in tensor_infos:
            dtype = GGUF_TENSOR_TYPES.get(tensor_type)
            if dtype is None:
                raise ValueError(f"unsupported GGUF tensor type {tensor_type} for tensor {name}")
            scalar_bytes = {"fp16": 2, "fp32": 4, "bf16": 2}[dtype]
            length = _product(dimensions) * scalar_bytes
            tensors.append(
                TensorSlice(
                    name=name,
                    shape=dimensions,
                    dtype=dtype,
                    source_path=path,
                    source_offset=data_start + offset,
                    length=length,
                )
            )
        tensors.sort(key=lambda tensor: tensor.name)
        return metadata, tensors
    finally:
        reader.close()


def _gguf_metadata(path: Path) -> tuple[dict[str, Any], list[TensorSlice], dict[str, Any], dict[str, Any]]:
    metadata, tensors = _parse_gguf(path)
    architecture = str(metadata.get("general.architecture", "unknown"))
    prefix = architecture
    hidden_dim = int(metadata.get(f"{prefix}.embedding_length", 0))
    num_heads = int(metadata.get(f"{prefix}.attention.head_count", 0))
    num_kv_heads = int(metadata.get(f"{prefix}.attention.head_count_kv", num_heads or 0))
    head_dim = int(metadata.get(f"{prefix}.attention.key_length", 0))
    if not head_dim and hidden_dim and num_heads:
        head_dim = hidden_dim // num_heads
    vocab_tokens = metadata.get("tokenizer.ggml.tokens", [])
    vocab_size = len(vocab_tokens) if isinstance(vocab_tokens, list) else int(metadata.get(f"{prefix}.vocab_size", 0))
    model = {
        "hidden_dim": hidden_dim,
        "intermediate_dim": int(metadata.get(f"{prefix}.feed_forward_length", 0)),
        "num_layers": int(metadata.get(f"{prefix}.block_count", 0)),
        "num_attention_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "context_length": int(metadata.get(f"{prefix}.context_length", 0)),
        "vocab_size": vocab_size,
        "rope": {
            "type": "rope",
            "theta": metadata.get(f"{prefix}.rope.freq_base", metadata.get(f"{prefix}.rope.freq_base_train")),
            "scaling": None,
        },
    }
    total_parameter_count = _int_or_none(
        _first_present(
            metadata,
            f"{prefix}.parameter_count",
            "general.parameter_count",
            "general.total_parameter_count",
        )
    )
    if total_parameter_count is not None:
        model["total_parameter_count"] = total_parameter_count
    active_parameter_count = _int_or_none(
        _first_present(
            metadata,
            f"{prefix}.active_parameter_count",
            "general.active_parameter_count",
        )
    )
    if active_parameter_count is not None:
        model["active_parameter_count"] = active_parameter_count
    num_experts = _int_or_none(
        _first_present(
            metadata,
            f"{prefix}.expert_count",
            f"{prefix}.num_experts",
        )
    )
    experts_per_token = _int_or_none(
        _first_present(
            metadata,
            f"{prefix}.expert_used_count",
            f"{prefix}.num_experts_used",
            f"{prefix}.experts_per_token",
        )
    )
    expert_intermediate_dim = _int_or_none(
        _first_present(
            metadata,
            f"{prefix}.expert_feed_forward_length",
            f"{prefix}.expert_intermediate_length",
        )
    )
    if num_experts and experts_per_token and expert_intermediate_dim:
        moe = {
            "num_experts": num_experts,
            "experts_per_token": experts_per_token,
            "expert_intermediate_dim": expert_intermediate_dim,
        }
        num_shared_experts = _int_or_none(_first_present(metadata, f"{prefix}.shared_expert_count"))
        if num_shared_experts is not None:
            moe["num_shared_experts"] = num_shared_experts
        shared_expert_intermediate_dim = _int_or_none(
            _first_present(metadata, f"{prefix}.shared_expert_feed_forward_length")
        )
        if shared_expert_intermediate_dim is not None:
            moe["shared_expert_intermediate_dim"] = shared_expert_intermediate_dim
        router_aux_loss_coef = _float_or_none(_first_present(metadata, f"{prefix}.router_aux_loss_coef"))
        if router_aux_loss_coef is not None:
            moe["router_aux_loss_coef"] = router_aux_loss_coef
        expert_layer_frequency = _int_or_none(_first_present(metadata, f"{prefix}.expert_layer_frequency"))
        if expert_layer_frequency is not None:
            moe["expert_layer_frequency"] = expert_layer_frequency
        model["moe"] = moe
    runtime = {
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
        "default_generation": {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_new_tokens": 256,
        },
    }
    return metadata, tensors, model, runtime


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    if not match:
        return None
    return int(match.group(1))


def _component_rank(name: str) -> int:
    lower = name.lower()
    if any(token in lower for token in ("input_layernorm", "attention_norm", "attn_norm")):
        return 0
    if any(token in lower for token in ("q_proj", "wq")):
        return 1
    if any(token in lower for token in ("k_proj", "wk")):
        return 2
    if any(token in lower for token in ("v_proj", "wv")):
        return 3
    if any(token in lower for token in ("o_proj", "wo")):
        return 4
    if any(token in lower for token in ("post_attention_layernorm", "ffn_norm")):
        return 5
    if any(token in lower for token in ("gate_proj", "w1")):
        return 6
    if any(token in lower for token in ("up_proj", "w3")):
        return 7
    if any(token in lower for token in ("down_proj", "w2")):
        return 8
    return 9


def _stream_order_key(name: str) -> tuple[int, int, int, str]:
    lower = name.lower()
    if any(token in lower for token in ("embed_tokens", "tok_embeddings", "wte", "token_embd", "embedding")):
        return (0, 0, 0, name)
    layer_index = _layer_index(name)
    if layer_index is not None:
        return (1, layer_index, _component_rank(name), name)
    if lower.endswith("norm.weight") or lower == "norm.weight":
        return (2, 0, 0, name)
    if "lm_head" in lower or lower.endswith("output.weight"):
        return (3, 0, 0, name)
    return (4, 0, 0, name)


def _sort_tensors(tensors: list[TensorSlice]) -> list[TensorSlice]:
    return sorted(tensors, key=lambda tensor: _stream_order_key(tensor.name))


def _is_quantizable_tensor(tensor: TensorSlice) -> bool:
    return len(tensor.shape) >= 2 and tensor.dtype in {"fp16", "bf16", "fp32"}


def _is_embedding_like(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ("embed", "tok_embeddings", "wte", "lm_head", "output.weight"))


def _mxq_bits_for_tensor(name: str, num_layers: int) -> int | None:
    lower = name.lower()
    if "router" in lower:
        return None
    layer_index = _layer_index(name)
    if _is_embedding_like(name):
        return 8
    if layer_index is not None and num_layers > 0 and layer_index in {0, max(0, num_layers - 1)}:
        return 8
    if any(token in lower for token in ("q_proj", "k_proj", "v_proj", "o_proj", ".attn.", ".self_attn.", "attention.wq", "attention.wk", "attention.wv", "attention.wo")):
        return 4
    if any(token in lower for token in ("gate_proj", "up_proj", "w1", "w3")):
        return 3
    if any(token in lower for token in ("down_proj", "w2")):
        return 2
    return 4


def _sensitivity_for_bits(bits: int) -> float:
    return {8: 0.92, 4: 0.68, 3: 0.42, 2: 0.18}.get(bits, 0.5)


def _prefetch_priority(name: str) -> float:
    lower = name.lower()
    if "router" in lower:
        return 1.0
    if _is_embedding_like(name):
        return 0.8
    if any(
        token in lower
        for token in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            ".attn.",
            ".self_attn.",
            "attention.wq",
            "attention.wk",
            "attention.wv",
            "attention.wo",
        )
    ):
        return 0.9
    if any(token in lower for token in ("gate_proj", "up_proj", "w1", "w3")):
        return 0.6
    if any(token in lower for token in ("down_proj", "w2")):
        return 0.4
    return 0.5


def _preferred_group_size(model: dict[str, Any], override: int | None) -> int:
    if override:
        return override
    hidden_dim = int(model.get("hidden_dim", 0) or 0)
    return 64 if hidden_dim and hidden_dim < 2048 else 128


def _is_shared_expert_tensor(name: str) -> bool:
    lower = name.lower()
    return "shared_expert" in lower or "shared_experts" in lower


def _is_split_expert_tensor(name: str) -> bool:
    if _is_shared_expert_tensor(name):
        return False
    return bool(_SPLIT_EXPERT_RE.search(name))


def _normalize_split_expert_name(name: str) -> str:
    return _SPLIT_EXPERT_RE.sub(r"\1*\2", name)


def _is_stacked_expert_tensor(tensor: TensorSlice, num_experts: int) -> bool:
    if _is_shared_expert_tensor(tensor.name):
        return False
    lower = tensor.name.lower()
    if "router" in lower or num_experts <= 0 or not tensor.shape:
        return False
    if int(tensor.shape[0]) != num_experts:
        return False
    return any(marker in lower for marker in ("experts", "expert", "_exps", ".exps"))


def _attach_parameter_counts(model: dict[str, Any], tensors: list[TensorSlice]) -> None:
    total_params = sum(_product(tensor.shape) for tensor in tensors)
    if total_params > 0 and model.get("total_parameter_count") is None:
        model["total_parameter_count"] = total_params

    moe = model.get("moe")
    if not isinstance(moe, dict):
        if total_params > 0 and model.get("active_parameter_count") is None:
            model["active_parameter_count"] = total_params
        return

    num_experts = int(moe.get("num_experts", 0) or 0)
    experts_per_token = int(moe.get("experts_per_token", 0) or 0)
    if num_experts <= 0 or experts_per_token <= 0:
        if total_params > 0 and model.get("active_parameter_count") is None:
            model["active_parameter_count"] = total_params
        return

    routed_total = 0
    routed_active = 0.0
    grouped_split_tensors: dict[str, list[float]] = {}
    for tensor in tensors:
        params = _product(tensor.shape)
        if params == 0:
            continue
        if _is_split_expert_tensor(tensor.name):
            routed_total += params
            group = grouped_split_tensors.setdefault(_normalize_split_expert_name(tensor.name), [0.0, 0.0])
            group[0] += float(params)
            group[1] += 1.0
            continue
        if _is_stacked_expert_tensor(tensor, num_experts):
            routed_total += params
            expert_slice = _product(tensor.shape[1:]) if len(tensor.shape) > 1 else 1
            routed_active += float(expert_slice * experts_per_token)

    for total_group, expert_count in grouped_split_tensors.values():
        if expert_count <= 0:
            continue
        routed_active += (total_group / expert_count) * experts_per_token

    if model.get("active_parameter_count") is None and total_params > 0:
        always_on = max(total_params - routed_total, 0)
        active = int(round(always_on + routed_active))
        model["active_parameter_count"] = min(total_params, max(active, 0))


def _fingerprint_path(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=8)
    if path.is_dir():
        for source in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            relative = source.relative_to(path).as_posix().encode("utf-8")
            hasher.update(relative)
            hasher.update(source.read_bytes())
    else:
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _boot_cutoff_stream_order(tensors: list[TensorSlice], num_layers: int, override_layers: int | None) -> int | None:
    if not tensors:
        return None
    cutoff_layers = override_layers
    if cutoff_layers is None and num_layers >= 4:
        cutoff_layers = max(1, num_layers // 4)
    if cutoff_layers is None or cutoff_layers <= 0:
        return None
    for stream_order, tensor in enumerate(tensors):
        layer_index = _layer_index(tensor.name)
        if layer_index is not None and layer_index >= cutoff_layers:
            return stream_order
    return None


def _dependency_graph_from_tensors(tensors: list[TensorSlice]) -> dict[str, Any] | None:
    groups: list[list[str]] = []
    by_layer: dict[int, list[str]] = {}
    for tensor in tensors:
        layer_index = _layer_index(tensor.name)
        if layer_index is None:
            continue
        by_layer.setdefault(layer_index, []).append(tensor.name)

    for layer_names in by_layer.values():
        attn = [name for name in layer_names if any(token in name.lower() for token in ("q_proj", "k_proj", "v_proj"))]
        if len(attn) > 1:
            groups.append(sorted(attn))
        mlp = [name for name in layer_names if any(token in name.lower() for token in ("gate_proj", "up_proj", "w1", "w3"))]
        if len(mlp) > 1:
            groups.append(sorted(mlp))

    if not groups:
        return None
    return {"parallel_groups": groups}


def _sparse_correction_payload(delta: np.ndarray) -> tuple[bytes | None, int]:
    if delta.ndim < 2:
        return None, 0
    rows = delta.reshape(delta.shape[0], -1).astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(rows))) if rows.size else 0.0
    threshold = max(1e-4, max_abs * 0.05)
    mask = np.abs(rows) >= threshold
    if not mask.any():
        return None, 0
    row_ptr = np.zeros(rows.shape[0] + 1, dtype="<u4")
    col_indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    nnz = 0
    for row_index in range(rows.shape[0]):
        indices = np.nonzero(mask[row_index])[0].astype(np.uint32)
        row_ptr[row_index] = nnz
        if indices.size:
            col_indices.append(indices.astype("<u4", copy=False))
            values.append(rows[row_index, indices].astype("<f2"))
            nnz += int(indices.size)
    row_ptr[-1] = nnz
    if nnz == 0:
        return None, 0
    payload = row_ptr.tobytes(order="C")
    payload += b"".join(index_array.tobytes(order="C") for index_array in col_indices)
    payload += b"".join(value_array.tobytes(order="C") for value_array in values)
    return payload, nnz


def _apply_expert_dedup(
    tensors: list[TensorSlice],
    tensor_map: dict[str, Any],
    tensor_sources: dict[str, Any],
    workspace: Path,
    similarity_threshold: float | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if similarity_threshold is None:
        return None, {}

    grouped: dict[str, list[TensorSlice]] = {}
    for tensor in tensors:
        if _is_split_expert_tensor(tensor.name):
            grouped.setdefault(_normalize_split_expert_name(tensor.name), []).append(tensor)

    canonical_map: dict[str, str] = {}
    correction_sources: dict[str, Any] = {}
    for group_tensors in grouped.values():
        if len(group_tensors) < 2:
            continue
        cached = {tensor.name: _tensor_array(tensor).astype(np.float32, copy=False) for tensor in group_tensors}
        canonicals: list[str] = []
        for tensor in group_tensors:
            candidate = cached[tensor.name]
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm == 0.0:
                canonicals.append(tensor.name)
                continue
            best_name = None
            best_similarity = -1.0
            for canonical_name in canonicals:
                base = cached[canonical_name]
                denom = candidate_norm * float(np.linalg.norm(base))
                if denom == 0.0:
                    continue
                similarity = float(np.dot(candidate.reshape(-1), base.reshape(-1)) / denom)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = canonical_name
            if best_name is None or best_similarity < similarity_threshold:
                canonicals.append(tensor.name)
                continue
            delta = candidate - cached[best_name]
            payload, nnz = _sparse_correction_payload(delta)
            if payload is None or nnz == 0:
                canonicals.append(tensor.name)
                continue
            density = nnz / max(delta.size, 1)
            if density > 0.5 or len(payload) >= tensor.length:
                canonicals.append(tensor.name)
                continue
            descriptor = tensor_map[tensor.name]
            descriptor["dedup_canonical"] = best_name
            descriptor["dedup_correction_offset"] = 0
            descriptor["dedup_correction_count"] = nnz
            descriptor["data_offset"] = 0
            descriptor["data_bytes"] = 0
            descriptor["checksum_xxh64"] = None
            canonical_map[tensor.name] = best_name
            tensor_sources.pop(tensor.name, None)
            correction_sources[tensor.name] = _write_blob(workspace, "expert-dedup", tensor.name, payload)

    if not canonical_map:
        return None, {}

    return (
        {
            "region_offset": 0,
            "region_bytes": sum(int(source["length"]) for source in correction_sources.values()),
            "canonical_map": canonical_map,
            "corrections_offset": 0,
            "similarity_threshold": similarity_threshold,
        },
        correction_sources,
    )


def _pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    if bits == 8:
        return codes.astype(np.uint8, copy=False)
    if bits == 4:
        padded = np.pad(codes, ((0, 0), (0, (-codes.shape[1]) % 2)), constant_values=0)
        reshaped = padded.reshape(padded.shape[0], -1, 2).astype(np.uint8, copy=False)
        return reshaped[:, :, 0] | (reshaped[:, :, 1] << 4)
    if bits == 2:
        padded = np.pad(codes, ((0, 0), (0, (-codes.shape[1]) % 4)), constant_values=0)
        reshaped = padded.reshape(padded.shape[0], -1, 4).astype(np.uint8, copy=False)
        return reshaped[:, :, 0] | (reshaped[:, :, 1] << 2) | (reshaped[:, :, 2] << 4) | (reshaped[:, :, 3] << 6)
    if bits == 3:
        padded = np.pad(codes, ((0, 0), (0, (-codes.shape[1]) % 8)), constant_values=0)
        reshaped = padded.reshape(padded.shape[0], -1, 8).astype(np.uint32, copy=False)
        words = (
            reshaped[:, :, 0]
            | (reshaped[:, :, 1] << 3)
            | (reshaped[:, :, 2] << 6)
            | (reshaped[:, :, 3] << 9)
            | (reshaped[:, :, 4] << 12)
            | (reshaped[:, :, 5] << 15)
            | (reshaped[:, :, 6] << 18)
            | (reshaped[:, :, 7] << 21)
        )
        return np.stack(
            [
                (words & 0xFF).astype(np.uint8),
                ((words >> 8) & 0xFF).astype(np.uint8),
                ((words >> 16) & 0xFF).astype(np.uint8),
            ],
            axis=2,
        ).reshape(codes.shape[0], -1)
    raise ValueError(f"unsupported bit width {bits}")


def _extract_outliers(weights: np.ndarray, sigma: float) -> tuple[np.ndarray, bytes | None, int]:
    if weights.ndim < 2:
        return weights, None, 0
    rows = weights.reshape(weights.shape[0], -1)
    row_mean = rows.mean(axis=1, keepdims=True)
    row_std = rows.std(axis=1, keepdims=True)
    threshold = np.abs(row_mean) + (sigma * row_std)
    mask = np.abs(rows) > threshold
    if not mask.any():
        return weights, None, 0

    residual = rows.copy()
    residual[mask] = 0.0
    row_ptr = np.zeros(rows.shape[0] + 1, dtype="<u4")
    col_indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    nnz = 0
    for row_index in range(rows.shape[0]):
        indices = np.nonzero(mask[row_index])[0].astype(np.uint32)
        row_ptr[row_index] = nnz
        if indices.size:
            col_indices.append(indices.astype("<u4", copy=False))
            values.append(rows[row_index, indices].astype("<f2"))
            nnz += int(indices.size)
    row_ptr[-1] = nnz
    col_idx_bytes = b"".join(index_array.tobytes(order="C") for index_array in col_indices)
    value_bytes = b"".join(value_array.tobytes(order="C") for value_array in values)
    payload = row_ptr.tobytes(order="C") + col_idx_bytes + value_bytes
    return residual.reshape(weights.shape), payload, nnz


def _extract_outliers_torch(weights: np.ndarray, sigma: float) -> tuple[np.ndarray, bytes | None, int]:
    if weights.ndim < 2:
        return weights, None, 0
    device = torch.device("cuda")
    rows_np = weights.reshape(weights.shape[0], -1).astype(np.float32, copy=False)
    rows = torch.from_numpy(rows_np).to(device=device, dtype=torch.float32)
    row_mean = rows.mean(dim=1, keepdim=True)
    row_std = rows.std(dim=1, keepdim=True, unbiased=False)
    threshold = row_mean.abs() + (sigma * row_std)
    mask = rows.abs() > threshold
    if not bool(mask.any().item()):
        return weights, None, 0

    residual = rows.masked_fill(mask, 0.0).cpu().numpy()
    mask_np = mask.cpu().numpy()
    row_ptr = np.zeros(rows_np.shape[0] + 1, dtype="<u4")
    col_indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    nnz = 0
    for row_index in range(rows_np.shape[0]):
        indices = np.nonzero(mask_np[row_index])[0].astype(np.uint32)
        row_ptr[row_index] = nnz
        if indices.size:
            col_indices.append(indices.astype("<u4", copy=False))
            values.append(rows_np[row_index, indices].astype("<f2"))
            nnz += int(indices.size)
    row_ptr[-1] = nnz
    col_idx_bytes = b"".join(index_array.tobytes(order="C") for index_array in col_indices)
    value_bytes = b"".join(value_array.tobytes(order="C") for value_array in values)
    payload = row_ptr.tobytes(order="C") + col_idx_bytes + value_bytes
    return residual.reshape(weights.shape), payload, nnz


def _mxq_pack(weights: np.ndarray, bits: int, group_size: int, outlier_sigma: float) -> tuple[bytes, bytes | None, int]:
    residual, outlier_payload, outlier_count = _extract_outliers(weights.astype(np.float32), outlier_sigma)
    flat = residual.astype(np.float32, copy=False).reshape(-1)
    if flat.size == 0:
        return b"", outlier_payload, outlier_count
    groups = math.ceil(flat.size / group_size)
    padded_size = groups * group_size
    if padded_size != flat.size:
        padded = np.zeros(padded_size, dtype=np.float32)
        padded[: flat.size] = flat
        flat = padded
    grouped = flat.reshape(groups, group_size)
    max_level = float((1 << (bits - 1)) - 1) if bits > 1 else 1.0
    scales = np.max(np.abs(grouped), axis=1)
    nonzero = scales > 0
    scales = np.where(nonzero, scales / max_level, 0.0).astype(np.float32)
    zero_point = 1 << (bits - 1)
    codes = np.full(grouped.shape, zero_point, dtype=np.uint8)
    if np.any(nonzero):
        normalized = np.zeros_like(grouped, dtype=np.float32)
        normalized[nonzero] = grouped[nonzero] / scales[nonzero, None]
        quantized = np.rint(normalized).astype(np.int32) + zero_point
        codes = np.clip(quantized, 0, (1 << bits) - 1).astype(np.uint8)
    packed = _pack_codes(codes, bits)
    scale_bytes = scales.astype("<f2").view(np.uint8).reshape(groups, 2)
    payload = bytearray()
    for group_index in range(groups):
        payload.extend(scale_bytes[group_index].tobytes(order="C"))
        payload.extend(packed[group_index].tobytes(order="C"))
    return bytes(payload), outlier_payload, outlier_count


def _mxq_pack_torch(weights: np.ndarray, bits: int, group_size: int, outlier_sigma: float) -> tuple[bytes, bytes | None, int]:
    residual, outlier_payload, outlier_count = _extract_outliers_torch(weights.astype(np.float32), outlier_sigma)
    flat = residual.astype(np.float32, copy=False).reshape(-1)
    if flat.size == 0:
        return b"", outlier_payload, outlier_count
    device = torch.device("cuda")
    flat_tensor = torch.from_numpy(flat).to(device=device, dtype=torch.float32)
    groups = math.ceil(flat.size / group_size)
    padded_size = groups * group_size
    if padded_size != flat.size:
        padded = torch.zeros(padded_size, device=device, dtype=torch.float32)
        padded[: flat.size] = flat_tensor
        flat_tensor = padded
    grouped = flat_tensor.reshape(groups, group_size)
    max_level = float((1 << (bits - 1)) - 1) if bits > 1 else 1.0
    scales = grouped.abs().amax(dim=1)
    nonzero = scales > 0
    scales = torch.where(nonzero, scales / max_level, torch.zeros_like(scales))
    zero_point = 1 << (bits - 1)
    codes = torch.full(grouped.shape, zero_point, device=device, dtype=torch.int32)
    if bool(nonzero.any().item()):
        normalized = torch.zeros_like(grouped)
        normalized[nonzero] = grouped[nonzero] / scales[nonzero].unsqueeze(1)
        quantized = torch.round(normalized).to(torch.int32) + zero_point
        codes = torch.clamp(quantized, 0, (1 << bits) - 1)
    codes_np = codes.to(dtype=torch.uint8).cpu().numpy()
    packed = _pack_codes(codes_np, bits)
    scale_bytes = scales.to(dtype=torch.float16).cpu().numpy().astype("<f2", copy=False).view(np.uint8).reshape(groups, 2)
    payload = bytearray()
    for group_index in range(groups):
        payload.extend(scale_bytes[group_index].tobytes(order="C"))
        payload.extend(packed[group_index].tobytes(order="C"))
    return bytes(payload), outlier_payload, outlier_count


def _nf_levels(bits: int) -> np.ndarray:
    if bits == 2:
        return np.array([-1.0, -0.3333, 0.3333, 1.0], dtype=np.float32)
    if bits == 3:
        return np.array([-1.0, -0.5774, -0.3333, -0.1111, 0.1111, 0.3333, 0.5774, 1.0], dtype=np.float32)
    raise ValueError(f"unsupported NF bit width {bits}")


def _nf_pack(weights: np.ndarray, bits: int, group_size: int, outlier_sigma: float) -> tuple[bytes, bytes | None, int]:
    residual, outlier_payload, outlier_count = _extract_outliers(weights.astype(np.float32), outlier_sigma)
    flat = residual.astype(np.float32, copy=False).reshape(-1)
    if flat.size == 0:
        return b"", outlier_payload, outlier_count
    groups = math.ceil(flat.size / group_size)
    padded_size = groups * group_size
    if padded_size != flat.size:
        padded = np.zeros(padded_size, dtype=np.float32)
        padded[: flat.size] = flat
        flat = padded
    grouped = flat.reshape(groups, group_size)
    scales = np.max(np.abs(grouped), axis=1).astype(np.float32)
    nonzero = scales > 0
    normalized = np.zeros_like(grouped, dtype=np.float32)
    normalized[nonzero] = grouped[nonzero] / scales[nonzero, None]
    levels = _nf_levels(bits)
    distances = np.abs(normalized[:, :, None] - levels[None, None, :])
    codes = distances.argmin(axis=2).astype(np.uint8)
    packed = _pack_codes(codes, bits)
    scale_bytes = scales.astype("<f2").view(np.uint8).reshape(groups, 2)
    payload = bytearray()
    for group_index in range(groups):
        payload.extend(scale_bytes[group_index].tobytes(order="C"))
        payload.extend(packed[group_index].tobytes(order="C"))
    return bytes(payload), outlier_payload, outlier_count


def _nf_pack_torch(weights: np.ndarray, bits: int, group_size: int, outlier_sigma: float) -> tuple[bytes, bytes | None, int]:
    residual, outlier_payload, outlier_count = _extract_outliers_torch(weights.astype(np.float32), outlier_sigma)
    flat = residual.astype(np.float32, copy=False).reshape(-1)
    if flat.size == 0:
        return b"", outlier_payload, outlier_count
    device = torch.device("cuda")
    flat_tensor = torch.from_numpy(flat).to(device=device, dtype=torch.float32)
    groups = math.ceil(flat.size / group_size)
    padded_size = groups * group_size
    if padded_size != flat.size:
        padded = torch.zeros(padded_size, device=device, dtype=torch.float32)
        padded[: flat.size] = flat_tensor
        flat_tensor = padded
    grouped = flat_tensor.reshape(groups, group_size)
    scales = grouped.abs().amax(dim=1)
    nonzero = scales > 0
    normalized = torch.zeros_like(grouped)
    normalized[nonzero] = grouped[nonzero] / scales[nonzero].unsqueeze(1)
    levels = torch.tensor(_nf_levels(bits), device=device, dtype=torch.float32)
    distances = (normalized.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
    codes = distances.argmin(dim=2).to(dtype=torch.uint8).cpu().numpy()
    packed = _pack_codes(codes, bits)
    scale_bytes = scales.to(dtype=torch.float16).cpu().numpy().astype("<f2", copy=False).view(np.uint8).reshape(groups, 2)
    payload = bytearray()
    for group_index in range(groups):
        payload.extend(scale_bytes[group_index].tobytes(order="C"))
        payload.extend(packed[group_index].tobytes(order="C"))
    return bytes(payload), outlier_payload, outlier_count


def _kmeans_codebook(vectors: np.ndarray, entries: int, seed: int = 0, max_iter: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample_count = min(len(vectors), max(entries * 32, entries))
    if sample_count < len(vectors):
        indices = np.linspace(0, len(vectors) - 1, sample_count, dtype=np.int64)
        sample = vectors[indices]
    else:
        sample = vectors
    if len(sample) <= entries:
        return sample.astype(np.float32, copy=True)
    initial = rng.choice(len(sample), size=entries, replace=False)
    centers = sample[initial].astype(np.float32, copy=True)
    for _ in range(max_iter):
        distances = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        for index in range(entries):
            members = sample[assignments == index]
            if len(members):
                centers[index] = members.mean(axis=0)
    return centers


def _assign_vectors(vectors: np.ndarray, centers: np.ndarray, batch: int = 4096) -> np.ndarray:
    assignments: list[np.ndarray] = []
    for start in range(0, len(vectors), batch):
        block = vectors[start : start + batch]
        distances = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignments.append(distances.argmin(axis=1).astype(np.uint8))
    return np.concatenate(assignments, axis=0) if assignments else np.zeros(0, dtype=np.uint8)


def _vq_pack(weights: np.ndarray, vq_dim: int = 8) -> tuple[bytes, bytes, int]:
    if weights.ndim != 2:
        raise ValueError("VQ packing only supports 2D tensors")
    rows, cols = weights.shape
    if cols % vq_dim != 0:
        raise ValueError(f"tensor width {cols} is not divisible by VQ dim {vq_dim}")
    vectors = weights.astype(np.float32, copy=False).reshape(rows * (cols // vq_dim), vq_dim)
    entries = min(256, len(vectors))
    if entries == 0:
        return b"", b"", vq_dim
    centers = _kmeans_codebook(vectors, entries)
    assignments = _assign_vectors(vectors, centers)
    codebook_bytes = centers.astype("<f2").tobytes(order="C")
    return assignments.tobytes(order="C"), codebook_bytes, vq_dim


def _lora_target_tensor(name: str, targets: set[str]) -> bool:
    lower = name.lower()
    return any(target in lower for target in targets)


def _factor_lora_delta(delta: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = delta.reshape(delta.shape[0], -1).astype(np.float32, copy=False)
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    effective_rank = max(1, min(rank, len(singular_values)))
    scales = np.sqrt(singular_values[:effective_rank]).astype(np.float32)
    a = (scales[:, None] * vh[:effective_rank, :]).astype(np.float32)
    b = (u[:, :effective_rank] * scales[None, :]).astype(np.float32)
    return a, b


def _build_lora_delta_plan(
    input_path: Path,
    base_path: Path,
    source_format: str,
    workspace: Path,
    rank: int,
    alpha: int,
    target_modules: list[str],
) -> dict[str, Any]:
    if source_format != "hf":
        raise ValueError("LoRA delta generation currently supports Hugging Face safetensors inputs only")

    config = _read_json(input_path / "config.json")
    base_config = _read_json(base_path / "config.json")
    if config.get("model_type") != base_config.get("model_type"):
        raise ValueError("LoRA base model_type does not match fine-tuned model_type")

    tensors = {tensor.name: tensor for tensor in _sort_tensors(_hf_tensors(input_path))}
    base_tensors = {tensor.name: tensor for tensor in _sort_tensors(_hf_tensors(base_path))}
    missing = sorted(name for name in tensors if name not in base_tensors)
    if missing:
        raise ValueError(f"base model is missing tensors required for LoRA delta: {missing[:4]}")

    model = _model_from_hf_config(config)
    runtime = _runtime_from_hf_config(input_path, config)
    assets, tokenizer, config_file, generation_config_file = _collect_assets(input_path)
    architecture = (config.get("architectures") or [config.get("model_type", "unknown")])[0]
    model_family = config.get("model_type") or input_path.name
    _attach_parameter_counts(model, list(tensors.values()))

    tensor_map: dict[str, Any] = {}
    lora_sources: dict[str, Any] = {}
    selected_targets = {target.lower() for target in target_modules}
    for name, tensor in tensors.items():
        if not _is_quantizable_tensor(tensor) or not _lora_target_tensor(name, selected_targets):
            continue
        fine = _tensor_array(tensor)
        base = _tensor_array(base_tensors[name])
        if fine.shape != base.shape or fine.ndim < 2:
            continue
        delta = fine.astype(np.float32) - base.astype(np.float32)
        if not np.any(np.abs(delta) > 1e-6):
            continue
        a, b = _factor_lora_delta(delta, rank)
        a_payload = a.astype("<f2").tobytes(order="C")
        b_payload = b.astype("<f2").tobytes(order="C")
        a_name = f"{name}.lora_A"
        b_name = f"{name}.lora_B"
        tensor_map[a_name] = {
            "shape": list(a.shape),
            "dtype": "fp16",
            "bits": None,
            "group_size": None,
            "source_tensor_name": a_name,
            "data_offset": 0,
            "data_bytes": len(a_payload),
            "scale_interleaved": False,
            "outlier_indices_offset": None,
            "outlier_count": None,
            "sensitivity_score": None,
            "stream_order": len(tensor_map),
            "per_head_bits": None,
            "nf_scale_fp16": False,
            "smoothquant_scale": None,
            "prefetch_priority": 0.9,
            "codebook_id": None,
            "vq_dim": None,
            "dedup_canonical": None,
            "dedup_correction_offset": None,
            "dedup_correction_count": None,
            "lora_rank": rank,
            "lora_alpha": alpha,
            "target": name,
            "checksum_xxh64": None,
        }
        tensor_map[b_name] = {
            "shape": list(b.shape),
            "dtype": "fp16",
            "bits": None,
            "group_size": None,
            "source_tensor_name": b_name,
            "data_offset": 0,
            "data_bytes": len(b_payload),
            "scale_interleaved": False,
            "outlier_indices_offset": None,
            "outlier_count": None,
            "sensitivity_score": None,
            "stream_order": len(tensor_map),
            "per_head_bits": None,
            "nf_scale_fp16": False,
            "smoothquant_scale": None,
            "prefetch_priority": 0.9,
            "codebook_id": None,
            "vq_dim": None,
            "dedup_canonical": None,
            "dedup_correction_offset": None,
            "dedup_correction_count": None,
            "lora_rank": rank,
            "lora_alpha": alpha,
            "target": name,
            "checksum_xxh64": None,
        }
        lora_sources[a_name] = _write_blob(workspace, "lora", a_name, a_payload)
        lora_sources[b_name] = _write_blob(workspace, "lora", b_name, b_payload)

    if not tensor_map:
        raise ValueError("no tensors matched the requested LoRA target modules")

    manifest: dict[str, Any] = {
        "manifest_version": "2.0.0-draft",
        "task": "causal_lm",
        "model_file": "model.axon",
    }
    if config_file:
        manifest["config_file"] = config_file
    if generation_config_file:
        manifest["generation_config_file"] = generation_config_file
    if tokenizer:
        manifest["tokenizer"] = tokenizer

    metadata = {
        "format": "axon",
        "version": "2.0.0-draft",
        "task": "causal_lm",
        "architecture": architecture,
        "model_family": model_family,
        "model": model,
        "runtime": runtime,
        "source": {
            "format": "hf_safetensors",
            "identifier": str(input_path),
            "conversion_tool": "axon-pack 2.0.0",
            "conversion_time_utc": _utc_now(),
        },
        "total_params": model.get("total_parameter_count"),
        "active_params": model.get("active_parameter_count"),
        "moe": None,
        "boot_region_bytes": 0,
        "kv_cache_hints": None,
        "tensor_dep_graph": None,
        "speculative_draft": None,
        "smoothquant_scales": {},
        "expert_dedup": None,
        "lora": {
            "base_model": f"{_safe_name(base_path.name).lower()}.axon",
            "base_hash": _fingerprint_path(base_path),
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules,
            "region_offset": 0,
            "region_bytes": sum(int(source["length"]) for source in lora_sources.values()),
        },
        "numa_hints": None,
        "avg_bits_per_weight": 0.0,
        "quant_method": "lora-delta",
        "calibration": None,
        "tensors": tensor_map,
        "codebooks": {},
        "hw_hints": {},
    }

    return {
        "manifest": manifest,
        "metadata": metadata,
        "boot_cutoff": None,
        "tensor_sources": {},
        "outlier_sources": {},
        "codebook_sources": {},
        "expert_dedup_sources": {},
        "speculative_draft_source": None,
        "lora_sources": lora_sources,
        "assets": assets,
    }


def _tensor_payload_plan(
    tensor: TensorSlice,
    model: dict[str, Any],
    stream_order: int,
    workspace: Path,
    options: PackOptions,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, tuple[str, dict[str, Any], dict[str, Any]] | None]:
    descriptor = _default_tensor_descriptor(tensor, stream_order)
    descriptor["prefetch_priority"] = _prefetch_priority(tensor.name)

    if options.quantization == "none" or not _is_quantizable_tensor(tensor):
        return (
            descriptor,
            {"path": str(tensor.source_path), "offset": tensor.source_offset, "length": tensor.length},
            None,
            None,
        )

    weights = _tensor_array(tensor)
    num_layers = int(model.get("num_layers", 0))
    use_vq = options.enable_vq and _is_embedding_like(tensor.name) and weights.ndim == 2 and weights.shape[1] % 8 == 0

    if use_vq:
        codes_payload, codebook_payload, vq_dim = _vq_pack(weights, vq_dim=8)
        codebook_id = f"{_safe_name(tensor.name)}-codebook"
        descriptor.update(
            {
                "dtype": "axon_vq",
                "bits": None,
                "group_size": None,
                "data_bytes": len(codes_payload),
                "scale_interleaved": False,
                "sensitivity_score": 0.95,
                "codebook_id": codebook_id,
                "vq_dim": vq_dim,
            }
        )
        data_source = _write_blob(workspace, "tensor", tensor.name, codes_payload)
        codebook_source = _write_blob(workspace, "codebook", codebook_id, codebook_payload)
        codebook_descriptor = {
            "offset": 0,
            "entries": min(256, weights.shape[0] * (weights.shape[1] // vq_dim)),
            "dim": vq_dim,
            "dtype": "fp16",
            "size": len(codebook_payload),
        }
        return descriptor, data_source, None, (codebook_id, codebook_descriptor, codebook_source)

    bits = _mxq_bits_for_tensor(tensor.name, num_layers)
    if bits is None:
        return (
            descriptor,
            {"path": str(tensor.source_path), "offset": tensor.source_offset, "length": tensor.length},
            None,
            None,
        )
    group_size = _preferred_group_size(model, options.group_size)
    use_gpu = _gpu_quantization_enabled(options, tensor, weights)
    if bits in {2, 3}:
        if use_gpu:
            packed_payload, outlier_payload, outlier_count = _nf_pack_torch(
                weights, bits, group_size, options.outlier_sigma
            )
        else:
            packed_payload, outlier_payload, outlier_count = _nf_pack(
                weights, bits, group_size, options.outlier_sigma
            )
        quant_dtype = "axon_nf2" if bits == 2 else "axon_nf3"
    else:
        if use_gpu:
            packed_payload, outlier_payload, outlier_count = _mxq_pack_torch(
                weights, bits, group_size, options.outlier_sigma
            )
        else:
            packed_payload, outlier_payload, outlier_count = _mxq_pack(
                weights, bits, group_size, options.outlier_sigma
            )
        quant_dtype = "axon_mxq"
    descriptor.update(
        {
            "dtype": quant_dtype,
            "bits": bits,
            "group_size": group_size,
            "data_bytes": len(packed_payload),
            "scale_interleaved": True,
            "sensitivity_score": _sensitivity_for_bits(bits),
            "nf_scale_fp16": bits in {2, 3},
        }
    )
    data_source = _write_blob(workspace, "tensor", tensor.name, packed_payload)
    outlier_source = None
    if outlier_payload:
        descriptor["outlier_indices_offset"] = 0
        descriptor["outlier_count"] = outlier_count
        outlier_source = _write_blob(workspace, "outlier", tensor.name, outlier_payload)
    return descriptor, data_source, outlier_source, None


def inspect_source(path: Path, source_format: str) -> dict[str, Any]:
    if source_format == "hf":
        config = _read_json(path / "config.json")
        tensors = _sort_tensors(_hf_tensors(path))
        quantizable = sum(1 for tensor in tensors if _is_quantizable_tensor(tensor))
        assets, _, _, _ = _asset_inventory(path)
        return {
            "source_format": "hf",
            "path": str(path),
            "architecture": (config.get("architectures") or [config.get("model_type", "unknown")])[0],
            "tensor_count": len(tensors),
            "tensor_dtypes": sorted({tensor.dtype for tensor in tensors}),
            "quantizable_tensors": quantizable,
            "asset_files": [asset["dest"] for asset in assets],
        }
    metadata, tensors, _, _ = _gguf_metadata(path)
    quantizable = sum(1 for tensor in tensors if _is_quantizable_tensor(tensor))
    return {
        "source_format": "gguf",
        "path": str(path),
        "architecture": metadata.get("general.architecture", "unknown"),
        "tensor_count": len(tensors),
        "tensor_dtypes": sorted({tensor.dtype for tensor in tensors}),
        "quantizable_tensors": quantizable,
        "metadata_keys": len(metadata),
    }


def build_plan(
    input_path: Path,
    source_format: str,
    workspace: Path,
    quantization: str,
    group_size: int | None,
    outlier_sigma: float,
    enable_vq: bool,
    boot_cutoff_layers: int | None = None,
    expert_dedup_threshold: float | None = None,
    speculative_draft: dict[str, Any] | None = None,
    lora_base: Path | None = None,
    lora_base_format: str | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: list[str] | None = None,
    jobs: int | None = None,
    prefer_gpu: bool = False,
) -> dict[str, Any]:
    print(
        f"[axon-pack] building plan from {input_path} ({source_format}, quantization={quantization})",
        file=sys.stderr,
        flush=True,
    )
    if lora_base is not None:
        return _build_lora_delta_plan(
            input_path=input_path,
            base_path=lora_base,
            source_format=lora_base_format or source_format,
            workspace=workspace,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    options = PackOptions(
        quantization=quantization,
        group_size=group_size,
        outlier_sigma=outlier_sigma,
        enable_vq=enable_vq,
        boot_cutoff_layers=boot_cutoff_layers,
        expert_dedup_threshold=expert_dedup_threshold,
        jobs=jobs,
        prefer_gpu=prefer_gpu,
    )

    if source_format == "hf":
        config = _read_json(input_path / "config.json")
        tensors = _sort_tensors(_hf_tensors(input_path))
        assets, tokenizer, config_file, generation_config_file = _collect_assets(input_path)
        architecture = (config.get("architectures") or [config.get("model_type", "unknown")])[0]
        model_family = config.get("model_type") or input_path.name
        model = _model_from_hf_config(config)
        runtime = _runtime_from_hf_config(input_path, config)
        source = {
            "format": "hf_safetensors",
            "identifier": str(input_path),
            "conversion_tool": "axon-pack 2.0.0",
            "conversion_time_utc": _utc_now(),
        }
    else:
        metadata, tensors, model, runtime = _gguf_metadata(input_path)
        tensors = _sort_tensors(tensors)
        assets = []
        tokenizer = None
        config_file = None
        generation_config_file = None
        architecture = str(metadata.get("general.architecture", "unknown"))
        model_family = input_path.stem
        source = {
            "format": "gguf",
            "identifier": str(input_path),
            "conversion_tool": "axon-pack 2.0.0",
            "conversion_time_utc": _utc_now(),
        }

    _attach_parameter_counts(model, tensors)
    compact_moe = None
    if isinstance(model.get("moe"), dict):
        moe = model["moe"]
        compact_moe = {
            "num_experts": int(moe.get("num_experts", 0) or 0),
            "active_experts": int(moe.get("experts_per_token", 0) or 0),
            "expert_hidden_dim": int(moe.get("expert_intermediate_dim", 0) or 0),
            "expert_similarity_dedup": False,
        }

    tensor_map: dict[str, Any] = {}
    tensor_sources: dict[str, Any] = {}
    outlier_sources: dict[str, Any] = {}
    codebooks: dict[str, Any] = {}
    codebook_sources: dict[str, Any] = {}
    expert_dedup_sources: dict[str, Any] = {}
    total_tensors = len(tensors)
    pack_jobs = _effective_pack_jobs(total_tensors, options)
    if options.prefer_gpu and not _torch_cuda_available():
        print(
            "[axon-pack] CUDA packing requested, but PyTorch/CUDA is unavailable; falling back to CPU packing",
            file=sys.stderr,
            flush=True,
        )
    if total_tensors and pack_jobs > 1:
        print(
            f"[axon-pack] packing tensors with {pack_jobs} worker(s)",
            file=sys.stderr,
            flush=True,
        )
    elif options.prefer_gpu:
        if _torch_cuda_available():
            print("[axon-pack] packing tensors with CUDA-accelerated quantization", file=sys.stderr, flush=True)

    if total_tensors == 0:
        tensor_results: dict[int, tuple[TensorSlice, dict[str, Any], dict[str, Any], dict[str, Any] | None, tuple[str, dict[str, Any], dict[str, Any]] | None]] = {}
    elif pack_jobs <= 1:
        tensor_results = {}
        for stream_order, tensor in enumerate(tensors):
            print(
                f"[axon-pack] tensor {stream_order + 1}/{total_tensors}: {tensor.name} "
                f"({tensor.dtype}, {_format_bytes(tensor.length)})",
                file=sys.stderr,
                flush=True,
            )
            descriptor, data_source, outlier_source, codebook_entry = _tensor_payload_plan(
                tensor=tensor,
                model=model,
                stream_order=stream_order,
                workspace=workspace,
                options=options,
            )
            tensor_results[stream_order] = (
                tensor,
                descriptor,
                data_source,
                outlier_source,
                codebook_entry,
            )
    else:
        tensor_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=pack_jobs) as executor:
            future_map = {
                executor.submit(
                    _tensor_payload_plan,
                    tensor=tensor,
                    model=model,
                    stream_order=stream_order,
                    workspace=workspace,
                    options=options,
                ): (stream_order, tensor)
                for stream_order, tensor in enumerate(tensors)
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_map):
                stream_order, tensor = future_map[future]
                descriptor, data_source, outlier_source, codebook_entry = future.result()
                completed += 1
                print(
                    f"[axon-pack] tensor {completed}/{total_tensors}: {tensor.name} "
                    f"({tensor.dtype}, {_format_bytes(tensor.length)})",
                    file=sys.stderr,
                    flush=True,
                )
                tensor_results[stream_order] = (
                    tensor,
                    descriptor,
                    data_source,
                    outlier_source,
                    codebook_entry,
                )

    for stream_order in range(total_tensors):
        tensor, descriptor, data_source, outlier_source, codebook_entry = tensor_results[stream_order]
        tensor_map[tensor.name] = descriptor
        tensor_sources[tensor.name] = data_source
        if outlier_source is not None:
            outlier_sources[tensor.name] = outlier_source
        if codebook_entry is not None:
            codebook_id, codebook_descriptor, codebook_source = codebook_entry
            codebooks[codebook_id] = codebook_descriptor
            codebook_sources[codebook_id] = codebook_source

    expert_dedup, expert_dedup_sources = _apply_expert_dedup(
        tensors=tensors,
        tensor_map=tensor_map,
        tensor_sources=tensor_sources,
        workspace=workspace,
        similarity_threshold=options.expert_dedup_threshold if model.get("moe") else None,
    )

    boot_cutoff = _boot_cutoff_stream_order(tensors, int(model.get("num_layers", 0) or 0), options.boot_cutoff_layers)

    manifest: dict[str, Any] = {
        "manifest_version": "2.0.0-draft",
        "task": "causal_lm",
        "model_file": "model.axon",
    }
    if config_file:
        manifest["config_file"] = config_file
    if generation_config_file:
        manifest["generation_config_file"] = generation_config_file
    if tokenizer:
        manifest["tokenizer"] = tokenizer

    use_quantization = quantization != "none"
    metadata = {
        "format": "axon",
        "version": "2.0.0-draft",
        "task": "causal_lm",
        "architecture": architecture,
        "model_family": model_family,
        "model": model,
        "runtime": runtime,
        "source": source,
        "total_params": model.get("total_parameter_count"),
        "active_params": model.get("active_parameter_count"),
        "moe": compact_moe,
        "boot_region_bytes": 0,
        "kv_cache_hints": None,
        "tensor_dep_graph": _dependency_graph_from_tensors(tensors),
        "speculative_draft": None
        if speculative_draft is None
        else {
            "arch": speculative_draft["arch"],
            "hidden_dim": speculative_draft["hidden_dim"],
            "num_layers": speculative_draft["num_layers"],
            "vocab_size": speculative_draft["vocab_size"],
            "draft_bytes": int(speculative_draft["length"]),
            "draft_offset": 0,
        },
        "smoothquant_scales": {},
        "expert_dedup": expert_dedup,
        "lora": None,
        "numa_hints": None,
        "avg_bits_per_weight": 0.0,
        "quant_method": "axon-v2-mixed" if use_quantization else "none",
        "calibration": None,
        "tensors": tensor_map,
        "codebooks": codebooks,
        "hw_hints": {
            "cuda_sm90": {"kernel": "axonal_cuda_mxq_sm90", "tile": [128, 128, 64], "unroll": 4},
            "cuda_sm89": {"kernel": "axonal_cuda_mxq_sm89", "tile": [128, 128, 64], "unroll": 4},
            "cuda_sm80": {"kernel": "axonal_cuda_mxq_sm80", "tile": [128, 128, 64], "unroll": 2},
            "cpu_avx512_vnni": {"kernel": "axonal_cpu_mxq_avx512"},
            "cpu_neon": {"kernel": "axonal_cpu_mxq_neon"},
        }
        if use_quantization
        else {},
    }

    print(
        f"[axon-pack] plan complete: {total_tensors} tensors, {len(assets)} copied asset(s)",
        file=sys.stderr,
        flush=True,
    )

    return {
        "manifest": manifest,
        "metadata": metadata,
        "boot_cutoff": boot_cutoff,
        "tensor_sources": tensor_sources,
        "outlier_sources": outlier_sources,
        "codebook_sources": codebook_sources,
        "expert_dedup_sources": expert_dedup_sources,
        "speculative_draft_source": None
        if speculative_draft is None
        else {
            "path": speculative_draft["path"],
            "offset": 0,
            "length": int(speculative_draft["length"]),
        },
        "lora_sources": {},
        "assets": assets,
    }
