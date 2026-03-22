from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHONPATH = str(ROOT / "python")


def run_cli(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["python3", "-m", "axon_pack.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def write_safetensors(path: Path) -> None:
    header = {
        "weight": {
            "dtype": "F16",
            "shape": [2, 2],
            "data_offsets": [0, 8],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = bytes([1, 0, 2, 0, 3, 0, 4, 0])
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)


def write_named_safetensors(path: Path, tensors: dict[str, dict[str, object]], payload: bytes) -> None:
    header_bytes = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def gguf_kv(key: str, value_type: int, value: bytes) -> bytes:
    return gguf_string(key) + struct.pack("<I", value_type) + value


def write_gguf(path: Path) -> None:
    metadata_entries = [
        gguf_kv("general.architecture", 8, gguf_string("llama")),
        gguf_kv("general.alignment", 4, struct.pack("<I", 32)),
        gguf_kv("llama.embedding_length", 4, struct.pack("<I", 4)),
        gguf_kv("llama.feed_forward_length", 4, struct.pack("<I", 8)),
        gguf_kv("llama.block_count", 4, struct.pack("<I", 1)),
        gguf_kv("llama.attention.head_count", 4, struct.pack("<I", 1)),
        gguf_kv("llama.attention.head_count_kv", 4, struct.pack("<I", 1)),
        gguf_kv("llama.context_length", 4, struct.pack("<I", 128)),
        gguf_kv("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 1)),
        gguf_kv("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 2)),
    ]
    tensor_info = (
        gguf_string("tok_embeddings.weight")
        + struct.pack("<I", 2)
        + struct.pack("<Q", 4)
        + struct.pack("<Q", 4)
        + struct.pack("<I", 0)
        + struct.pack("<Q", 0)
    )
    payload = bytes(range(64))
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<I", 3))
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<Q", len(metadata_entries)))
        for entry in metadata_entries:
            handle.write(entry)
        handle.write(tensor_info)
        padding = (-handle.tell()) % 32
        handle.write(b"\x00" * padding)
        handle.write(payload)


def read_axon_header(path: Path) -> dict[str, int]:
    data = path.read_bytes()[:64]
    return {
        "version_major": int.from_bytes(data[4:6], "little"),
        "version_minor": int.from_bytes(data[6:8], "little"),
        "flags": int.from_bytes(data[8:12], "little"),
        "header_len": int.from_bytes(data[12:16], "little"),
        "data_offset": int.from_bytes(data[16:24], "little"),
        "outlier_offset": int.from_bytes(data[24:32], "little"),
        "codebook_offset": int.from_bytes(data[32:40], "little"),
        "tail_offset": int.from_bytes(data[40:48], "little"),
        "boot_cutoff": data[48],
        "speculative_offset": int.from_bytes(data[52:60], "little"),
    }


class PackTests(unittest.TestCase):
    def test_pack_hf_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axon-hf-") as temp_dir:
            source = Path(temp_dir) / "hf-model"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "hidden_size": 4,
                        "intermediate_size": 8,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 1,
                        "num_key_value_heads": 1,
                        "max_position_embeddings": 128,
                        "vocab_size": 32,
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                        "pad_token_id": 0,
                    }
                ),
                encoding="utf-8",
            )
            write_safetensors(source / "model.safetensors")
            output = Path(temp_dir) / "bundle"
            run_cli("pack", "--input", str(source), "--source-format", "hf", "--output", str(output))
            validate = run_cli("validate", "--bundle", str(output))
            summary = json.loads(validate.stdout)
            self.assertEqual(summary["metadata"]["task"], "causal_lm")
            self.assertEqual(summary["manifest"]["manifest_version"], "2.0.0-draft")
            self.assertEqual(summary["metadata"]["version"], "2.0.0-draft")
            self.assertTrue((output / "model.axon").exists())
            self.assertTrue((output / "manifest.json").exists())
            self.assertEqual(summary["metadata"]["tensors"]["weight"]["dtype"], "axon_mxq")

    def test_pack_defaults_to_library_install(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axon-hf-library-") as temp_dir:
            source = Path(temp_dir) / "hf-model"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "hidden_size": 4,
                        "intermediate_size": 8,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 1,
                        "num_key_value_heads": 1,
                        "max_position_embeddings": 128,
                        "vocab_size": 32,
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                        "pad_token_id": 0,
                    }
                ),
                encoding="utf-8",
            )
            write_safetensors(source / "model.safetensors")
            library = Path(temp_dir) / "library"
            run_cli(
                "pack",
                "--input",
                str(source),
                "--source-format",
                "hf",
                env_extra={"AXONAL_MODELS": str(library)},
            )
            output = library / "hf-model"
            summary = json.loads(run_cli("inspect-bundle", "--bundle", str(output)).stdout)
            self.assertTrue((output / "model.axon").exists())
            self.assertEqual(summary["metadata"]["architecture"], "LlamaForCausalLM")

    def test_pack_gguf_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axon-gguf-") as temp_dir:
            source = Path(temp_dir) / "toy.gguf"
            write_gguf(source)
            output = Path(temp_dir) / "bundle"
            run_cli("pack", "--input", str(source), "--source-format", "gguf", "--output", str(output))
            inspect = run_cli("inspect-bundle", "--bundle", str(output))
            summary = json.loads(inspect.stdout)
            self.assertEqual(summary["metadata"]["source"]["format"], "gguf")
            self.assertEqual(summary["metadata"]["model"]["num_layers"], 1)

    def test_pack_hf_vq_and_outliers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axon-hf-vq-") as temp_dir:
            source = Path(temp_dir) / "hf-model"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "hidden_size": 8,
                        "intermediate_size": 128,
                        "num_hidden_layers": 4,
                        "num_attention_heads": 1,
                        "num_key_value_heads": 1,
                        "max_position_embeddings": 128,
                        "vocab_size": 32,
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                        "pad_token_id": 0,
                    }
                ),
                encoding="utf-8",
            )
            header = {
                "model.embed_tokens.weight": {
                    "dtype": "F16",
                    "shape": [4, 8],
                    "data_offsets": [0, 64],
                },
                "model.layers.1.mlp.down_proj.weight": {
                    "dtype": "F16",
                    "shape": [2, 64],
                    "data_offsets": [64, 320],
                },
            }
            header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
            embed = bytes(range(64))
            down = bytearray(256)
            down[0:2] = b"\xff\x7b"
            payload = embed + bytes(down)
            with (source / "model.safetensors").open("wb") as handle:
                handle.write(struct.pack("<Q", len(header_bytes)))
                handle.write(header_bytes)
                handle.write(payload)
            output = Path(temp_dir) / "bundle"
            run_cli("pack", "--input", str(source), "--source-format", "hf", "--output", str(output))
            summary = json.loads(run_cli("inspect-bundle", "--bundle", str(output)).stdout)
            tensors = summary["metadata"]["tensors"]
            self.assertEqual(tensors["model.embed_tokens.weight"]["dtype"], "axon_vq")
            self.assertTrue(summary["metadata"]["codebooks"])
            self.assertEqual(tensors["model.layers.1.mlp.down_proj.weight"]["dtype"], "axon_nf2")
            self.assertEqual(tensors["model.layers.1.mlp.down_proj.weight"]["bits"], 2)
            self.assertTrue(tensors["model.layers.1.mlp.down_proj.weight"]["nf_scale_fp16"])
            self.assertGreater(tensors["model.layers.1.mlp.down_proj.weight"]["outlier_count"], 0)

    def test_pack_hf_moe_active_parameter_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axon-hf-moe-") as temp_dir:
            source = Path(temp_dir) / "hf-model"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2_moe",
                        "architectures": ["Qwen2MoeForCausalLM"],
                        "hidden_size": 4,
                        "intermediate_size": 8,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 1,
                        "num_key_value_heads": 1,
                        "max_position_embeddings": 128,
                        "vocab_size": 64,
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                        "pad_token_id": 0,
                        "num_local_experts": 2,
                        "num_experts_per_tok": 1,
                        "moe_intermediate_size": 6,
                        "num_shared_experts": 1,
                        "shared_expert_intermediate_size": 8,
                        "router_aux_loss_coef": 0.01,
                    }
                ),
                encoding="utf-8",
            )
            header = {
                "model.embed_tokens.weight": {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]},
                "model.layers.0.self_attn.q_proj.weight": {"dtype": "F16", "shape": [4, 4], "data_offsets": [32, 64]},
                "model.layers.0.mlp.router.weight": {"dtype": "F16", "shape": [2, 4], "data_offsets": [64, 80]},
                "model.layers.0.mlp.shared_expert.gate_proj.weight": {
                    "dtype": "F16",
                    "shape": [8, 4],
                    "data_offsets": [80, 144],
                },
                "model.layers.0.mlp.experts.0.gate_proj.weight": {
                    "dtype": "F16",
                    "shape": [6, 4],
                    "data_offsets": [144, 192],
                },
                "model.layers.0.mlp.experts.1.gate_proj.weight": {
                    "dtype": "F16",
                    "shape": [6, 4],
                    "data_offsets": [192, 240],
                },
            }
            header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
            payload = bytes(range(240))
            with (source / "model.safetensors").open("wb") as handle:
                handle.write(struct.pack("<Q", len(header_bytes)))
                handle.write(header_bytes)
                handle.write(payload)
            output = Path(temp_dir) / "bundle"
            run_cli("pack", "--input", str(source), "--source-format", "hf", "--output", str(output))
            summary = json.loads(run_cli("inspect-bundle", "--bundle", str(output)).stdout)
            model = summary["metadata"]["model"]
            self.assertEqual(model["moe"]["num_experts"], 2)
            self.assertEqual(model["moe"]["experts_per_token"], 1)
            self.assertEqual(model["moe"]["num_shared_experts"], 1)
            self.assertEqual(model["total_parameter_count"], 120)
            self.assertEqual(model["active_parameter_count"], 96)

    def test_pack_hf_preserves_extended_assets_and_uses_index(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axon-hf-assets-") as temp_dir:
            source = Path(temp_dir) / "hf-model"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "hidden_size": 4,
                        "intermediate_size": 8,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 1,
                        "num_key_value_heads": 1,
                        "max_position_embeddings": 128,
                        "vocab_size": 32,
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                        "pad_token_id": 0,
                    }
                ),
                encoding="utf-8",
            )
            (source / "generation_config.json").write_text(
                json.dumps({"temperature": 0.7, "top_p": 0.9, "top_k": 32, "max_new_tokens": 64}),
                encoding="utf-8",
            )
            (source / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>"}), encoding="utf-8")
            (source / "special_tokens_map.json").write_text(json.dumps({"bos_token": "<s>"}), encoding="utf-8")
            (source / "added_tokens.json").write_text(json.dumps({"<extra>": 32000}), encoding="utf-8")
            (source / "vocab.json").write_text(json.dumps({"hello": 0, "world": 1}), encoding="utf-8")
            (source / "merges.txt").write_text("#version: 0.2\nh e\n", encoding="utf-8")
            (source / "chat_template.jinja").write_text("{{ messages[0].content }}", encoding="utf-8")
            (source / "preprocessor_config.json").write_text(json.dumps({"do_resize": False}), encoding="utf-8")
            (source / "video_preprocessor_config.json").write_text(
                json.dumps({"num_frames": 8}),
                encoding="utf-8",
            )

            shard_header = {
                "weight": {
                    "dtype": "F16",
                    "shape": [2, 2],
                    "data_offsets": [0, 8],
                }
            }
            write_named_safetensors(source / "model-00001-of-00002.safetensors", shard_header, bytes([1, 0, 2, 0, 3, 0, 4, 0]))
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": {"weight": "model-00001-of-00002.safetensors"}}),
                encoding="utf-8",
            )
            (source / "model-00002-of-00002.safetensors").write_text("not-a-safetensors-file", encoding="utf-8")

            output = Path(temp_dir) / "bundle"
            run_cli("pack", "--input", str(source), "--source-format", "hf", "--output", str(output))

            summary = json.loads(run_cli("inspect-bundle", "--bundle", str(output)).stdout)
            tokenizer_files = summary["manifest"]["tokenizer"]["files"]
            self.assertIn("tokenizer_config.json", tokenizer_files)
            self.assertIn("chat_template.jinja", tokenizer_files)
            self.assertIn("vocab.json", tokenizer_files)
            self.assertIn("merges.txt", tokenizer_files)
            self.assertTrue((output / "preprocessor_config.json").exists())
            self.assertTrue((output / "video_preprocessor_config.json").exists())
            self.assertTrue((output / "chat_template.jinja").exists())

    def test_pack_hf_emits_boot_split_and_speculative_draft(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axon-hf-draft-") as temp_dir:
            source = Path(temp_dir) / "main"
            draft = Path(temp_dir) / "draft"
            source.mkdir()
            draft.mkdir()
            base_config = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 4,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "vocab_size": 32,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
            }
            draft_config = {
                **base_config,
                "hidden_size": 2,
                "intermediate_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 16,
            }
            (source / "config.json").write_text(json.dumps(base_config), encoding="utf-8")
            (draft / "config.json").write_text(json.dumps(draft_config), encoding="utf-8")
            main_header = {
                "model.layers.0.self_attn.q_proj.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
                "model.layers.1.self_attn.q_proj.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [8, 16]},
                "model.layers.2.self_attn.q_proj.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [16, 24]},
                "model.layers.3.self_attn.q_proj.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [24, 32]},
            }
            draft_header = {
                "model.layers.0.self_attn.q_proj.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
            }
            write_named_safetensors(source / "model.safetensors", main_header, bytes(range(32)))
            write_named_safetensors(draft / "model.safetensors", draft_header, bytes(range(8)))
            output = Path(temp_dir) / "bundle"
            run_cli(
                "pack",
                "--input",
                str(source),
                "--source-format",
                "hf",
                "--output",
                str(output),
                "--boot-cutoff-layers",
                "1",
                "--speculative-draft",
                str(draft),
            )
            summary = json.loads(run_cli("inspect-bundle", "--bundle", str(output)).stdout)
            header = read_axon_header(output / "model.axon")
            self.assertEqual(header["version_major"], 2)
            self.assertGreater(header["tail_offset"], 0)
            self.assertGreater(header["speculative_offset"], 0)
            self.assertEqual(header["boot_cutoff"], 1)
            self.assertGreater(summary["metadata"]["boot_region_bytes"], 0)
            self.assertIsNotNone(summary["metadata"]["speculative_draft"])
            self.assertGreater(summary["metadata"]["speculative_draft"]["draft_bytes"], 0)

    def test_pack_hf_emits_expert_dedup_region(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axon-hf-dedup-") as temp_dir:
            source = Path(temp_dir) / "hf-model"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2_moe",
                        "architectures": ["Qwen2MoeForCausalLM"],
                        "hidden_size": 4,
                        "intermediate_size": 8,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 1,
                        "num_key_value_heads": 1,
                        "max_position_embeddings": 128,
                        "vocab_size": 32,
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                        "pad_token_id": 0,
                        "num_local_experts": 2,
                        "num_experts_per_tok": 1,
                        "moe_intermediate_size": 4,
                    }
                ),
                encoding="utf-8",
            )
            header = {
                "model.layers.0.mlp.experts.0.gate_proj.weight": {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]},
                "model.layers.0.mlp.experts.1.gate_proj.weight": {"dtype": "F16", "shape": [4, 4], "data_offsets": [32, 64]},
            }
            first = bytearray(32)
            first[0:2] = b"\x00\x3c"
            second = bytearray(first)
            second[-2:] = b"\x00\x2c"
            payload = bytes(first + second)
            write_named_safetensors(source / "model.safetensors", header, payload)
            output = Path(temp_dir) / "bundle"
            run_cli("pack", "--input", str(source), "--source-format", "hf", "--output", str(output))
            summary = json.loads(run_cli("inspect-bundle", "--bundle", str(output)).stdout)
            dedup = summary["metadata"]["expert_dedup"]
            self.assertIsNotNone(dedup)
            tensor = summary["metadata"]["tensors"]["model.layers.0.mlp.experts.1.gate_proj.weight"]
            self.assertEqual(tensor["dedup_canonical"], "model.layers.0.mlp.experts.0.gate_proj.weight")
            self.assertEqual(tensor["data_bytes"], 0)
            self.assertGreater(tensor["dedup_correction_count"], 0)

    def test_pack_hf_emits_lora_delta_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axon-hf-lora-") as temp_dir:
            base = Path(temp_dir) / "base"
            fine = Path(temp_dir) / "fine"
            base.mkdir()
            fine.mkdir()
            config = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "vocab_size": 32,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
            }
            (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (fine / "config.json").write_text(json.dumps(config), encoding="utf-8")
            header = {
                "model.layers.0.self_attn.q_proj.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
                "model.layers.0.self_attn.k_proj.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [8, 16]},
            }
            write_named_safetensors(base / "model.safetensors", header, bytes([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            write_named_safetensors(fine / "model.safetensors", header, bytes([0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            output = Path(temp_dir) / "bundle"
            run_cli(
                "pack",
                "--input",
                str(fine),
                "--source-format",
                "hf",
                "--output",
                str(output),
                "--lora-base",
                str(base),
                "--lora-rank",
                "1",
            )
            summary = json.loads(run_cli("inspect-bundle", "--bundle", str(output)).stdout)
            self.assertIsNotNone(summary["metadata"]["lora"])
            self.assertIn("model.layers.0.self_attn.q_proj.weight.lora_A", summary["metadata"]["tensors"])
            self.assertIn("model.layers.0.self_attn.q_proj.weight.lora_B", summary["metadata"]["tensors"])


if __name__ == "__main__":
    unittest.main()
