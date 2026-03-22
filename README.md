# axon-pack

`axon-pack` converts source models into AXON bundles.

It is split into two parts:

- a Python frontend for source import, normalization, and install-path handling
- a Rust core for AXON bundle writing and validation

By default, packed bundles install into the same local model library that `axonal` uses:

- `$AXONAL_MODELS` when set
- otherwise `~/.axonal/models`

## What It Supports

- local Hugging Face-style model directories with `config.json` and `safetensors`
- safetensors shard indexes via `*.safetensors.index.json`
- local GGUF inputs
- AXON v2 bundle metadata
- mixed scalar quantization with `axon_mxq`
- NF quantization with `axon_nf2` and `axon_nf3`
- outlier spine emission
- VQ codebooks for embedding-like tensors
- boot/tail split metadata
- MoE metadata and active-parameter metadata
- expert deduplication regions
- appended speculative draft bundles
- LoRA delta bundle emission against a base HF model

## Installation

### Python Frontend

The Python CLI is the main entrypoint today:

```bash
PYTHONPATH=python python3 -m axon_pack.cli --help
```

### Rust Core

The Rust writer/validator binary can also be installed directly:

```bash
cargo install --path . --bin axon-pack-rs
```

## Quickstart

Pack a local Hugging Face model into the shared AXON library:

```bash
PYTHONPATH=python python3 -m axon_pack.cli pack \
  --input /path/to/model \
  --source-format hf \
  --name my-model
```

Pack to an explicit output directory:

```bash
PYTHONPATH=python python3 -m axon_pack.cli pack \
  --input /path/to/model \
  --source-format hf \
  --output /tmp/my-model-axon
```

Pack a GGUF source:

```bash
PYTHONPATH=python python3 -m axon_pack.cli pack \
  --input /path/to/model.gguf \
  --source-format gguf \
  --name my-gguf-model
```

Emit a LoRA delta bundle:

```bash
PYTHONPATH=python python3 -m axon_pack.cli pack \
  --input /path/to/fine-tuned-model \
  --source-format hf \
  --lora-base /path/to/base-model \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-targets q_proj,k_proj,v_proj,o_proj \
  --name my-model-lora
```

Emit a speculative-draft bundle:

```bash
PYTHONPATH=python python3 -m axon_pack.cli pack \
  --input /path/to/main-model \
  --source-format hf \
  --speculative-draft /path/to/draft-model \
  --name my-model-with-draft
```

## Commands

- `inspect`: inspect a source model before packing
- `pack`: convert a source model into an AXON bundle
- `validate`: validate an existing AXON bundle
- `inspect-bundle`: print metadata from an AXON bundle

More detail: [docs/packing.md](/home/oscar/Repos/axon-pack/docs/packing.md)

## Output Layout

A packed bundle directory contains:

- `manifest.json`
- `model.axon`
- tokenizer and config assets when available

## Current Limits

- remote model download is handled by `axonal`, not `axon-pack`
- GGUF import support is still narrower than HF import support
- custom Hugging Face Python code is not executed
- the high-level `axon-pack` frontend is Python-first today

## Development

```bash
cargo test
PYTHONPATH=python python3 -m unittest discover -s tests -v
```
