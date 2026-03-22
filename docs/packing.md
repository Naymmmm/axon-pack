# axon-pack Packing Guide

## Source Formats

`axon-pack` accepts:

- `--source-format hf` for local Hugging Face model directories
- `--source-format gguf` for local GGUF files

For Hugging Face directories, the importer uses:

- `config.json`
- `generation_config.json`
- tokenizer assets
- `*.safetensors.index.json` when present
- the safetensors shard files referenced by the index

It also preserves auxiliary assets such as preprocessor and chat-template files inside the emitted bundle.

## Install Targets

If you omit `--output`, `pack` installs into:

- `$AXONAL_MODELS`, or
- `~/.axonal/models`

Use:

- `--name` to control the installed bundle name
- `--library` to override the library root
- `--replace` to overwrite an existing bundle

## Quantization

Supported `--quantization` modes:

- `none`: float passthrough
- `auto`: VQ for embedding-like tensors and scalar quantization for eligible linear weights
- `mxq`: scalar quantization for all eligible quantizable weights

Useful related flags:

- `--group-size`
- `--outlier-sigma`
- `--no-vq`
- `--jobs` (default `6`)
- GPU packing is preferred by default when PyTorch/CUDA is available
- `--no-gpu`

## AXON v2 Features

Current `pack` options also cover several AXON v2 bundle features:

- `--boot-cutoff-layers`: write boot/tail split metadata
- `--expert-dedup-threshold`: emit expert dedup regions for similar MoE experts
- `--speculative-draft`: append a draft AXON bundle
- `--lora-base`, `--lora-rank`, `--lora-alpha`, `--lora-targets`: emit a LoRA delta bundle

## Validation

Validate a bundle after packing:

```bash
PYTHONPATH=python python3 -m axon_pack.cli validate /path/to/bundle
```

Inspect a produced bundle:

```bash
PYTHONPATH=python python3 -m axon_pack.cli inspect-bundle /path/to/bundle
```

## Notes

- The Rust binary owns final AXON writing and structural validation.
- The Python frontend owns source loading, tensor normalization, and bundle installation.
- `axonal convert` is the preferred path when you want remote Hugging Face download plus installation into the local model library in one command.
