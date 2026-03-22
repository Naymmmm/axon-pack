from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .sources import build_plan, inspect_source


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _binary_is_current(binary: Path, repo_root: Path) -> bool:
    if not binary.is_file():
        return False
    binary_mtime = binary.stat().st_mtime
    watched_paths = [repo_root / "Cargo.toml", *sorted((repo_root / "src").glob("**/*.rs"))]
    return all(not path.exists() or path.stat().st_mtime <= binary_mtime for path in watched_paths)


def _rust_invocation() -> list[str]:
    repo_root = _repo_root()
    configured = os.environ.get("AXON_PACK_RS_BIN")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return [str(candidate)]
    for candidate in [repo_root / "target" / "debug" / "axon-pack-rs", repo_root / "target" / "release" / "axon-pack-rs"]:
        if _binary_is_current(candidate, repo_root):
            return [str(candidate)]
    cargo = shutil.which("cargo")
    if cargo:
        return [
            cargo,
            "run",
            "--quiet",
            "--manifest-path",
            str(repo_root / "Cargo.toml"),
            "--bin",
            "axon-pack-rs",
            "--",
        ]
    which_binary = shutil.which("axon-pack-rs")
    if which_binary:
        return [which_binary]
    raise RuntimeError("could not locate axon-pack-rs or cargo; set AXON_PACK_RS_BIN or install axon-pack-rs")


def _run_rust(args: list[str]) -> int:
    command = _rust_invocation() + args
    completed = subprocess.run(command)
    return completed.returncode


def _default_library_dir() -> Path:
    configured = os.environ.get("AXONAL_MODELS")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".axonal" / "models").resolve()


def _normalize_model_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-._")
    return cleaned.lower() or "model"


def _infer_model_name(input_path: Path) -> str:
    if input_path.is_dir():
        return _normalize_model_name(input_path.name)
    return _normalize_model_name(input_path.stem)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()
    library_root = Path(args.library).expanduser().resolve() if args.library else _default_library_dir()
    model_name = _normalize_model_name(args.name or _infer_model_name(Path(args.input).resolve()))
    return library_root / model_name


def _prepare_output_dir(output_dir: Path, replace: bool) -> None:
    if output_dir.exists():
        if not replace:
            raise RuntimeError(
                f"output directory already exists: {output_dir}. Pass --replace to overwrite it."
            )
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.parent.mkdir(parents=True, exist_ok=True)


def _infer_source_format(path: Path) -> str:
    if path.is_dir():
        return "hf"
    if path.suffix.lower() == ".gguf":
        return "gguf"
    raise RuntimeError(f"could not infer source format for {path}; pass an explicit source-format option")


def _prepare_speculative_draft(args: argparse.Namespace, workspace: Path) -> dict[str, object] | None:
    if not args.speculative_draft:
        return None
    draft_input = Path(args.speculative_draft).expanduser().resolve()
    draft_source_format = args.speculative_source_format or _infer_source_format(draft_input)
    draft_workspace = workspace / "speculative-draft"
    draft_workspace.mkdir(parents=True, exist_ok=True)
    draft_plan = build_plan(
        input_path=draft_input,
        source_format=draft_source_format,
        workspace=draft_workspace,
        quantization=args.quantization,
        group_size=args.group_size,
        outlier_sigma=args.outlier_sigma,
        enable_vq=not args.no_vq,
        boot_cutoff_layers=args.boot_cutoff_layers,
        expert_dedup_threshold=None,
    )
    draft_plan_path = draft_workspace / "draft-plan.json"
    draft_plan_path.write_text(json.dumps(draft_plan, indent=2), encoding="utf-8")
    draft_output = draft_workspace / "bundle"
    exit_code = _run_rust(["build", "--plan", str(draft_plan_path), "--output", str(draft_output)])
    if exit_code != 0:
        raise RuntimeError("failed to build speculative draft bundle")
    draft_model = draft_output / draft_plan["manifest"]["model_file"]
    draft_metadata = draft_plan["metadata"]
    return {
        "path": str(draft_model),
        "length": draft_model.stat().st_size,
        "arch": draft_metadata["architecture"],
        "hidden_dim": draft_metadata["model"]["hidden_dim"],
        "num_layers": draft_metadata["model"]["num_layers"],
        "vocab_size": draft_metadata["model"]["vocab_size"],
    }


def cmd_inspect(args: argparse.Namespace) -> int:
    summary = inspect_source(Path(args.path), args.source_format)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    output_dir = _resolve_output_dir(args)
    _prepare_output_dir(output_dir, args.replace)
    print(
        f"[axon-pack] packing {Path(args.input).resolve()} -> {output_dir}",
        file=sys.stderr,
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="axon-pack-") as temp_dir:
        workspace = Path(temp_dir)
        draft = _prepare_speculative_draft(args, workspace)
        plan = build_plan(
            input_path=Path(args.input).resolve(),
            source_format=args.source_format,
            workspace=workspace,
            quantization=args.quantization,
            group_size=args.group_size,
            outlier_sigma=args.outlier_sigma,
            enable_vq=not args.no_vq,
            boot_cutoff_layers=args.boot_cutoff_layers,
            expert_dedup_threshold=args.expert_dedup_threshold,
            speculative_draft=draft,
            lora_base=Path(args.lora_base).expanduser().resolve() if args.lora_base else None,
            lora_base_format=args.lora_base_format,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=[value.strip() for value in args.lora_targets.split(",") if value.strip()]
            if args.lora_targets
            else None,
        )
        plan_path = workspace / "build-plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"[axon-pack] writing bundle to {output_dir}", file=sys.stderr, flush=True)
        exit_code = _run_rust(["build", "--plan", str(plan_path), "--output", str(output_dir)])
        if exit_code == 0:
            print(f"installed bundle at {output_dir}", file=sys.stderr)
        return exit_code


def cmd_validate(args: argparse.Namespace) -> int:
    return _run_rust(["validate", "--bundle", str(Path(args.bundle).resolve())])


def cmd_inspect_bundle(args: argparse.Namespace) -> int:
    return _run_rust(["inspect", "--bundle", str(Path(args.bundle).resolve())])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="axon-pack", description="AXON bundle packer")
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subcommands.add_parser("inspect", help="Inspect a source model")
    inspect_parser.add_argument("path")
    inspect_parser.add_argument("--source-format", choices=["hf", "gguf"], required=True)
    inspect_parser.set_defaults(func=cmd_inspect)

    pack_parser = subcommands.add_parser("pack", help="Pack a source model into an AXON bundle")
    pack_parser.add_argument("--input", required=True)
    pack_parser.add_argument("--source-format", choices=["hf", "gguf"], required=True)
    pack_parser.add_argument(
        "--output",
        help="Explicit bundle output directory. Defaults to the shared local model library.",
    )
    pack_parser.add_argument(
        "--library",
        help="Override the local model library root used when --output is omitted.",
    )
    pack_parser.add_argument(
        "--name",
        help="Install name to use inside the local model library. Defaults to the source directory or file stem.",
    )
    pack_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing bundle at the destination path.",
    )
    pack_parser.add_argument(
        "--quantization",
        choices=["none", "auto", "mxq"],
        default="auto",
        help="none = passthrough floats, auto = VQ embeddings + MXQ linear weights, mxq = MXQ all eligible weights",
    )
    pack_parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Override MXQ group size. Defaults to 128 or 64 for narrow models.",
    )
    pack_parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=6.0,
        help="Outlier threshold multiplier in |w| > |mu| + sigma*std per output row.",
    )
    pack_parser.add_argument(
        "--no-vq",
        action="store_true",
        help="Disable VQ for embedding and lm_head tensors.",
    )
    pack_parser.add_argument(
        "--boot-cutoff-layers",
        type=int,
        default=None,
        help="Boot region layer cutoff. Defaults to the first quarter of layers when omitted.",
    )
    pack_parser.add_argument(
        "--expert-dedup-threshold",
        type=float,
        default=0.97,
        help="Cosine similarity threshold for MoE expert deduplication. Applied only to split expert tensors.",
    )
    pack_parser.add_argument(
        "--speculative-draft",
        help="Optional secondary model source to append as a speculative draft bundle.",
    )
    pack_parser.add_argument(
        "--speculative-source-format",
        choices=["hf", "gguf"],
        default=None,
        help="Source format for --speculative-draft when it cannot be inferred.",
    )
    pack_parser.add_argument(
        "--lora-base",
        help="Optional base Hugging Face model directory to emit a LoRA delta bundle instead of a full bundle.",
    )
    pack_parser.add_argument(
        "--lora-base-format",
        choices=["hf", "gguf"],
        default=None,
        help="Source format for --lora-base. Defaults to the main source format.",
    )
    pack_parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="Rank to use when factorizing LoRA deltas.",
    )
    pack_parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="Alpha metadata to store for emitted LoRA delta tensors.",
    )
    pack_parser.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target module substrings for LoRA delta extraction.",
    )
    pack_parser.set_defaults(func=cmd_pack)

    validate_parser = subcommands.add_parser("validate", help="Validate an AXON bundle")
    validate_parser.add_argument("--bundle", required=True)
    validate_parser.set_defaults(func=cmd_validate)

    inspect_bundle_parser = subcommands.add_parser("inspect-bundle", help="Inspect an AXON bundle")
    inspect_bundle_parser.add_argument("--bundle", required=True)
    inspect_bundle_parser.set_defaults(func=cmd_inspect_bundle)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
