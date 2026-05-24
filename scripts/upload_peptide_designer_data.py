#!/usr/bin/env python3
"""Upload the structured peptide designer landscape bundle to Hugging Face."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO_NAME = "peptide_designer_data"
DEFAULT_BUNDLE_ROOT = Path("output/hf/peptide_designer_data")
RAW_DATA_DIR = Path("data")
FORBIDDEN_NAMES = {"source_peptides.parquet"}
LEGACY_REMOTE_DELETE_PATTERNS = ["models/**", "output/**"]


@dataclass(frozen=True)
class ManifestEntry:
    source: Path
    destination: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the structured peptide designer dataset bundle."
    )
    parser.add_argument(
        "--repo-id",
        help="Full Hugging Face dataset repo id, e.g. user/peptide_designer_data.",
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("HF_NAMESPACE"),
        help="Hugging Face namespace. Defaults to HF_NAMESPACE.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Local structured bundle root containing README.md and landscapes/.",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        action="store_true",
        dest="private",
        help="Create/upload to a private dataset repo. Default is public.",
    )
    visibility.add_argument(
        "--public",
        action="store_false",
        dest="private",
        help="Create/upload to a public dataset repo. This is the default.",
    )
    parser.set_defaults(private=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest without uploading. This is the default.",
    )
    mode.add_argument(
        "--upload",
        action="store_true",
        help="Create the dataset repo if needed and upload the bundle.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload structured peptide GTM landscape bundle",
        help="Commit message for Hugging Face upload.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def import_hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run `uv sync --extra dev` first."
        ) from exc
    return HfApi


def infer_namespace(token: str) -> str:
    HfApi = import_hf_api()
    whoami = HfApi(token=token).whoami()
    namespace = whoami.get("name")
    if not namespace:
        raise SystemExit("Could not infer Hugging Face namespace from the token.")
    return namespace


def resolve_repo_id(args: argparse.Namespace) -> str:
    if args.repo_id:
        return args.repo_id
    if args.namespace:
        return f"{args.namespace}/{DEFAULT_REPO_NAME}"
    if args.upload:
        token = get_hf_token()
        if not token:
            raise SystemExit(
                "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running with --upload."
            )
        return f"{infer_namespace(token)}/{DEFAULT_REPO_NAME}"
    return f"<HF_NAMESPACE>/{DEFAULT_REPO_NAME}"


def assert_safe_path(path: Path) -> None:
    if path.parts and path.parts[0] == RAW_DATA_DIR.name:
        raise SystemExit(f"Refusing to include raw data path: {path}")
    if ".ipynb_checkpoints" in path.parts:
        raise SystemExit(f"Refusing to include notebook checkpoint: {path}")
    if path.name in FORBIDDEN_NAMES:
        raise SystemExit(f"Refusing to include row-level source table: {path}")
    if path.suffix.lower() == ".csv":
        raise SystemExit(f"Refusing to include CSV file: {path}")


def build_manifest(root: Path, bundle_root: Path) -> list[ManifestEntry]:
    bundle_root = (root / bundle_root).resolve()
    if not bundle_root.exists():
        raise SystemExit(f"Bundle root does not exist: {bundle_root}")
    if not (bundle_root / "README.md").exists():
        raise SystemExit(f"Bundle root is missing README.md: {bundle_root}")
    if not (bundle_root / "landscapes").exists():
        raise SystemExit(f"Bundle root is missing landscapes/: {bundle_root}")

    entries: list[ManifestEntry] = []
    for source in sorted(path for path in bundle_root.rglob("*") if path.is_file()):
        relative_path = source.relative_to(bundle_root)
        assert_safe_path(relative_path)
        entries.append(ManifestEntry(source, relative_path))

    required = {
        Path("README.md"),
        Path("landscapes/dbaasp_amp_v1/landscape.json"),
        Path("landscapes/dbaasp_amp_v1/landscape.safetensors"),
        Path("landscapes/dbaasp_amp_v1/nodes.parquet"),
        Path("landscapes/dbaasp_amp_v1/sampler.json"),
    }
    present = {entry.destination for entry in entries}
    missing = sorted(required.difference(present))
    if missing:
        raise SystemExit(f"Bundle is missing required files: {missing}")

    return entries


def size_label(size_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def print_manifest(repo_id: str, private: bool, entries: list[ManifestEntry]) -> None:
    total_size = sum(entry.source.stat().st_size for entry in entries)
    print(f"Repo: {repo_id}")
    print(f"Visibility: {'private' if private else 'public'}")
    print(f"Files: {len(entries)}")
    print(f"Total size: {size_label(total_size)}")
    print("\nIncluded files:")
    for entry in entries:
        print(f"  {entry.destination} ({size_label(entry.source.stat().st_size)})")
    print("\nExcluded:")
    print("  data/")
    print("  *.csv")
    print("  source_peptides.parquet")
    print("\nRemote legacy paths deleted on upload:")
    for pattern in LEGACY_REMOTE_DELETE_PATTERNS:
        print(f"  {pattern}")


def upload_bundle(bundle_root: Path, repo_id: str, private: bool, commit_message: str) -> None:
    token = get_hf_token()
    if not token:
        raise SystemExit("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running with --upload.")

    HfApi = import_hf_api()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(bundle_root),
        path_in_repo=".",
        commit_message=commit_message,
        delete_patterns=LEGACY_REMOTE_DELETE_PATTERNS,
    )


def main() -> int:
    args = parse_args()
    root = project_root()
    bundle_root = (root / args.bundle_root).resolve()
    repo_id = resolve_repo_id(args)
    entries = build_manifest(root, args.bundle_root)

    if not args.upload:
        print_manifest(repo_id, args.private, entries)
        print("\nDry run only. Re-run with --upload to publish to Hugging Face.")
        return 0

    upload_bundle(bundle_root, repo_id, args.private, args.commit_message)
    print(f"Uploaded dataset assets to https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
