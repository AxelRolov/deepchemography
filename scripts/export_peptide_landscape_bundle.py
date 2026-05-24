#!/usr/bin/env python3
"""Export an agent-friendly peptide GTM landscape bundle."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from safetensors.numpy import save_file

from deepchemography.peptides import encode_peptide, load_peptide_model


ORIGINAL_TORCH_LOAD = torch.load
LANDSCAPE_ID = "dbaasp_amp_v1"
DATASET_REPO_ID = "axelrolov/peptide_designer_data"
DECODER_REPO_ID = "axelrolov/wae_peptides"
DEFAULT_LEGACY_DIR = Path("output/sampling_analysis/peptides_gtm_analysis")
DEFAULT_BUNDLE_ROOT = Path("output/hf/peptide_designer_data")
AMINO_ACID_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a structured HF landscape bundle from local aggregate inputs."
    )
    parser.add_argument(
        "--activity-csv",
        default=os.environ.get("DBAASP_ACTIVITY_CSV"),
        help="External local DBAASP-style activity CSV. Defaults to DBAASP_ACTIVITY_CSV.",
    )
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        default=DEFAULT_LEGACY_DIR,
        help="Directory containing existing GTM pickle and HTML plots.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Root directory to write the Hugging Face dataset bundle.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/peptides/model_344000.pt"),
        help="Local peptide WAE checkpoint used for activity aggregation.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for peptide encoding.",
    )
    parser.add_argument(
        "--decoder-revision",
        default="main",
        help="Compatible decoder repo revision recorded in landscape.json.",
    )
    parser.add_argument(
        "--min-observations",
        type=float,
        default=1.0,
        help="Minimum weighted node support for assigning activity class.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def torch_load_cpu(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", False)
    return ORIGINAL_TORCH_LOAD(*args, **kwargs)


def load_legacy_bundle(path: Path) -> dict[str, Any]:
    original_load = torch.load
    torch.load = torch_load_cpu
    try:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
    finally:
        torch.load = original_load

    required = {"model", "scaler", "config"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Legacy GTM bundle is missing keys: {sorted(missing)}")

    model = bundle["model"]
    if hasattr(model, "to"):
        model.to("cpu")
    if hasattr(model, "device"):
        model.device = torch.device("cpu")
    return bundle


def as_numpy(value: Any, dtype: np.dtype | str = np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def grid_table(num_nodes: int) -> pd.DataFrame:
    axis_len = int(np.sqrt(num_nodes))
    if axis_len * axis_len != num_nodes:
        raise ValueError(f"Expected square GTM grid, got {num_nodes} nodes")

    x_grid, y_grid = np.meshgrid(range(1, axis_len + 1), range(1, axis_len + 1))
    node_ids = np.arange(1, num_nodes + 1).reshape((axis_len, axis_len)).T.ravel()
    table = pd.DataFrame(
        {"x": x_grid.ravel(), "y": y_grid.ravel(), "node_id": node_ids}
    )
    return table.sort_values("node_id").reset_index(drop=True)


def encode_activity_sequences(
    activity_csv: Path,
    model_path: Path,
    device: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not activity_csv.exists():
        raise FileNotFoundError(f"Activity CSV does not exist: {activity_csv}")

    df = pd.read_csv(activity_csv)
    if "SEQUENCE" not in df.columns:
        raise ValueError("Activity CSV must contain a SEQUENCE column")

    activity_cols = [col for col in df.columns if col != "SEQUENCE"]
    if not activity_cols:
        raise ValueError("Activity CSV must contain at least one activity column")

    model, vocab = load_peptide_model(str(model_path), device=device)
    spaced_sequences = [" ".join(str(seq)) for seq in df["SEQUENCE"].values]

    latents: list[np.ndarray] = []
    for sequence in spaced_sequences:
        z = encode_peptide(model, vocab, sequence, sample_q="max")
        latents.append(z.detach().cpu().numpy())

    return df, np.vstack(latents), activity_cols


def project_activity(
    latents: np.ndarray,
    scaler: Any,
    gtm_model: Any,
) -> np.ndarray:
    scaled = scaler.transform(latents)
    tensor = torch.from_numpy(scaled).to(dtype=torch.float64)
    responsibilities, _ = gtm_model.project(tensor)
    return responsibilities.detach().cpu().numpy()


def weighted_activity_tables(
    df: pd.DataFrame,
    responsibilities: np.ndarray,
    activity_cols: list[str],
    min_observations: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    num_nodes = responsibilities.shape[1]
    grid = grid_table(num_nodes)
    density_all = responsibilities.sum(axis=0).astype(np.float32)

    organisms: list[str] = []
    activity_mean: list[np.ndarray] = []
    activity_uncertainty: list[np.ndarray] = []
    node_support: list[np.ndarray] = []
    rows: list[pd.DataFrame] = []

    for organism in activity_cols:
        values = pd.to_numeric(df[organism], errors="coerce")
        mask = values.notna().to_numpy()
        if not np.any(mask):
            continue

        y = values[mask].to_numpy(dtype=np.float64)
        resp = responsibilities[mask]
        support = resp.sum(axis=0)
        active_weight = (resp * y[:, None]).sum(axis=0)

        mean = np.divide(
            active_weight,
            support,
            out=np.full(num_nodes, np.nan, dtype=np.float64),
            where=support > 0,
        )
        uncertainty = np.sqrt(
            np.divide(
                mean * (1.0 - mean),
                support,
                out=np.full(num_nodes, np.nan, dtype=np.float64),
                where=support > 0,
            )
        )

        activity_class = np.full(num_nodes, "insufficient_support", dtype=object)
        enough = support >= min_observations
        activity_class[enough & (mean >= 0.5)] = "active_enriched"
        activity_class[enough & (mean < 0.5)] = "inactive_enriched"

        organism_rows = grid.copy()
        organism_rows["organism"] = organism
        organism_rows["density"] = density_all
        organism_rows["activity_mean"] = mean.astype(np.float32)
        organism_rows["activity_class"] = activity_class
        organism_rows["uncertainty"] = uncertainty.astype(np.float32)
        organism_rows["n_observations"] = support.astype(np.float32)
        rows.append(organism_rows)

        organisms.append(organism)
        activity_mean.append(np.nan_to_num(mean, nan=-1.0).astype(np.float32))
        activity_uncertainty.append(
            np.nan_to_num(uncertainty, nan=-1.0).astype(np.float32)
        )
        node_support.append(support.astype(np.float32))

    if not rows:
        raise ValueError("No activity columns contained usable values")

    arrays = {
        "landscape.density": density_all,
        "landscape.activity_mean": np.vstack(activity_mean).astype(np.float32),
        "landscape.activity_uncertainty": np.vstack(activity_uncertainty).astype(
            np.float32
        ),
        "landscape.node_support": np.vstack(node_support).astype(np.float32),
    }
    return pd.concat(rows, ignore_index=True), arrays | {
        "organisms": np.asarray(organisms, dtype=object)
    }


def build_tensor_payload(
    gtm_bundle: dict[str, Any], activity_arrays: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    gtm_model = gtm_bundle["model"]
    scaler = gtm_bundle["scaler"]
    tensors = {
        "gtm.nodes": as_numpy(gtm_model.nodes),
        "gtm.basis_centers": as_numpy(gtm_model.mu),
        "gtm.phi": as_numpy(gtm_model.phi),
        "gtm.weights": as_numpy(gtm_model.weights),
        "gtm.beta": as_numpy(gtm_model.beta).reshape(1),
        "scaler.mean": as_numpy(scaler.mean_),
        "scaler.scale": as_numpy(scaler.scale_),
        "grid.xy": grid_table(gtm_bundle["config"]["num_nodes"])[["x", "y"]].to_numpy(
            dtype=np.int32
        ),
    }
    for key, value in activity_arrays.items():
        if key == "organisms":
            continue
        tensors[key] = np.asarray(value)
    return tensors


def copy_plots(legacy_dir: Path, bundle_dir: Path) -> None:
    plots_dir = bundle_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_map = {
        "gtm_dbaasp_density.html": "density.html",
        "gtm_dbaasp_activity_landscapes.html": "activity.html",
    }
    for source_name, target_name in plot_map.items():
        source = legacy_dir / source_name
        if source.exists():
            shutil.copy2(source, plots_dir / target_name)


def copy_images(legacy_dir: Path, bundle_dir: Path) -> list[str]:
    image_dir = bundle_dir / "plots" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    for source in sorted(legacy_dir.glob("*")):
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
            continue
        target = image_dir / source.name
        shutil.copy2(source, target)
        image_paths.append(str(target.relative_to(bundle_dir)))
    return image_paths


def slugify_organism(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def display_organism(value: str) -> str:
    return value.replace("_", " ")


def bundle_relative(path: Path, bundle_dir: Path) -> str:
    return path.relative_to(bundle_dir).as_posix()


def build_hover_text(row: pd.Series) -> str:
    activity = "NA" if pd.isna(row["activity_mean"]) else f"{row['activity_mean']:.3f}"
    uncertainty = "NA" if pd.isna(row["uncertainty"]) else f"{row['uncertainty']:.3f}"
    support = "NA" if pd.isna(row["n_observations"]) else f"{row['n_observations']:.2f}"
    return (
        f"Organism: {display_organism(str(row['organism']))}<br>"
        f"Node: {int(row['node_id'])}<br>"
        f"x: {int(row['x'])}, y: {int(row['y'])}<br>"
        f"Activity mean: {activity}<br>"
        f"Class: {row['activity_class']}<br>"
        f"Uncertainty: {uncertainty}<br>"
        f"Weighted observations: {support}"
    )


def pivot_organism_grid(
    organism_df: pd.DataFrame,
    value_col: str,
) -> tuple[list[int], list[int], np.ndarray]:
    x_values = sorted(int(value) for value in organism_df["x"].unique())
    y_values = sorted(int(value) for value in organism_df["y"].unique())
    grid = (
        organism_df.pivot(index="y", columns="x", values=value_col)
        .reindex(index=y_values, columns=x_values)
        .to_numpy()
    )
    return x_values, y_values, grid


def base_landscape_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        width=900,
        height=800,
        margin={"l": 70, "r": 40, "t": 80, "b": 70},
        xaxis_title="GTM x",
        yaxis_title="GTM y",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")


def write_plotly_artifacts(fig: go.Figure, html_path: Path, image_path: Path) -> None:
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    fig.write_image(str(image_path), format="png", width=900, height=800, scale=2)


def write_organism_plotly_landscapes(
    nodes_df: pd.DataFrame, bundle_dir: Path
) -> list[dict[str, str]]:
    plot_entries: list[dict[str, str]] = []
    organism_root = bundle_dir / "plots" / "organisms"

    for organism, organism_df in sorted(
        nodes_df.groupby("organism"), key=lambda item: slugify_organism(str(item[0]))
    ):
        organism = str(organism)
        slug = slugify_organism(organism)
        output_dir = organism_root / slug
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_df = organism_df.copy()
        plot_df["hover"] = plot_df.apply(build_hover_text, axis=1)
        x_values, y_values, activity_grid = pivot_organism_grid(
            plot_df, "activity_mean"
        )
        _, _, hover_grid = pivot_organism_grid(plot_df, "hover")

        activity_fig = go.Figure(
            data=[
                go.Heatmap(
                    x=x_values,
                    y=y_values,
                    z=activity_grid,
                    zmin=0,
                    zmax=1,
                    colorscale="RdYlBu_r",
                    colorbar={"title": "Activity"},
                    text=hover_grid,
                    hoverinfo="text",
                )
            ]
        )
        base_landscape_layout(
            activity_fig,
            f"{display_organism(organism)} activity landscape",
        )
        activity_html = output_dir / "activity.html"
        activity_png = output_dir / "activity.png"
        write_plotly_artifacts(activity_fig, activity_html, activity_png)

        class_map = {
            "inactive_enriched": 0,
            "insufficient_support": 1,
            "active_enriched": 2,
        }
        plot_df["activity_class_code"] = (
            plot_df["activity_class"].map(class_map).fillna(1)
        )
        _, _, class_grid = pivot_organism_grid(plot_df, "activity_class_code")
        class_fig = go.Figure(
            data=[
                go.Heatmap(
                    x=x_values,
                    y=y_values,
                    z=class_grid,
                    zmin=0,
                    zmax=2,
                    colorscale=[
                        [0.0, "#2f6f9f"],
                        [0.333333, "#2f6f9f"],
                        [0.333334, "#d1d5db"],
                        [0.666666, "#d1d5db"],
                        [0.666667, "#b73535"],
                        [1.0, "#b73535"],
                    ],
                    colorbar={
                        "title": "Class",
                        "tickmode": "array",
                        "tickvals": [0, 1, 2],
                        "ticktext": ["inactive", "insufficient", "active"],
                    },
                    text=hover_grid,
                    hoverinfo="text",
                )
            ]
        )
        base_landscape_layout(
            class_fig,
            f"{display_organism(organism)} activity class landscape",
        )
        class_html = output_dir / "activity_class.html"
        class_png = output_dir / "activity_class.png"
        write_plotly_artifacts(class_fig, class_html, class_png)

        plot_entries.append(
            {
                "organism": organism,
                "slug": slug,
                "activity_html": bundle_relative(activity_html, bundle_dir),
                "activity_png": bundle_relative(activity_png, bundle_dir),
                "activity_class_html": bundle_relative(class_html, bundle_dir),
                "activity_class_png": bundle_relative(class_png, bundle_dir),
            }
        )

    return plot_entries


def copy_gtm_runtime_model(legacy_dir: Path, bundle_dir: Path) -> None:
    legacy_path = legacy_dir / "gtm_wae_model.pkl"
    if not legacy_path.exists():
        raise FileNotFoundError(f"Missing GTM pickle: {legacy_path}")

    targets = [
        bundle_dir / "runtime" / "gtm.pkl.gz",
        bundle_dir / "legacy" / "gtm.pkl.gz",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with legacy_path.open("rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_dataset_card(bundle_root: Path, repo_id: str, organisms: list[str]) -> None:
    card = f"""---
pretty_name: Peptide Designer Data
license: other
tags:
  - peptides
  - antimicrobial-peptides
  - gtm
  - wasserstein-autoencoder
  - safetensors
  - parquet
---

# Peptide Designer Data

This dataset contains structured, aggregate GTM landscape assets for peptide design. The primary bundle is `landscapes/{LANDSCAPE_ID}/`.

Raw DBAASP records and raw peptide source datasets are not redistributed. The bundle contains node-level aggregate tensors, aggregate Parquet tables, plots, metadata, and a compressed GTM runtime pickle for agent sampling.

## Decoder

Compatible peptide decoder: `{DECODER_REPO_ID}`.

## Landscape Bundle

- `landscape.json`: machine-readable landscape contract
- `landscape.safetensors`: dense GTM and aggregate landscape arrays
- `nodes.parquet`: aggregate node-level activity/density records
- `sampler.json`: default sampling policy
- `plots/`: rendered HTML plots
- `plots/organisms/<organism>/`: per-organism Plotly HTML and PNG activity landscapes
- `plots/images/`: static image artifacts
- `runtime/gtm.pkl.gz`: compressed GTM/scaler/config pickle for trusted agent sampling workflows
- `legacy/gtm.pkl.gz`: backward-compatible copy of the same GTM runtime payload

Activity organisms included: {len(organisms)}.

## Source and Attribution

Data were obtained from the DBAASP (https://dbaasp.org) which is an open-access AMP data resource supported by I. Beritashvili Center of Experimental Biomedicine (IBCEB), Tbilisi, Georgia and the National Institute of Allergy and Infectious Diseases (NIAID) Office of Cyber Infrastructure and Computational Biology (OCICB) in Bethesda, MD. These data were collected and submitted by members of the DBAASP team.

Please cite:

Pirtskhalava M, Amstrong AA, Grigolava M, Chubinidze M, Alimbarashvili E, Vishnepolsky B, Gabrielian A, Rosenthal A, Hurt DE, Tartakovsky M. DBAASP v3: database of antimicrobial/cytotoxic activity and structure of peptides as a resource for development of new therapeutics, Nucleic Acids Research, Volume 49, Issue D1, 8 January 2021, Pages D288-D297, https://doi.org/10.1093/nar/gkaa991
"""
    (bundle_root / "README.md").write_text(card, encoding="utf-8")


def export_bundle(args: argparse.Namespace) -> Path:
    root = project_root()
    legacy_dir = (root / args.legacy_dir).resolve()
    bundle_root = (root / args.bundle_root).resolve()
    bundle_dir = bundle_root / "landscapes" / LANDSCAPE_ID
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    gtm_bundle = load_legacy_bundle(legacy_dir / "gtm_wae_model.pkl")
    activity_csv = (
        Path(args.activity_csv).expanduser().resolve() if args.activity_csv else None
    )
    if activity_csv is None:
        raise SystemExit("Provide --activity-csv or set DBAASP_ACTIVITY_CSV.")

    df_activity, latents, activity_cols = encode_activity_sequences(
        activity_csv=activity_csv,
        model_path=(root / args.model_path).resolve(),
        device=args.device,
    )
    responsibilities = project_activity(
        latents, gtm_bundle["scaler"], gtm_bundle["model"]
    )
    nodes_df, activity_arrays = weighted_activity_tables(
        df_activity,
        responsibilities,
        activity_cols,
        min_observations=args.min_observations,
    )
    organisms = [str(x) for x in activity_arrays["organisms"].tolist()]

    nodes_df.to_parquet(bundle_dir / "nodes.parquet", index=False)
    tensor_payload = build_tensor_payload(gtm_bundle, activity_arrays)
    save_file(tensor_payload, str(bundle_dir / "landscape.safetensors"))

    config = gtm_bundle["config"]
    image_paths = copy_images(legacy_dir, bundle_dir)
    organism_plot_paths = write_organism_plotly_landscapes(nodes_df, bundle_dir)

    landscape = {
        "schema_version": "1.0.0",
        "landscape_id": LANDSCAPE_ID,
        "dataset_repo": DATASET_REPO_ID,
        "compatible_decoder_repo": DECODER_REPO_ID,
        "compatible_decoder_revision": args.decoder_revision,
        "required_loader_version": "0.1.0",
        "raw_source_data_redistributed": False,
        "peptide_alphabet": AMINO_ACID_ALPHABET,
        "max_sequence_length": 25,
        "latent_dim": 100,
        "condition_dim": 2,
        "gtm": {
            "num_nodes": int(config["num_nodes"]),
            "grid_shape": [
                int(np.sqrt(config["num_nodes"])),
                int(np.sqrt(config["num_nodes"])),
            ],
            "num_basis_functions": int(config["num_basis_functions"]),
            "basis_width": float(config["basis_width"]),
            "reg_coeff": float(config["reg_coeff"]),
            "max_iter": int(config["max_iter"]),
        },
        "source_dataset": {
            "name": "DBAASP",
            "url": "https://dbaasp.org",
            "citation_doi": "10.1093/nar/gkaa991",
            "redistribution_note": "Raw DBAASP records are not included; only aggregate node-level landscape values are published.",
        },
        "activity_endpoint": {
            "type": "binary antimicrobial activity",
            "units": "active/inactive class",
            "organisms": organisms,
            "missing_value": -1.0,
        },
        "files": {
            "tensors": "landscape.safetensors",
            "nodes": "nodes.parquet",
            "sampler": "sampler.json",
            "density_plot": "plots/density.html",
            "activity_plot": "plots/activity.html",
            "images": image_paths,
            "organism_plots": organism_plot_paths,
            "gtm_runtime": "runtime/gtm.pkl.gz",
            "legacy_gtm": "legacy/gtm.pkl.gz",
        },
        "tensor_names": sorted(tensor_payload.keys()),
    }
    write_json(bundle_dir / "landscape.json", landscape)

    sampler = {
        "schema_version": "1.0.0",
        "target_landscape": LANDSCAPE_ID,
        "decoder_repo": DECODER_REPO_ID,
        "decoder_revision": args.decoder_revision,
        "objective_weights": {
            "activity": 1.0,
            "density_penalty": 0.2,
            "uncertainty_penalty": 0.2,
            "diversity": 0.1,
        },
        "temperature": 1.0,
        "activity_threshold": 0.5,
        "uncertainty_policy": "prefer_low_uncertainty",
        "diversity_policy": "deduplicate_sequences",
    }
    write_json(bundle_dir / "sampler.json", sampler)

    copy_plots(legacy_dir, bundle_dir)
    copy_gtm_runtime_model(legacy_dir, bundle_dir)
    write_dataset_card(bundle_root, DATASET_REPO_ID, organisms)

    print(f"Exported structured landscape bundle to {bundle_dir}")
    print(f"Organisms: {len(organisms)}")
    print(f"Node rows: {len(nodes_df)}")
    return bundle_dir


def main() -> int:
    export_bundle(parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
