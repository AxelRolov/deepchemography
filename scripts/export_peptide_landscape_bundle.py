#!/usr/bin/env python3
"""Export an agent-friendly peptide GTM landscape bundle."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from chemographykit.plots.altair_landscapes import (
    altair_discrete_class_landscape,
    altair_discrete_density_landscape,
)
from chemographykit.plots.plotly_landscapes import plotly_smooth_density_landscape
from chemographykit.utils.classification import (
    class_density_to_table,
    get_class_density_matrix,
)
from chemographykit.utils.density import density_to_table, get_density_matrix
from safetensors.numpy import save_file

from deepchemography.peptides import encode_peptide, load_peptide_model


ORIGINAL_TORCH_LOAD = torch.load
LANDSCAPE_ID = "dbaasp_amp_v1"
DATASET_REPO_ID = "axelrolov/peptide_designer_data"
DECODER_REPO_ID = "axelrolov/wae_peptides"
DEFAULT_ARTIFACT_DIR = Path("output/sampling_analysis/peptides_gtm_analysis")
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
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing the local GTM pickle, images, and HTML plots.",
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
    parser.add_argument(
        "--min-organism-data-points",
        type=int,
        default=500,
        help="Minimum non-null organism labels for notebook-style landscape plots.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def torch_load_cpu(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", False)
    return ORIGINAL_TORCH_LOAD(*args, **kwargs)


def load_gtm_bundle(path: Path) -> dict[str, Any]:
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
        raise ValueError(f"GTM bundle is missing keys: {sorted(missing)}")

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


def copy_images(artifact_dir: Path, bundle_dir: Path) -> list[str]:
    image_dir = bundle_dir / "plots" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    for source in sorted(artifact_dir.glob("*")):
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
            continue
        target = image_dir / source.name
        shutil.copy2(source, target)
        image_paths.append(str(target.relative_to(bundle_dir)))
    return image_paths


def bundle_relative(path: Path, bundle_dir: Path) -> str:
    return path.relative_to(bundle_dir).as_posix()


def write_plotly_artifacts(fig: go.Figure, html_path: Path, image_path: Path) -> None:
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    fig.write_image(str(image_path), format="png", width=800, height=800, scale=2)


def write_dbaasp_density_landscape(
    responsibilities: np.ndarray,
    bundle_dir: Path,
) -> dict[str, str]:
    plots_dir = bundle_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    density_dbaasp_vec = get_density_matrix(responsibilities)
    density_table_dbaasp = density_to_table(
        density_dbaasp_vec,
        node_threshold=0.01,
        output_csv_file=None,
    )

    fig_dbaasp_density = plotly_smooth_density_landscape(
        density_table_dbaasp,
        node_threshold=0.01,
        use_smooth=True,
        title="DBAASP Peptides — Density on WAE GTM",
        width=800,
        height=800,
        background_color="white",
    )

    density_html = plots_dir / "density.html"
    density_png = plots_dir / "density.png"
    write_plotly_artifacts(fig_dbaasp_density, density_html, density_png)
    return {
        "html": bundle_relative(density_html, bundle_dir),
        "png": bundle_relative(density_png, bundle_dir),
    }


def write_organism_chemographykit_landscapes(
    df_dbaasp: pd.DataFrame,
    resp_dbaasp: np.ndarray,
    activity_cols: list[str],
    bundle_dir: Path,
    min_data_points: int,
) -> list[dict[str, Any]]:
    plot_entries: list[dict[str, Any]] = []
    organism_root = bundle_dir / "plots" / "organisms"
    selected_organisms = [
        col for col in activity_cols if df_dbaasp[col].notna().sum() >= min_data_points
    ]
    classes_str = ["Inactive", "Active"]

    for organism in selected_organisms:
        mask = df_dbaasp[organism].notna()
        n_total = int(mask.sum())
        n_active = int((df_dbaasp.loc[mask, organism] == 1.0).sum())
        slug = organism.lower().replace(" ", "_")
        output_dir = organism_root / slug
        output_dir.mkdir(parents=True, exist_ok=True)

        resp_subset = resp_dbaasp[mask.values]
        class_labels = [
            "Active" if v == 1.0 else "Inactive"
            for v in df_dbaasp.loc[mask, organism].values
        ]

        density, class_density, class_prob = get_class_density_matrix(
            resp_subset,
            class_labels=class_labels,
            class_name=classes_str,
            normalize=True,
        )

        source_alt = density_to_table(density=density, node_threshold=0.1)
        source_alt_class = class_density_to_table(
            density=density,
            class_density=class_density,
            class_prob=class_prob,
            class_name=classes_str,
            normalized=True,
            node_threshold=0.1,
        )

        chart_density = altair_discrete_density_landscape(
            source_alt, title=f"{organism} — Density"
        )
        chart_class = altair_discrete_class_landscape(
            source_alt_class,
            title=f"{organism} (Inactive=0, Active=1)",
            first_class_density_column_name=classes_str[0] + "_norm_density",
            first_class_prob_column_name=classes_str[0] + "_norm_prob",
            second_class_density_column_name=classes_str[1] + "_norm_density",
            second_class_prob_column_name=classes_str[1] + "_norm_prob",
            use_density=True,
            colorset="redblue",
            reverse=True,
        )

        combined_chart = (
            chart_density.properties(width=600, height=600)
            | chart_class.properties(width=600, height=600)
        ).resolve_scale(x="independent", y="independent", color="independent")

        activity_class_html = output_dir / "activity_class.html"
        combined_chart.save(str(activity_class_html))
        activity_class_altair_html = output_dir / "activity_class_altair.html"
        shutil.copy2(activity_class_html, activity_class_altair_html)

        active_idx = classes_str.index("Active")
        active_prob = class_prob[:, active_idx]
        density_norm = density / (density.max() + 1e-12)
        score = active_prob * density_norm

        if np.any(density > 0):
            dens_thresh = np.percentile(density[density > 0], 50)
            score = np.where(density > dens_thresh, score, -np.inf)

        best_node_idx = int(np.argmax(score))
        active_prob_table = density_to_table(active_prob, node_threshold=0.0)
        best_row = active_prob_table.loc[
            active_prob_table["nodes"] == best_node_idx + 1
        ].iloc[0]
        node_grid = (int(best_row["x"]), int(best_row["y"]))

        fig_active = plotly_smooth_density_landscape(
            active_prob_table,
            node_threshold=0.0,
            use_smooth=True,
            title=f"{organism} - Active Probability Landscape",
            width=800,
            height=800,
            background_color="white",
        )
        fig_active.add_trace(
            go.Scatter(
                x=[node_grid[0]],
                y=[node_grid[1]],
                mode="markers",
                marker=dict(size=12, color="blue", symbol="star"),
                name="Selected node",
            )
        )

        activity_html = output_dir / "activity.html"
        activity_png = output_dir / "activity.png"
        write_plotly_artifacts(fig_active, activity_html, activity_png)
        active_probability_html = output_dir / "active_probability_plotly.html"
        active_probability_png = output_dir / "active_probability_plotly.png"
        shutil.copy2(activity_html, active_probability_html)
        shutil.copy2(activity_png, active_probability_png)

        plot_entries.append(
            {
                "organism": organism,
                "slug": slug,
                "n_total": n_total,
                "n_active": n_active,
                "active_fraction": float(n_active / n_total) if n_total else 0.0,
                "best_node_id": best_node_idx + 1,
                "best_node_grid": [node_grid[0], node_grid[1]],
                "activity_html": bundle_relative(activity_html, bundle_dir),
                "activity_png": bundle_relative(activity_png, bundle_dir),
                "active_probability_plotly_html": bundle_relative(
                    active_probability_html, bundle_dir
                ),
                "active_probability_plotly_png": bundle_relative(
                    active_probability_png, bundle_dir
                ),
                "activity_class_html": bundle_relative(activity_class_html, bundle_dir),
                "activity_class_altair_html": bundle_relative(
                    activity_class_altair_html, bundle_dir
                ),
            }
        )

    return plot_entries


def copy_gtm_runtime_model(artifact_dir: Path, bundle_dir: Path) -> None:
    runtime_path = artifact_dir / "gtm_wae_model.pkl"
    if not runtime_path.exists():
        raise FileNotFoundError(f"Missing GTM pickle: {runtime_path}")

    target = bundle_dir / "runtime" / "gtm.pkl.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with runtime_path.open("rb") as src, gzip.open(target, "wb") as dst:
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
- `plots/organisms/<organism>/`: notebook-style ChemographyKit Altair class landscapes and Plotly active-probability HTML/PNG landscapes
- `plots/images/`: static image artifacts
- `runtime/gtm.pkl.gz`: compressed GTM/scaler/config pickle for trusted agent sampling workflows

Activity organisms included: {len(organisms)}.

## Source and Attribution

Data were obtained from the DBAASP (https://dbaasp.org) which is an open-access AMP data resource supported by I. Beritashvili Center of Experimental Biomedicine (IBCEB), Tbilisi, Georgia and the National Institute of Allergy and Infectious Diseases (NIAID) Office of Cyber Infrastructure and Computational Biology (OCICB) in Bethesda, MD. These data were collected and submitted by members of the DBAASP team.

Please cite:

Pirtskhalava M, Amstrong AA, Grigolava M, Chubinidze M, Alimbarashvili E, Vishnepolsky B, Gabrielian A, Rosenthal A, Hurt DE, Tartakovsky M. DBAASP v3: database of antimicrobial/cytotoxic activity and structure of peptides as a resource for development of new therapeutics, Nucleic Acids Research, Volume 49, Issue D1, 8 January 2021, Pages D288-D297, https://doi.org/10.1093/nar/gkaa991
"""
    (bundle_root / "README.md").write_text(card, encoding="utf-8")


def export_bundle(args: argparse.Namespace) -> Path:
    root = project_root()
    artifact_dir = (root / args.artifact_dir).resolve()
    bundle_root = (root / args.bundle_root).resolve()
    bundle_dir = bundle_root / "landscapes" / LANDSCAPE_ID
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    gtm_bundle = load_gtm_bundle(artifact_dir / "gtm_wae_model.pkl")
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
    image_paths = copy_images(artifact_dir, bundle_dir)
    density_plot_paths = write_dbaasp_density_landscape(responsibilities, bundle_dir)
    organism_plot_paths = write_organism_chemographykit_landscapes(
        df_activity,
        responsibilities,
        activity_cols,
        bundle_dir,
        min_data_points=args.min_organism_data_points,
    )

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
            "plotted_organisms": [entry["organism"] for entry in organism_plot_paths],
            "plot_min_data_points": int(args.min_organism_data_points),
            "missing_value": -1.0,
        },
        "files": {
            "tensors": "landscape.safetensors",
            "nodes": "nodes.parquet",
            "sampler": "sampler.json",
            "density_plot": density_plot_paths["html"],
            "density_plot_png": density_plot_paths["png"],
            "images": image_paths,
            "organism_plots": organism_plot_paths,
            "gtm_runtime": "runtime/gtm.pkl.gz",
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

    copy_gtm_runtime_model(artifact_dir, bundle_dir)
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
