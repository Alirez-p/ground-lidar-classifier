"""
predict_ground3cls_xgb.py

Use a trained 3-class ground classifier to label LAS point clouds.

Classes
-------
0 = non-ground / everything else
1 = near-ground (slightly above ground)
2 = ground

High-level workflow
-------------------
1. Load a trained XGBoost model and a sklearn scaler.
2. Read one or more LAS/LAZ files.
3. Build multi-scale features for all points:
       - Base features:
           z, intensity
       - For each grid size g in MULTI_GRID_SCALES:
           dz_min_g, dz_mean_g, height_norm_g,
           density_g, z_std_g, z_range_g, slope_g
4. Predict classes in chunks (to control memory usage).
5. Optionally refine class 1 vs 2 using smallest-scale features.
6. Write updated 'classification' values back to LAS and save.

This script is intended to be:
- reproducible (same features as model training),
- memory-aware (chunked prediction),
- easy to use from CLI or as an imported module.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Iterable, List, Sequence

import joblib
import laspy
import numpy as np
import xgboost as xgb


# =====================================================================
# Defaults (used when running as a script)
# =====================================================================

DEFAULT_MULTI_GRID_SCALES: List[float] = [0.25, 1.0, 2.0]
DEFAULT_CHUNK_SIZE: int = 3_000_000  # points per prediction chunk


# =====================================================================
# Utility: timing decorator
# =====================================================================

def timed(name: str):
    """
    Decorator to print wall-clock time for a function call.

    Parameters
    ----------
    name : str
        Human-readable name for the timed section.
    """

    def deco(func):
        def wrapper(*args, **kwargs):
            print(f"\n⏳ {name} ...")
            t0 = time.time()
            result = func(*args, **kwargs)
            dt = time.time() - t0
            print(f"✅ {name} done in {dt:.2f} s")
            return result

        return wrapper

    return deco


# =====================================================================
# Refinement: fix obvious 1 ↔ 2 misclassifications
# =====================================================================

def refine_ground_labels(
    x_all: np.ndarray,
    y_all: np.ndarray,
    X_all: np.ndarray,
    labels: np.ndarray,
    grid_scales: Sequence[float],
    dz_promote_max: float = 0.05,   # e.g. ≤ 5 cm above local min → likely ground
    hn_promote_max: float = 0.3,    # low normalized height
    cell2_majority_thresh: float = 0.7,
    dz_demote_min: float = 0.20,    # e.g. ≥ 20 cm above local min → suspicious
    hn_demote_min: float = 0.9,
    cell2_min_frac: float = 0.3,
    use_cell_fraction_rules: bool = False,
) -> np.ndarray:
    """
    Post-process predicted labels to fix obvious 1 ↔ 2 misclassifications.

    This operates at the smallest grid scale in `grid_scales`:

    - Promote some near-ground (1) points to ground (2) if:
        * very close to local min-z and
        * low normalized height
        * optionally: belong to a cell strongly dominated by class 2
    - Demote suspicious ground (2) points to near-ground (1) if:
        * notably above local min-z or have large normalized height
        * optionally: belong to a cell that is not ground-dominated

    Parameters
    ----------
    x_all, y_all : np.ndarray
        Coordinates of all points (1D arrays, same length as `labels`).
    X_all : np.ndarray
        Full feature matrix used for classification. Must match the layout
        produced by `extract_multiscale_features`.
    labels : np.ndarray
        Predicted labels, shape (N,), values in {0, 1, 2}.
    grid_scales : Sequence[float]
        List of grid sizes used in feature extraction, e.g. [0.25, 1.0, 2.0].
    dz_promote_max : float, optional
        Max dz_min threshold below which class 1 points can be promoted to 2.
    hn_promote_max : float, optional
        Max height_norm threshold below which class 1 points can be promoted.
    cell2_majority_thresh : float, optional
        If `use_cell_fraction_rules=True`, minimum fraction of class 2 in a cell
        to allow 1→2 promotion.
    dz_demote_min : float, optional
        Min dz_min threshold above which class 2 points can be demoted to 1.
    hn_demote_min : float, optional
        Min height_norm above which class 2 points can be demoted to 1.
    cell2_min_frac : float, optional
        If `use_cell_fraction_rules=True`, max fraction of class 2 in a cell
        below which 2→1 demotion is allowed.
    use_cell_fraction_rules : bool, optional
        If True, the per-cell class fractions are also used as part of
        promote/demote conditions. Defaults to False to match original
        behaviour.

    Returns
    -------
    np.ndarray
        Refined labels, same shape as input.
    """
    print("\n🔧 Refining ground / near-ground labels ...")
    t0 = time.time()

    labels = np.asarray(labels, dtype=np.uint8)
    refined = labels.copy()

    # ---------- feature layout indices for smallest grid ----------
    # Layout:
    #   0: z
    #   1: intensity
    #   For each grid i (0-based):
    #       2 + 7*i + 0 : dz_min_i
    #       2 + 7*i + 1 : dz_mean_i
    #       2 + 7*i + 2 : height_norm_i
    #       2 + 7*i + 3 : density_i
    #       2 + 7*i + 4 : z_std_i
    #       2 + 7*i + 5 : z_range_i
    #       2 + 7*i + 6 : slope_i
    smallest_grid = grid_scales[0]
    scale_index = 0
    base_idx = 2 + 7 * scale_index

    dz_min0 = X_all[:, base_idx + 0]
    height_norm0 = X_all[:, base_idx + 2]

    # ---------- build grid cells on smallest scale ----------
    x = np.asarray(x_all)
    y = np.asarray(y_all)

    xmin, ymin = x.min(), y.min()
    g = float(smallest_grid)

    ix = ((x - xmin) / g).astype(np.int32)
    iy = ((y - ymin) / g).astype(np.int32)

    cells = np.stack([ix, iy], axis=1)
    cell_ids, inv = np.unique(cells, axis=0, return_inverse=True)
    n_cells = cell_ids.shape[0]

    print(
        f"   • Using {n_cells:,} grid cells at scale {smallest_grid} m "
        "for refinement"
    )

    # per-cell class composition
    cell_total = np.bincount(inv, minlength=n_cells)

    is0 = (labels == 0).astype(np.int32)
    is1 = (labels == 1).astype(np.int32)
    is2 = (labels == 2).astype(np.int32)

    cell_c0 = np.bincount(inv, weights=is0, minlength=n_cells)
    cell_c1 = np.bincount(inv, weights=is1, minlength=n_cells)
    cell_c2 = np.bincount(inv, weights=is2, minlength=n_cells)

    cell_total_safe = cell_total + 1e-6  # avoid divide by zero
    frac2 = cell_c2 / cell_total_safe
    frac1 = cell_c1 / cell_total_safe

    frac2_pt = frac2[inv]
    frac1_pt = frac1[inv]

    # ---------- RULE 1: promote 1 → 2 ----------
    promote_mask = (
        (labels == 1)
        & (dz_min0 <= dz_promote_max)
        & (height_norm0 <= hn_promote_max)
    )
    if use_cell_fraction_rules:
        promote_mask &= frac2_pt >= cell2_majority_thresh

    n_promote = int(promote_mask.sum())
    refined[promote_mask] = 2

    # ---------- RULE 2: demote 2 → 1 ----------
    demote_mask = (
        (labels == 2)
        & ((dz_min0 >= dz_demote_min) | (height_norm0 >= hn_demote_min))
    )
    if use_cell_fraction_rules:
        demote_mask &= frac2_pt <= cell2_min_frac

    n_demote = int(demote_mask.sum())
    refined[demote_mask] = 1

    dt = time.time() - t0
    print(f"   • Promoted 1 → 2: {n_promote:,} points")
    print(f"   • Demoted  2 → 1: {n_demote:,} points")
    print(f"✅ Refinement done in {dt:.2f} s")

    return refined


# =====================================================================
# Multi-scale feature extraction
# =====================================================================

def _grid_stats_single_scale(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: float,
) -> tuple[np.ndarray, ...]:
    """
    Compute local grid statistics for a single grid size.

    For each point, the following features are returned:
        dz_min      : z - local_min_z_in_cell
        dz_mean     : z - local_mean_z_in_cell
        height_norm : dz_min / (z_range + eps)
        density     : points per m² (cell_count / grid²)
        z_std       : std(z) in cell
        z_range     : max_z - min_z in cell
        slope       : max |dz| / grid using 4-neighbor min-z cells
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    xmin, ymin = x.min(), y.min()

    ix = ((x - xmin) / grid).astype(np.int32)
    iy = ((y - ymin) / grid).astype(np.int32)

    cells = np.stack([ix, iy], axis=1)
    cell_ids, inv = np.unique(cells, axis=0, return_inverse=True)
    n_cells = cell_ids.shape[0]

    # counts per cell
    cell_counts = np.bincount(inv, minlength=n_cells)

    # min z per cell (≈ ground)
    cell_zmin = np.full(n_cells, np.inf, dtype=np.float64)
    np.minimum.at(cell_zmin, inv, z)

    # max z per cell
    cell_zmax = np.full(n_cells, -np.inf, dtype=np.float64)
    np.maximum.at(cell_zmax, inv, z)

    # sums for mean/std
    cell_zsum = np.zeros(n_cells, dtype=np.float64)
    cell_zsqsum = np.zeros(n_cells, dtype=np.float64)
    np.add.at(cell_zsum, inv, z)
    np.add.at(cell_zsqsum, inv, z * z)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_z = cell_zsum / cell_counts
        var_z = (cell_zsqsum / cell_counts) - mean_z**2
    var_z[var_z < 0] = 0.0
    std_z = np.sqrt(var_z)

    # slope based on min-z (ground-like surface)
    coord_to_idx = {
        (int(cx), int(cy)): i for i, (cx, cy) in enumerate(cell_ids)
    }
    cell_slope = np.zeros(n_cells, dtype=np.float64)
    neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for i, (cx, cy) in enumerate(cell_ids):
        zc = cell_zmin[i]
        max_s = 0.0
        for dx_n, dy_n in neighbor_offsets:
            nb = (int(cx + dx_n), int(cy + dy_n))
            j = coord_to_idx.get(nb)
            if j is None:
                continue
            dz = cell_zmin[j] - zc
            s = abs(dz) / grid
            if s > max_s:
                max_s = s
        cell_slope[i] = max_s

    # map back to points
    z_min = cell_zmin[inv]
    z_max = cell_zmax[inv]
    z_mean = mean_z[inv]
    slope = cell_slope[inv]

    dz_min = z - z_min
    dz_mean = z - z_mean
    z_range = z_max - z_min
    density = cell_counts[inv] / (grid * grid)
    z_std = std_z[inv]

    eps = 1e-3
    height_norm = dz_min / (z_range + eps)

    return (
        dz_min.astype(np.float32),
        dz_mean.astype(np.float32),
        height_norm.astype(np.float32),
        density.astype(np.float32),
        z_std.astype(np.float32),
        z_range.astype(np.float32),
        slope.astype(np.float32),
    )


def extract_multiscale_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    intensity: np.ndarray,
    grid_scales: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Build multi-scale features for all points.

    Feature layout (current implementation):
        base features:
            0: z
            1: intensity
        for each grid scale g (in order):
            dz_min_g
            dz_mean_g
            height_norm_g
            density_g
            z_std_g
            z_range_g
            slope_g

    Total dimension:
        2 + 7 * len(grid_scales)
    """
    if grid_scales is None:
        grid_scales = DEFAULT_MULTI_GRID_SCALES

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    intensity = np.asarray(intensity, dtype=np.float32)

    n_points = len(z)
    print(f"\n🧠 Building multi-scale features for {n_points:,} points ...")
    t_global = time.time()

    feature_blocks: List[np.ndarray] = [
        z.astype(np.float32),
        intensity,
    ]

    n_scales = len(grid_scales)
    for i, g in enumerate(grid_scales, start=1):
        print(
            f"   → [{i}/{n_scales}] scale {g} m: computing grid stats ...",
            end="",
            flush=True,
        )
        t0 = time.time()
        (
            dz_min,
            dz_mean,
            height_norm,
            density,
            z_std,
            z_range,
            slope,
        ) = _grid_stats_single_scale(x, y, z, grid=float(g))

        feature_blocks.extend(
            [
                dz_min,
                dz_mean,
                height_norm,
                density,
                z_std,
                z_range,
                slope,
            ]
        )
        dt = time.time() - t0
        print(f" done in {dt:.2f} s")

    feats = np.column_stack(feature_blocks)
    dt_total = time.time() - t_global
    print(
        f"✅ Multi-scale features built: shape={feats.shape}, "
        f"total time={dt_total:.2f} s\n"
    )

    return feats


# =====================================================================
# Model / scaler loading
# =====================================================================

@timed("Loading model & scaler")
def load_model_and_scaler(
    model_path: str,
    scaler_path: str,
) -> tuple[xgb.XGBClassifier, object]:
    """
    Load XGBoost model and sklearn-compatible scaler from disk.
    """
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    scaler = joblib.load(scaler_path)

    print("   • Model and scaler loaded successfully.")
    return model, scaler


# =====================================================================
# LAS classification
# =====================================================================

@timed("Classifying LAS file")
def classify_las_file(
    in_path: str,
    out_path: str,
    model: xgb.XGBClassifier,
    scaler,
    grid_scales: Sequence[float],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    refine: bool = True,
    use_cell_fraction_rules: bool = False,
) -> None:
    """
    Run 3-class ground classification on a single LAS/LAZ file.

    Parameters
    ----------
    in_path : str
        Path to input LAS/LAZ file.
    out_path : str
        Path to output LAS/LAZ file. Will be overwritten if exists.
    model : xgb.XGBClassifier
        Trained 3-class XGBoost classifier.
    scaler : object
        sklearn-compatible transformer (e.g. StandardScaler).
    grid_scales : Sequence[float]
        Grid sizes used for the multi-scale features.
    chunk_size : int, optional
        Number of points per prediction chunk.
    refine : bool, optional
        Whether to run the 1↔2 refinement step.
    use_cell_fraction_rules : bool, optional
        If True, enable cell-fraction conditions inside refinement.
    """
    print(f"📂 Reading input LAS: {in_path}")
    las = laspy.read(in_path)
    n_points = len(las.x)
    print(f"   • Total points: {n_points:,}")

    # Allocate prediction array
    new_classes = np.zeros(n_points, dtype=np.uint8)

    x_all = las.x
    y_all = las.y
    z_all = las.z

    if hasattr(las, "intensity"):
        intensity_all = np.asarray(las.intensity, dtype=np.float32)
    else:
        intensity_all = np.zeros(n_points, dtype=np.float32)

    # -----------------------------------------------------------------
    # Global feature matrix (so refinement can see all points at once)
    # -----------------------------------------------------------------
    print("   • Building global multi-scale features for all points ...")
    X_all = extract_multiscale_features(
        x=x_all,
        y=y_all,
        z=z_all,
        intensity=intensity_all,
        grid_scales=grid_scales,
    )
    print(f"   • Feature matrix shape: {X_all.shape}")

    # -----------------------------------------------------------------
    # Chunked classification
    # -----------------------------------------------------------------
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        print(f"   → Predicting chunk {start:,} : {end:,}")

        X_chunk = X_all[start:end]
        X_scaled = scaler.transform(X_chunk.astype(np.float32))
        pred_chunk = model.predict(X_scaled).astype(np.uint8)

        new_classes[start:end] = pred_chunk

    # -----------------------------------------------------------------
    # Optional refinement of class 1 vs 2
    # -----------------------------------------------------------------
    if refine:
        new_classes = refine_ground_labels(
            x_all=x_all,
            y_all=y_all,
            X_all=X_all,
            labels=new_classes,
            grid_scales=grid_scales,
            use_cell_fraction_rules=use_cell_fraction_rules,
        )

    # -----------------------------------------------------------------
    # Save output LAS
    # -----------------------------------------------------------------
    las.classification = new_classes  # laspy expects uint8, same length as points

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    las.write(out_path)

    unique, counts = np.unique(new_classes, return_counts=True)
    print(f"\n📊 Predicted classes distribution for {os.path.basename(out_path)}:")
    for cls_val, cnt in zip(unique, counts):
        print(f"   class {cls_val}: {cnt:,} points")

    print(f"\n💾 Saved classified LAS to: {out_path}")


# =====================================================================
# Folder-level helper
# =====================================================================

def find_las_files(
    input_dir: str,
    pattern: str = "*.las",
) -> List[str]:
    """
    Find LAS/LAZ files in a folder matching the given pattern.
    """
    search_pattern = os.path.join(input_dir, pattern)
    paths = sorted(glob.glob(search_pattern))
    return paths


def process_las_folder(
    model_path: str,
    scaler_path: str,
    input_dir: str,
    output_dir: str,
    pattern: str = "*.las",
    grid_scales: Sequence[float] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    refine: bool = True,
    use_cell_fraction_rules: bool = False,
) -> None:
    """
    Run classification for all LAS/LAZ files in a folder.
    """
    if grid_scales is None:
        grid_scales = DEFAULT_MULTI_GRID_SCALES

    model, scaler = load_model_and_scaler(model_path, scaler_path)

    las_paths = find_las_files(input_dir, pattern=pattern)
    if not las_paths:
        raise FileNotFoundError(
            f"No files found in '{input_dir}' with pattern '{pattern}'"
        )

    print(f"\n🔍 Found {len(las_paths)} LAS file(s) for prediction.")

    os.makedirs(output_dir, exist_ok=True)

    for idx, in_las in enumerate(las_paths, start=1):
        print(f"\n========== File {idx}/{len(las_paths)} ==========")
        base = os.path.basename(in_las)
        name, ext = os.path.splitext(base)
        out_las = os.path.join(output_dir, f"{name}_ground3cls{ext}")

        classify_las_file(
            in_path=in_las,
            out_path=out_las,
            model=model,
            scaler=scaler,
            grid_scales=grid_scales,
            chunk_size=chunk_size,
            refine=refine,
            use_cell_fraction_rules=use_cell_fraction_rules,
        )

    print("\n✅ All files classified.")


# =====================================================================
# CLI
# =====================================================================

def _parse_grid_scales(arg: str | None) -> List[float]:
    """
    Parse a comma-separated string like '0.25,1.0,2.0' into [0.25, 1.0, 2.0].
    """
    if not arg:
        return DEFAULT_MULTI_GRID_SCALES
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [float(p) for p in parts]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3-class ground classifier for LAS/LAZ point clouds "
        "using a trained XGBoost model and multi-scale features.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained XGBoost model (.json).",
    )
    parser.add_argument(
        "--scaler",
        required=True,
        help="Path to fitted scaler (.pkl).",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input folder containing LAS/LAZ files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output folder for classified LAS/LAZ files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.las",
        help="Glob pattern for input files (default: '*.las').",
    )
    parser.add_argument(
        "--grid-scales",
        default=None,
        help=(
            "Comma-separated list of grid scales in meters. "
            f"Default: {','.join(map(str, DEFAULT_MULTI_GRID_SCALES))}"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Number of points per prediction chunk "
            f"(default: {DEFAULT_CHUNK_SIZE})."
        ),
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable post-classification 1↔2 refinement.",
    )
    parser.add_argument(
        "--use-cell-fraction-rules",
        action="store_true",
        help=(
            "Enable cell-fraction rules in refinement "
            "(class-2 dominance / scarcity checks)."
        ),
    )

    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)

    grid_scales = _parse_grid_scales(parsed.grid_scales)

    process_las_folder(
        model_path=parsed.model,
        scaler_path=parsed.scaler,
        input_dir=parsed.input_dir,
        output_dir=parsed.output_dir,
        pattern=parsed.pattern,
        grid_scales=grid_scales,
        chunk_size=parsed.chunk_size,
        refine=not parsed.no_refine,
        use_cell_fraction_rules=parsed.use_cell_fraction_rules,
    )


if __name__ == "__main__":
    # Example usage (CLI):
    #
    #   python predict_ground3cls_xgb_multiscale.py ^
    #       --model "path/to/ground3cls_xgb_model.json" ^
    #       --scaler "path/to/ground3cls_scaler.pkl" ^
    #       --input-dir "path/to/input_las_folder" ^
    #       --output-dir "path/to/output_las_folder" ^
    #       --pattern "*.las" ^
    #       --grid-scales "0.25,1.0,2.0" ^
    #       --chunk-size 3000000
    #
    main()
