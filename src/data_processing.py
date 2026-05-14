# -*- coding: utf-8 -*-
"""
data_processing.py
==================
Handles all pre-modelling steps for the SSEC pipeline:
  - Raw data download / loading from data.npz
  - Train / val / test split (stratified random, seed=42)
  - StandardScaler on raw S-parameter features
  - PCA compression to 10 components (99.9999% variance, chosen to capture Rav)
  - PCA ↔ Target correlation heatmap (diagnostic)
  - Saving all artefacts to data/ subfolders

Expected S-parameter channels (new data configuration):
  S11_Re, S11_Im, S21_Re, S21_Im, S22_Re, S22_Im   → 6 × 435 = 2610 features

Outputs (written to paths/<subfolder>):
  data/raw/data.npz                     — raw data archive
  data/splits/split_{train,val,test}_idx.npy
  data/processed/ssec_pca_final_v2.csv  — normalised PCA scores + targets
  data/pca_artifacts/V_pca_bridge.npy
  data/pca_artifacts/mu_scaler_bridge.npy
  data/pca_artifacts/std_scaler_bridge.npy
  data/pca_artifacts/scaler_mean_bridge.npy
  data/pca_artifacts/pca_score_scaler_params.npz
  outputs/plots/data/pca_correlation_heatmap.png
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NF              = 435          # number of frequency points
DIM_RAW         = 2610         # 6 × NF  (S11_Re, S11_Im, S21_Re, S21_Im, S22_Re, S22_Im)
N_COMPONENTS    = 10           # forced 10 PCs to capture Rav in PC10
VARIANCE_TARGET = 0.999999     # for reference / logging only
TARGET_COLS     = ['Rbv', 'Cbv', 'Rdv', 'Ldv', 'Cdv', 'Rav']
GDRIVE_FILE_ID  = "1EhVuXg7rZus_efQG9aHYI3i39AauItAp"

FREQ_START_GHZ  = 0.04
FREQ_STOP_GHZ   = 43.5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _freq_val(col: str) -> float:
    """Parse frequency value (GHz) from column name like 'S11_Re_1.5GHz'."""
    return float(col.split('_')[-1].replace('GHz', ''))


def _ensure_dir(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — Download / load raw data
# ─────────────────────────────────────────────────────────────────────────────
def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load data.npz from data_dir.  If not present, attempt a gdown download.
    Returns a DataFrame with all S-parameter columns + target columns.
    """
    _ensure_dir(data_dir)
    npz_path = os.path.join(data_dir, "data.npz")

    if not os.path.exists(npz_path):
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            print(f"[data] Downloading data.npz from Google Drive …")
            gdown.download(url, npz_path, quiet=False)
        except Exception as e:
            raise FileNotFoundError(
                f"data.npz not found at '{npz_path}' and auto-download failed: {e}\n"
                f"Place data.npz in '{data_dir}' manually and re-run."
            )

    archive = np.load(npz_path, allow_pickle=True)
    df = pd.DataFrame(archive['matrix'], columns=archive['headers'])
    print(f"[data] Loaded dataset: {df.shape[0]} samples × {df.shape[1]} columns  "
          f"(S21 included)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Build feature column list (S11 + S21 + S22)
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_cols(df: pd.DataFrame):
    """
    Returns the ordered list of 2610 feature columns and the raw X / Y arrays.
    Column order: [S11_Re, S11_Im, S21_Re, S21_Im, S22_Re, S22_Im] each sorted by freq.
    """
    s11_re = sorted([c for c in df.columns if c.startswith("S11_Re")], key=_freq_val)
    s11_im = sorted([c for c in df.columns if c.startswith("S11_Im")], key=_freq_val)
    s21_re = sorted([c for c in df.columns if c.startswith("S21_Re")], key=_freq_val)
    s21_im = sorted([c for c in df.columns if c.startswith("S21_Im")], key=_freq_val)
    s22_re = sorted([c for c in df.columns if c.startswith("S22_Re")], key=_freq_val)
    s22_im = sorted([c for c in df.columns if c.startswith("S22_Im")], key=_freq_val)

    feature_cols = s11_re + s11_im + s21_re + s21_im + s22_re + s22_im
    assert len(feature_cols) == DIM_RAW, (
        f"Expected {DIM_RAW} feature columns, got {len(feature_cols)}. "
        f"Check that S21 data is present in the NPZ file."
    )

    X_raw = df[feature_cols].values.astype(np.float64)
    Y_raw = df[TARGET_COLS].values.astype(np.float64)
    print(f"[data] Feature matrix: {X_raw.shape}  (S11+S21+S22, {len(s11_re)} freq pts each)")
    return feature_cols, X_raw, Y_raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Train / val / test split
# ─────────────────────────────────────────────────────────────────────────────
def make_splits(n_total: int, splits_dir: str = "data/splits", seed: int = 42):
    """
    70 / 15 / 15 random split.  Saves .npy index files and returns the arrays.
    """
    _ensure_dir(splits_dir)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train : n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        np.save(os.path.join(splits_dir, f"split_{name}_idx.npy"), idx)

    print(f"[data] Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Standardise + PCA
# ─────────────────────────────────────────────────────────────────────────────
def run_pca_pipeline(X_raw: np.ndarray,
                     Y_raw: np.ndarray,
                     train_idx: np.ndarray,
                     pca_dir: str    = "data/pca_artifacts",
                     proc_dir: str   = "data/processed",
                     n_components: int = N_COMPONENTS,
                     seed: int = 42):
    """
    1. Fit StandardScaler on training rows.
    2. Fit full PCA on standardised training data.
    3. Keep n_components PCs.
    4. Score-normalise the PCA projections.
    5. Save all bridge matrices and the final CSV.

    Returns: (X_pca_norm, pca_full, scaler, score_mean, score_std)
    """
    _ensure_dir(pca_dir, proc_dir)

    # — Standardise
    scaler = StandardScaler()
    scaler.fit(X_raw[train_idx])
    X_std = scaler.transform(X_raw)

    # — Full PCA (fit on train only)
    np.random.seed(seed)
    pca_full = PCA(svd_solver="full", random_state=seed)
    pca_full.fit(X_std[train_idx])

    evr     = pca_full.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    print(f"[data] Variance captured by {n_components} PCs: {cum_evr[n_components-1]*100:.6f}%")

    # — Project to n_components
    X_pca    = pca_full.transform(X_std)[:, :n_components]
    V_pca_np = pca_full.components_[:n_components].T      # shape (DIM_RAW, n_components)

    # — Score normalisation (zero-mean, unit-std on training scores)
    score_mean = X_pca[train_idx].mean(axis=0)
    score_std  = X_pca[train_idx].std(axis=0)
    X_pca_norm = (X_pca - score_mean) / score_std

    mu_scaler_np = pca_full.mean_.astype(np.float64)

    # — Save bridge matrices
    np.save(os.path.join(pca_dir, "V_pca_bridge.npy"),      V_pca_np)
    np.save(os.path.join(pca_dir, "mu_scaler_bridge.npy"),  mu_scaler_np)
    np.save(os.path.join(pca_dir, "std_scaler_bridge.npy"), scaler.scale_)
    np.save(os.path.join(pca_dir, "scaler_mean_bridge.npy"),scaler.mean_)
    np.savez(os.path.join(pca_dir, "pca_score_scaler_params.npz"),
             score_mean=score_mean, score_std=score_std, n_components=n_components)

    # — Build and save the final CSV
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    df_pca  = pd.DataFrame(X_pca_norm, columns=pc_cols)
    for k, t in enumerate(TARGET_COLS):
        df_pca[t] = Y_raw[:, k]
    csv_path = os.path.join(proc_dir, "ssec_pca_final_v2.csv")
    df_pca.to_csv(csv_path, index=False)

    print(f"[data] PCA pipeline complete.  "
          f"Reduced {DIM_RAW} features → {n_components} PCs.  "
          f"Saved to '{csv_path}'.")
    return X_pca_norm, pca_full, scaler, score_mean, score_std


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — PCA ↔ Target Correlation Analysis (diagnostic plot)
# ─────────────────────────────────────────────────────────────────────────────
def plot_pca_target_correlation(proc_dir: str  = "data/processed",
                                plot_dir: str  = "outputs/plots/data"):
    """
    Load ssec_pca_final_v2.csv and plot the PCA–target Pearson correlation heatmap.
    Saves to plot_dir/pca_correlation_heatmap.png.
    """
    _ensure_dir(plot_dir)
    csv_path = os.path.join(proc_dir, "ssec_pca_final_v2.csv")
    df_pca   = pd.read_csv(csv_path)

    pc_cols  = [c for c in df_pca.columns if c.startswith('PC')]
    corr     = df_pca[pc_cols + TARGET_COLS].corr()
    pc_tgt   = corr.loc[pc_cols, TARGET_COLS]

    plt.figure(figsize=(12, max(6, len(pc_cols) * 0.6)))
    sns.heatmap(pc_tgt, annot=True, cmap='RdBu_r', center=0, fmt=".2f",
                linewidths=0.5, cbar_kws={'label': 'Pearson Correlation (r)'})
    plt.title(f"Correlation: {len(pc_cols)} Principal Components vs. 6 Physical Targets",
              fontsize=14, pad=15)
    plt.ylabel("Principal Components", fontsize=12)
    plt.xlabel("Physical Targets", fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()

    out = os.path.join(plot_dir, "pca_correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[data] Correlation heatmap saved → '{out}'")

    # — Console summary
    print("\n── Maximum Linear Correlation per Target ────────────────────────")
    print(f"  {'Target':>6s} | {'Best PC':>8s} | {'Max |r|':>8s} | {'Status':>12s}")
    print("-" * 48)
    for target in TARGET_COLS:
        if target in pc_tgt.columns:
            abs_c   = pc_tgt[target].abs()
            best_pc = abs_c.idxmax()
            best_v  = pc_tgt.loc[best_pc, target]
            m       = abs_c.max()
            status  = "✅ Strong" if m >= 0.5 else ("⚠️ Moderate" if m >= 0.2 else "❌ Weak")
            print(f"  {target:>6s} | {best_pc:>8s} | {abs(best_v):>8.2f} | {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner (called from pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
def run(data_dir:    str = "data/raw",
        splits_dir:  str = "data/splits",
        pca_dir:     str = "data/pca_artifacts",
        proc_dir:    str = "data/processed",
        plot_dir:    str = "outputs/plots/data",
        seed:        int = 42):
    """Full data-processing pipeline.  Returns paths dict for downstream use."""
    np.random.seed(seed)

    df                          = load_raw_data(data_dir)
    feature_cols, X_raw, Y_raw  = build_feature_cols(df)
    train_idx, val_idx, test_idx = make_splits(len(df), splits_dir, seed)
    X_pca_norm, pca_full, scaler, score_mean, score_std = run_pca_pipeline(
        X_raw, Y_raw, train_idx,
        pca_dir=pca_dir, proc_dir=proc_dir, seed=seed
    )
    plot_pca_target_correlation(proc_dir=proc_dir, plot_dir=plot_dir)

    print("\n[data] ✅ Data processing complete.\n")
    return {
        "data_dir":   data_dir,
        "splits_dir": splits_dir,
        "pca_dir":    pca_dir,
        "proc_dir":   proc_dir,
        "plot_dir":   plot_dir,
    }
