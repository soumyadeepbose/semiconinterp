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
# VNA Noise Augmentation
# ─────────────────────────────────────────────────────────────────────────────
# Frequency axis consistent with NF=435 points from 40 MHz to 43.5 GHz
_FREQS_GHZ = np.linspace(FREQ_START_GHZ, FREQ_STOP_GHZ, NF)


def realistic_vna_noise(X_2610: np.ndarray,
                         freqs_GHz: np.ndarray = _FREQS_GHZ,
                         sigma_0: float = 0.002,
                         beta: float = 2.0,
                         seed: int = None) -> np.ndarray:
    """
    Add frequency-dependent VNA measurement noise to the raw 2610-dim feature matrix.

    Noise model: σ(f) = σ_0 · (1 + β · f / f_max)
      - At low frequencies: σ ≈ σ_0  (thermal/floor noise)
      - At high frequencies: σ → σ_0·(1+β) (cable/probe losses grow with freq)

    The same envelope is applied to all 6 channels (Re/Im of S11, S21, S22),
    reflecting a shared cable/probe assembly.

    Args:
        X_2610    : (N, 2610) raw S-parameter matrix
        freqs_GHz : (NF,) frequency axis in GHz  [default: linspace(0.04, 43.5, 435)]
        sigma_0   : base noise level (default 0.002 — ~0.2% of unit S-param range)
        beta      : high-frequency noise growth factor (default 2.0 → 3× at f_max)
        seed      : optional RNG seed for reproducibility

    Returns:
        X_noisy : (N, 2610) array with additive noise applied
    """
    rng = np.random.default_rng(seed)
    # σ(f) shape: (NF,)
    noise_envelope = sigma_0 * (1.0 + beta * freqs_GHz / freqs_GHz[-1])
    # Build noise for all 6 channels using the same envelope, then concatenate
    noise = np.concatenate(
        [rng.standard_normal(X_2610[:, ch*NF:(ch+1)*NF].shape) * noise_envelope
         for ch in range(6)],
        axis=1
    )
    return X_2610 + noise


# ─────────────────────────────────────────────────────────────────────────────
# PCA Diagnostic Plots
# ─────────────────────────────────────────────────────────────────────────────
def _plot_scree(pca_full, n_kept: int, plot_dir: str, n_show: int = 80):
    """
    Scree plot: individual + cumulative explained variance ratio per PC.
    Helps decide how many PCs to keep.
    """
    _ensure_dir(plot_dir)
    evr     = pca_full.explained_variance_ratio_
    n_show  = min(n_show, len(evr))
    xs      = np.arange(1, n_show + 1)
    cum     = np.cumsum(evr[:n_show]) * 100.0
    indiv   = evr[:n_show] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Left: individual EVR (log scale) ────────────────────────────────
    ax = axes[0]
    ax.bar(xs, indiv, color='steelblue', alpha=0.75, width=0.85)
    ax.axvline(n_kept + 0.5, color='crimson', lw=2.0, linestyle='--',
               label=f'Chosen n={n_kept}  ({indiv[:n_kept].sum():.4f}%)')
    ax.set_yscale('log')
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('Explained Variance (%, log)', fontsize=11)
    ax.set_title('Scree Plot — Individual Explained Variance\n'
                 'Log-scale reveals the elbow clearly', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.28, which='both')
    ax.set_xlim(0.5, n_show + 0.5)
    ax.set_xticks(xs[::5])

    # ── Right: cumulative EVR ───────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(xs, cum, 'ko-', ms=4, lw=1.8)
    for threshold, color, label in [
        (99.0,    'orange', '99%'),
        (99.9,    'green',  '99.9%'),
        (99.9999, 'red',    '99.9999%'),
    ]:
        ax2.axhline(threshold, color=color, lw=1.2, linestyle=':', alpha=0.8, label=label)
    ax2.axvline(n_kept + 0.5, color='crimson', lw=2.0, linestyle='--',
                label=f'Chosen n={n_kept}\n→ {cum[n_kept-1]:.6f}%')
    ax2.set_xlabel('Number of Principal Components', fontsize=11)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
    ax2.set_title('Cumulative Scree Plot\n'
                  'Use to pick threshold for desired variance retention', fontsize=10)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.28)
    ax2.set_xlim(0.5, n_show + 0.5)
    ax2.set_ylim(None, 100.02)
    ax2.set_xticks(xs[::5])

    plt.suptitle(
        f'PCA Scree Plot — 2610-dim SSEC S-parameter Feature Space\n'
        f'Full SVD: {len(evr)} components  |  Kept: {n_kept}  |'
        f'  Showing first {n_show}',
        fontsize=11, fontweight='bold')
    out = os.path.join(plot_dir, 'pca_scree_plot.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[data] Scree plot → '{out}'")


def plot_extended_correlation(proc_dir: str = "data/processed",
                               plot_dir: str = "outputs/plots/data",
                               n_ext:    int = 30):
    """
    Extended PCA–target correlation heatmap using the first n_ext PCs.
    Reads ssec_pca_30pc_diag.csv saved by run_pca_pipeline().
    """
    _ensure_dir(plot_dir)
    csv_path = os.path.join(proc_dir, "ssec_pca_30pc_diag.csv")
    if not os.path.exists(csv_path):
        print(f"[data] Extended correlation: '{csv_path}' not found — skipping.")
        return

    df_ext  = pd.read_csv(csv_path)
    pc_cols = [f'PC{i+1}' for i in range(n_ext) if f'PC{i+1}' in df_ext.columns]
    tgt     = [t for t in TARGET_COLS if t in df_ext.columns]
    corr    = df_ext[pc_cols + tgt].corr()
    pc_tgt  = corr.loc[pc_cols, tgt]

    fig_h = max(10, len(pc_cols) * 0.52)
    plt.figure(figsize=(13, fig_h))
    sns.heatmap(
        pc_tgt, annot=True, cmap='RdBu_r', center=0, fmt=".2f",
        linewidths=0.4, annot_kws={'size': 8},
        cbar_kws={'label': 'Pearson Correlation (r)', 'shrink': 0.6}
    )
    plt.title(
        f"Extended Correlation: First {len(pc_cols)} PCs vs. {len(tgt)} Physical Targets\n"
        f"(PCs beyond the model's kept {N_COMPONENTS} may still encode useful variance)",
        fontsize=12, pad=14)
    plt.ylabel("Principal Components", fontsize=11)
    plt.xlabel("Physical Targets", fontsize=11)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    out = os.path.join(plot_dir, f"pca_correlation_heatmap_{n_ext}pc.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[data] Extended ({n_ext}-PC) correlation heatmap → '{out}'")

    # Per-target best-PC summary for all 30
    print(f"\n── Extended Correlation Summary (top-{n_ext} PCs) ──────────────────")
    for target in tgt:
        abs_c   = pc_tgt[target].abs()
        best_pc = abs_c.idxmax()
        best_v  = pc_tgt.loc[best_pc, target]
        rank    = (abs_c.sort_values(ascending=False).index.tolist()[:3])
        print(f"  {target:>6s}  best={best_pc} (r={best_v:+.3f})  "
              f"top-3={rank}")


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
                     plot_dir: str   = "outputs/plots/data",
                     n_components: int = N_COMPONENTS,
                     seed: int = 42,
                     add_noise: bool = True,
                     n_ext_pcs: int = 30):
    """
    1. (Optional) Add realistic VNA noise to X_raw before any fitting.
    2. Fit StandardScaler on training rows.
    3. Fit full PCA on standardised training data.
    4. Keep n_components PCs (model artefacts) + n_ext_pcs for diagnostics.
    5. Score-normalise the PCA projections.
    6. Save all bridge matrices, the final CSV, and a 30-PC diagnostic CSV.
    7. Plot the scree plot.

    Returns: (X_pca_norm, pca_full, scaler, score_mean, score_std)
    """
    _ensure_dir(pca_dir, proc_dir, plot_dir)

    # — Optional VNA noise augmentation (applied before any normalisation)
    if add_noise:
        print(f"[data] Applying realistic VNA noise (σ_0=0.002, β=2.0) …")
        X_raw = realistic_vna_noise(X_raw, seed=seed)
        print(f"[data] Noise added.  X_raw shape: {X_raw.shape}")

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

    # — Scree plot (while full PCA object is available)
    _plot_scree(pca_full, n_components, plot_dir)

    # — Project to n_components (model artefacts)
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

    # — Build and save the final model CSV (n_components PCs)
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    df_pca  = pd.DataFrame(X_pca_norm, columns=pc_cols)
    for k, t in enumerate(TARGET_COLS):
        df_pca[t] = Y_raw[:, k]
    csv_path = os.path.join(proc_dir, "ssec_pca_final_v2.csv")
    df_pca.to_csv(csv_path, index=False)

    # — Save extended diagnostic CSV (n_ext_pcs PCs, un-normalised raw scores)
    n_ext_actual = min(n_ext_pcs, pca_full.n_components_)
    X_pca_ext    = pca_full.transform(X_std)[:, :n_ext_actual]
    pc_cols_ext  = [f'PC{i+1}' for i in range(n_ext_actual)]
    df_ext       = pd.DataFrame(X_pca_ext, columns=pc_cols_ext)
    for k, t in enumerate(TARGET_COLS):
        df_ext[t] = Y_raw[:, k]
    diag_path = os.path.join(proc_dir, "ssec_pca_30pc_diag.csv")
    df_ext.to_csv(diag_path, index=False)
    print(f"[data] Extended {n_ext_actual}-PC diagnostic CSV → '{diag_path}'")

    print(f"[data] PCA pipeline complete.  "
          f"Reduced {DIM_RAW} features → {n_components} PCs (model) + "
          f"{n_ext_actual} PCs (diagnostic).  Saved to '{csv_path}'.")
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
        seed:        int = 42,
        add_noise:   bool = True):
    """Full data-processing pipeline.  Returns paths dict for downstream use."""
    np.random.seed(seed)

    df                          = load_raw_data(data_dir)
    feature_cols, X_raw, Y_raw  = build_feature_cols(df)
    train_idx, val_idx, test_idx = make_splits(len(df), splits_dir, seed)
    X_pca_norm, pca_full, scaler, score_mean, score_std = run_pca_pipeline(
        X_raw, Y_raw, train_idx,
        pca_dir=pca_dir, proc_dir=proc_dir, plot_dir=plot_dir,
        seed=seed, add_noise=add_noise
    )
    plot_pca_target_correlation(proc_dir=proc_dir, plot_dir=plot_dir)
    plot_extended_correlation(proc_dir=proc_dir, plot_dir=plot_dir, n_ext=30)

    print("\n[data] ✅ Data processing complete.\n")
    return {
        "data_dir":   data_dir,
        "splits_dir": splits_dir,
        "pca_dir":    pca_dir,
        "proc_dir":   proc_dir,
        "plot_dir":   plot_dir,
    }
