# -*- coding: utf-8 -*-
"""
src/interp2.py
==============
Extended Interpretability — Part 2.

Implements four new analysis methods on top of interpretability.py:

  A. Counterfactual Interventions + Causal Closure Test
     For each parameter: sweep its bottleneck value from min→max while
     holding others fixed.  Re-encode the decoded spectrum and check
     whether the encoder recovers the intervened value (causal closure).

  B. Round-Trip Decay Analysis  (user extension of A)
     Starting from each test sample's original S-parameter input, cycle:
       encode → decode → pca_invert → pca_encode → encode → ...
     for N rounds.  Plot how the parameter estimates drift from ground
     truth across rounds.  A fixed-point model shows zero drift; an
     inconsistent model shows monotonically growing error.

  C. Resonance Probing
     Register hooks on every activation layer of the encoder.  Compute
     derived physical quantities from true parameter values (corner
     frequencies f_d, f_b; time-constant products τ_d, τ_b; resistance
     ratio Rd/Ra).  Train linear (Ridge) and nonlinear (MLP) probes at
     each layer and plot R² vs layer depth.  Increasing R² with depth
     shows the model is progressively building up physics representations.

  D. Weight Matrix SVD
     SVD of every Linear layer in encoder and decoder.  Reports singular-
     value spectrum, effective rank, and — for the first encoder layer —
     alignment of the input singular vectors with the 10-dim PCA basis
     lifted back to the 2610-dim spectral space.

Call:  python -m src.interp2 --model-path <ckpt> --model-type piae
Or from pipeline.py: import src.interp2 as interp2; interp2.run(...)
"""

import os, math, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler as _SS
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from csv_utils import csv_save

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NF      = 435
DIM_RAW = 2610

PIAE_TARGETS = ['Rbv', 'Cbv', 'Rdv', 'Cdv', 'Rav']
CVAE_TARGETS = ['Rbv', 'Cbv', 'Rdv', 'Ldv', 'Cdv', 'Rav']
LDV_IDX      = CVAE_TARGETS.index('Ldv')

_PARAM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#17becf']

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pca_encode(model: nn.Module, X_phys: torch.Tensor) -> torch.Tensor:
    """
    Physical 2610-dim spectrum → normalised PCA scores (n_pca-dim).
    This is the inverse of model.pca_invert(), implemented via the
    pseudo-inverse of the truncated PCA projection.

    Derivation:
      pca_invert:  X_phys = ((z_norm * ss + sm) @ V.T + mu) * fs + fm
      pca_encode:
        X_std = (X_phys - fm) / fs
        z_u   = (X_std - mu) @ V        [V orthonormal ⟹ V^T V = I_{n_pca}]
        z_norm = (z_u - sm) / ss
    """
    X_std = (X_phys - model.feature_mean) / model.feature_std
    z_u   = (X_std   - model.mu_pca)      @ model.V_pca
    return (z_u - model.score_mean) / model.score_std


def _get_bottleneck(model: nn.Module, x_pca: torch.Tensor,
                    model_type: str) -> torch.Tensor:
    """Return the deterministic point-estimate bottleneck vector."""
    if model_type == 'piae':
        _, y_bott, _, _ = model(x_pca)
        return y_bott
    else:  # cvae — use mu (not sampled z)
        mu, _ = model.encoder(x_pca)
        return mu


def _decode_to_spectrum(model: nn.Module, bott: torch.Tensor,
                        model_type: str) -> torch.Tensor:
    """Bottleneck vector → 2610-dim physical spectrum."""
    if model_type == 'piae':
        z_pca  = model.decoder(bott)
        X_base = model.pca_invert(z_pca)
        return X_base + 0.01 * model.residual_net(X_base)
    else:
        z_pca = model.decoder(bott)
        return model.pca_invert(z_pca)


def _inverse_transform(Y_sc: np.ndarray, y_scalers: dict,
                        targets: list, log_idx: list) -> np.ndarray:
    Y_out = np.zeros_like(Y_sc, dtype=np.float64)
    for i, t in enumerate(targets):
        Y_out[:, i] = y_scalers[t].inverse_transform(Y_sc[:, i:i+1]).ravel()
    for i in log_idx:
        Y_out[:, i] = 10.0 ** Y_out[:, i]
    return Y_out


def _mape(pred_phys: np.ndarray, true_phys: np.ndarray) -> np.ndarray:
    """Per-parameter MAPE, shape (D,)."""
    return (np.abs(pred_phys - true_phys) /
            (np.abs(true_phys) + 1e-8)).mean(axis=0) * 100.


# ─────────────────────────────────────────────────────────────────────────────
# ── PART A: COUNTERFACTUAL INTERVENTIONS + CAUSAL CLOSURE TEST ───────────────
# ─────────────────────────────────────────────────────────────────────────────
def _get_nominal_bottleneck(model: nn.Module, loader: DataLoader,
                             model_type: str, device: torch.device) -> np.ndarray:
    """Compute the test-set median of the bottleneck (used as 'hold fixed' nominal)."""
    bott_list = []
    model.eval()
    with torch.no_grad():
        for x_b, _ in loader:
            bott_list.append(
                _get_bottleneck(model, x_b.to(device), model_type).cpu().numpy())
    return np.median(np.concatenate(bott_list), axis=0)   # (D,)


def _sweep_and_reencode(model: nn.Module, model_type: str,
                         nominal_bott: np.ndarray, param_idx: int,
                         device: torch.device, n_steps: int = 80):
    """
    Sweep bottleneck parameter `param_idx` from its 5th to 95th percentile
    (estimated from the nominal ± 2σ range), holding all others fixed.

    Returns:
        sweep_vals   : (n_steps,) the swept values
        reencoded    : (n_steps, D) re-encoded bottleneck after decode→reencode
    """
    D = len(nominal_bott)

    # Sweep range: use ±1.8 for PIAE ([0,1] Sigmoid space) or ±2.0 for CVAE (Z-score space)
    lo, hi = (0.05, 0.95) if model_type == 'piae' else (-2.0, 2.0)
    sweep_vals = np.linspace(lo, hi, n_steps, dtype=np.float32)

    reencoded  = np.zeros((n_steps, D), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i, sv in enumerate(sweep_vals):
            bott    = nominal_bott.copy()
            bott[param_idx] = sv
            bott_t  = torch.tensor(bott, dtype=torch.float32).unsqueeze(0).to(device)

            X_phys  = _decode_to_spectrum(model, bott_t, model_type)   # (1, 2610)
            x_pca_n = _pca_encode(model, X_phys)                        # (1, n_pca)
            bott_re = _get_bottleneck(model, x_pca_n, model_type)       # (1, D)
            reencoded[i] = bott_re.cpu().numpy()[0]

    return sweep_vals, reencoded


def run_counterfactual_analysis(model: nn.Module, model_type: str,
                                 loader: DataLoader, device: torch.device,
                                 targets: list, plot_dir: str,
                                 n_steps: int = 80):
    """
    Part A — Counterfactual Interventions + Causal Closure.

    For each parameter k, sweeps z_k while holding others at their
    test-set median.  After decode → re-encode, measures:
      (i)  Causal closure: re-encoded[k] ≈ swept[k]  (self-consistency)
      (ii) Independence:  re-encoded[j≠k] ≈ nominal[j] (no crosstalk)

    Produces two figures:
      {model_type}_counterfactual_causal_closure.png
      {model_type}_counterfactual_independence.png
    """
    os.makedirs(plot_dir, exist_ok=True)
    D = len(targets)
    nominal = _get_nominal_bottleneck(model, loader, model_type, device)
    print(f"\n[interp2-A] Counterfactual interventions ({model_type.upper()}, D={D}) …")

    results = {}   # param → (sweep_vals, reencoded)
    for k, name in enumerate(targets):
        print(f"  Sweeping {name} …", end=" ", flush=True)
        sv, re = _sweep_and_reencode(model, model_type, nominal, k, device, n_steps)
        results[name] = (sv, re)
        print(f"max closure error = {np.max(np.abs(re[:, k] - sv)):.4f}")

    # ── Fig 1: Causal closure (diagonal: swept vs re-encoded for each param) ─
    n_cols = min(D, 3)
    n_rows = math.ceil(D / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes_flat = np.array(axes).flatten()

    for k, name in enumerate(targets):
        sv, re = results[name]
        ax = axes_flat[k]
        ax.plot(sv, sv,      'k--', lw=1.2, alpha=0.5, label='Ideal (slope=1)')
        ax.plot(sv, re[:, k], color=_PARAM_COLORS[k % len(_PARAM_COLORS)],
                lw=2.0, label='Re-encoded')
        ax.set_xlabel(f'Intervened value of {name}', fontsize=9)
        ax.set_ylabel(f'Re-encoded {name}', fontsize=9)
        closure_err = np.mean(np.abs(re[:, k] - sv))
        ax.set_title(f'{name}  mean|Δ|={closure_err:.4f}', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    for ax in axes_flat[D:]:
        ax.set_visible(False)

    plt.suptitle(f'Part A — Causal Closure: Enc(Dec(do({name}=α)))[k] ≈ α?\n'
                 f'({model_type.upper()})  Diagonal = perfect self-consistency',
                 fontsize=10, fontweight='bold')
    out1 = os.path.join(plot_dir, f"{model_type}_counterfactual_causal_closure.png")
    # — CSV: causal closure error per parameter
    rows = []
    for k, name in enumerate(targets):
        sv, re = results[name]
        rows.append({'param': name,
                     'closure_error_mean': np.mean(np.abs(re[:, k] - sv)),
                     'closure_error_max':  np.max(np.abs(re[:, k] - sv))})
    csv_save(pd.DataFrame(rows), out1)
    plt.tight_layout(); plt.savefig(out1, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [A] Causal closure → '{out1}'")

    # ── Fig 2: Independence grid (all re-encoded params when sweeping k) ──────
    fig2, axes2 = plt.subplots(D, D, figsize=(3.5 * D, 3.0 * D))
    for ks in range(D):   # source (swept)
        sv, re = results[targets[ks]]
        for ke in range(D):   # effect (re-encoded output)
            ax = axes2[ke, ks]
            c  = _PARAM_COLORS[ke % len(_PARAM_COLORS)]
            ax.plot(sv, re[:, ke], color=c, lw=1.5)
            if ke == ks:
                ax.plot(sv, sv, 'k--', lw=1.0, alpha=0.5)
                ax.set_facecolor('#f0fff0')   # light green for diagonal
            ax.axhline(nominal[ke], color='gray', lw=0.8, linestyle=':', alpha=0.7)
            ax.tick_params(labelsize=6)
            if ke == D - 1: ax.set_xlabel(f'↑ {targets[ks]}', fontsize=8)
            if ks == 0:     ax.set_ylabel(f'Δ {targets[ke]}', fontsize=8)

    plt.suptitle(f'Part A — Intervention Independence Grid\n'
                 f'Row ke, Col ks: re-encoded {targets[0]}..{targets[-1]} '
                 f'when {targets[0]}..{targets[-1]} is swept\n'
                 f'Diagonal (green): self-response.  Off-diagonal: crosstalk.  '
                 f'Dotted: nominal.',
                 fontsize=9, fontweight='bold')
    out2 = os.path.join(plot_dir, f"{model_type}_counterfactual_independence.png")
    # — CSV: full independence grid (source param × effect param × sweep step)
    rows = []
    for ks, src_name in enumerate(targets):
        sv, re = results[src_name]
        for step_i, sv_val in enumerate(sv):
            row = {'source_param': src_name, 'sweep_val': sv_val}
            for ke, eff_name in enumerate(targets):
                row[f're_encoded_{eff_name}'] = re[step_i, ke]
            rows.append(row)
    csv_save(pd.DataFrame(rows), out2)
    plt.tight_layout(); plt.savefig(out2, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [A] Independence grid → '{out2}'")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ── PART B: ROUND-TRIP DECAY ANALYSIS ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def run_roundtrip_decay(model: nn.Module, model_type: str,
                         X_pca_test: np.ndarray, Y_scaled_test: np.ndarray,
                         y_scalers: dict, log_idx: list, targets: list,
                         device: torch.device, plot_dir: str,
                         n_rounds: int = 10, n_samples: int = 400,
                         seed: int = 42):
    """
    Part B — Round-Trip Decay Analysis.

    Starting from each test sample's original PCA-encoded input, repeatedly
    cycles through:
        x_pca_r → encoder → ŷ_r → decoder → z_pca → pca_invert
                → X_2610 → pca_encode → x_pca_{r+1} → encoder → ŷ_{r+1} …

    At each round r, ŷ_r is inverse-transformed to physical space and
    compared against the TRUE ground-truth parameters for that sample.

    A self-consistent model is a fixed point of this map: ŷ_r ≈ ŷ_0 ≈ y_gt.
    Growing error indicates the model is NOT self-consistent.

    Plots:
      {model_type}_roundtrip_mape_per_round.png
      {model_type}_roundtrip_per_param.png
    """
    os.makedirs(plot_dir, exist_ok=True)
    D = len(targets)
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(X_pca_test), min(n_samples, len(X_pca_test)), replace=False)

    X_pca_sub  = torch.tensor(X_pca_test[idx],  dtype=torch.float32).to(device)
    Y_true_sub = Y_scaled_test[idx]   # (n_samples, D) — scaled

    # Compute true physical targets for comparison
    Y_gt_phys = _inverse_transform(Y_true_sub, y_scalers, targets, log_idx)  # (n_samples, D)

    # Storage: mape[round, param]
    mape_all = np.zeros((n_rounds + 1, D))

    print(f"\n[interp2-B] Round-trip decay (n_rounds={n_rounds}, "
          f"n_samples={len(idx)}, model={model_type.upper()}) …")

    model.eval()
    x_cur = X_pca_sub.clone()

    with torch.no_grad():
        for r in range(n_rounds + 1):
            # ── Encode ─────────────────────────────────────────────────────────
            bott = _get_bottleneck(model, x_cur, model_type)   # (n_samples, D)
            bott_np = bott.cpu().numpy()

            # ── Inverse-transform and compute MAPE ────────────────────────────
            pred_phys = _inverse_transform(bott_np, y_scalers, targets, log_idx)
            mape_all[r] = _mape(pred_phys, Y_gt_phys)

            print(f"  Round {r:>2d}  mean_MAPE = "
                  f"{mape_all[r].mean():.2f}%  "
                  f"[{', '.join(f'{m:.1f}' for m in mape_all[r])}]")

            if r < n_rounds:
                # ── Decode → pca_encode → next x_pca ─────────────────────────
                X_phys = _decode_to_spectrum(model, bott, model_type)  # (n, 2610)
                x_cur  = _pca_encode(model, X_phys)                    # (n, n_pca)

    rounds = np.arange(n_rounds + 1)

    # ── Fig 1: Mean MAPE across all params per round ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    mean_mape = mape_all.mean(axis=1)
    ax.plot(rounds, mean_mape, 'ko-', lw=2, ms=7, label='Mean MAPE (all params)')
    ax.axhline(mean_mape[0], color='red', lw=1, linestyle='--',
               label=f'Round-0 baseline = {mean_mape[0]:.2f}%')
    ax.fill_between(rounds, mape_all.min(axis=1), mape_all.max(axis=1),
                    alpha=0.15, color='steelblue', label='Min–Max range')
    ax.set_xlabel('Round-trip iteration', fontsize=11)
    ax.set_ylabel('Mean MAPE (%)', fontsize=11)
    ax.set_title('Round-Trip Decay — Mean MAPE vs Round\n'
                 'Flat line = stable fixed point  |  Rising = unstable', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xticks(rounds)

    # ── Fig 2: Per-parameter MAPE per round ───────────────────────────────────
    ax2 = axes[1]
    for k, name in enumerate(targets):
        ax2.plot(rounds, mape_all[:, k], 'o-',
                 color=_PARAM_COLORS[k % len(_PARAM_COLORS)],
                 lw=1.8, ms=5, label=name)
    ax2.set_xlabel('Round-trip iteration', fontsize=11)
    ax2.set_ylabel('MAPE (%)', fontsize=11)
    ax2.set_title('Round-Trip Decay — Per-parameter MAPE\n'
                  'Stable parameters: flat.  Unstable: monotone rise.', fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)

    plt.suptitle(f'Part B — Round-Trip Decay Analysis ({model_type.upper()})\n'
                 f'Encode → decode → pca_invert → pca_encode → repeat {n_rounds}× '
                 f'  (n={len(idx)} test samples)',
                 fontsize=10, fontweight='bold')
    out1 = os.path.join(plot_dir, f"{model_type}_roundtrip_mape_per_round.png")
    # — CSV: MAPE per round and per parameter
    rt_df = pd.DataFrame(mape_all, columns=targets)
    rt_df.insert(0, 'round', rounds)
    csv_save(rt_df, out1)
    plt.tight_layout(); plt.savefig(out1, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [B] Round-trip decay → '{out1}'")

    # ── Fig 3: Normalised deviation (MAPE_r / MAPE_0) — contraction test ─────
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    for k, name in enumerate(targets):
        base = mape_all[0, k] + 1e-6
        ax3.plot(rounds, mape_all[:, k] / base,
                 'o-', color=_PARAM_COLORS[k % len(_PARAM_COLORS)],
                 lw=1.8, ms=5, label=name)
    ax3.axhline(1.0, color='black', lw=1, linestyle='--', label='Baseline (round 0)')
    ax3.axhline(1.1, color='orange', lw=0.8, linestyle=':', alpha=0.8, label='+10%')
    ax3.axhline(2.0, color='red',    lw=0.8, linestyle=':', alpha=0.8, label='×2')
    ax3.set_xlabel('Round-trip iteration', fontsize=11)
    ax3.set_ylabel('MAPE_r / MAPE_0  (normalised)', fontsize=11)
    ax3.set_title(f'Contraction Test  ({model_type.upper()})\n'
                  f'< 1.0 = model is a contraction (improving)  '
                  f'|  > 1.0 = expanding (degrading)', fontsize=10)
    ax3.legend(fontsize=9, ncol=2); ax3.grid(True, alpha=0.3)
    ax3.set_xticks(rounds)

    out2 = os.path.join(plot_dir, f"{model_type}_roundtrip_contraction.png")
    # — CSV: normalised MAPE (relative to round-0 baseline)
    norm_df = pd.DataFrame({'round': rounds})
    for k, name in enumerate(targets):
        norm_df[f'{name}_norm'] = mape_all[:, k] / (mape_all[0, k] + 1e-6)
    csv_save(norm_df, out2)
    plt.tight_layout(); plt.savefig(out2, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [B] Contraction test  → '{out2}'")

    return mape_all


# ─────────────────────────────────────────────────────────────────────────────
# ── PART C: RESONANCE PROBING ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def _compute_derived_quantities(Y_phys: np.ndarray, targets: list,
                                 model_type: str) -> dict:
    """
    Compute derived physical circuit quantities from true parameter values.

    Physical units assumed (matching LHS bounds in data generation):
      Rbv : Ω          Cbv : fF (× 1e-15 F)
      Rdv : Ω          Cdv : fF
      Rav : Ω
      Ldv : nH (× 1e-9 H)  [CVAE only]

    Returns dict:  name → (N,) array of derived values
    """
    idx = {t: targets.index(t) for t in targets if t in targets}
    N   = len(Y_phys)
    derived = {}

    if 'Rdv' in idx and 'Cdv' in idx:
        Rd, Cd = Y_phys[:, idx['Rdv']], Y_phys[:, idx['Cdv']]
        tau_d  = Rd * Cd                             # Ω·fF  (∝ time in ps range)
        f_d    = 1.0 / (2 * np.pi * Rd * Cd * 1e-6) # GHz
        derived['tau_d (Ω·fF)']          = np.log10(tau_d + 1e-30)
        derived['f_d_corner (GHz, log)'] = np.log10(np.abs(f_d) + 1e-30)

    if 'Rbv' in idx and 'Cbv' in idx:
        Rb, Cb = Y_phys[:, idx['Rbv']], Y_phys[:, idx['Cbv']]
        tau_b  = Rb * Cb
        f_b    = 1.0 / (2 * np.pi * Rb * Cb * 1e-6)
        derived['tau_b (Ω·fF)']          = np.log10(tau_b + 1e-30)
        derived['f_b_corner (GHz, log)'] = np.log10(np.abs(f_b) + 1e-30)

    if 'Rdv' in idx and 'Rav' in idx:
        Rd, Ra = Y_phys[:, idx['Rdv']], Y_phys[:, idx['Rav']]
        derived['log(Rd/Ra)'] = np.log10(Rd / (Ra + 1e-8))

    if 'Rbv' in idx and 'Rdv' in idx:
        Rb, Rd = Y_phys[:, idx['Rbv']], Y_phys[:, idx['Rdv']]
        derived['log(Rb/Rd)'] = np.log10(Rb / (Rd + 1e-8))

    if model_type == 'cvae' and 'Ldv' in idx and 'Cdv' in idx:
        Ld, Cd = Y_phys[:, idx['Ldv']], Y_phys[:, idx['Cdv']]
        # f_j in GHz: 1/(2π√(Ld·nH × Cd·fF × 1e-24))
        with np.errstate(divide='ignore', invalid='ignore'):
            f_j = 1.0 / (2 * np.pi * np.sqrt(np.abs(Ld * Cd) * 1e-24)) * 1e-9
        derived['f_junction (GHz, log)'] = np.log10(np.abs(f_j) + 1e-30)

    return derived


class _HookStore:
    """Context manager that hooks all ReLU/GELU/Sigmoid activations in a module."""
    def __init__(self, model: nn.Module, model_type: str):
        self.acts     = {}    # name → last output tensor
        self.handles  = []
        self.model    = model
        self.model_type = model_type

    def __enter__(self):
        enc = self.model.encoder
        net = enc.net if self.model_type == 'piae' else enc.shared

        for i, layer in enumerate(net):
            if isinstance(layer, (nn.ReLU, nn.GELU, nn.Sigmoid)):
                name = f'enc_act_{i}'
                def _hook(mod, inp, out, _n=name):
                    self.acts[_n] = out.detach().cpu()
                self.handles.append(layer.register_forward_hook(_hook))

        if self.model_type == 'cvae':
            def _mu_hook(mod, inp, out):
                self.acts['enc_mu'] = out.detach().cpu()
            self.handles.append(enc.fc_mu.register_forward_hook(_mu_hook))

        return self

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _collect_activations(model: nn.Module, model_type: str,
                          loader: DataLoader, device: torch.device) -> dict:
    """
    Run the full test loader once with hooks active.
    Returns dict: layer_name → (N_test, hidden_dim) numpy array.
    """
    all_acts  = {}
    n_batches = 0

    model.eval()
    with _HookStore(model, model_type) as store:
        with torch.no_grad():
            for x_b, _ in loader:
                _ = _get_bottleneck(model, x_b.to(device), model_type)
                # Accumulate activations from this batch
                for k, v in store.acts.items():
                    all_acts.setdefault(k, []).append(v.numpy())
                n_batches += 1

    return {k: np.concatenate(v, axis=0) for k, v in all_acts.items()}


def run_resonance_probing(model: nn.Module, model_type: str,
                           loader: DataLoader, device: torch.device,
                           targets: list, y_scalers: dict,
                           log_idx: list, Y_scaled_test: np.ndarray,
                           plot_dir: str, seed: int = 42):
    """
    Part C — Resonance Probing.

    Trains linear (Ridge) and nonlinear (MLP) probes at each encoder
    activation layer to predict derived physical quantities (f_d, f_b,
    τ_d, τ_b, Rd/Ra, …).  If R² increases monotonically with layer depth,
    the encoder progressively builds up physically meaningful representations.

    Produces:
      {model_type}_resonance_probe_r2_vs_depth.png
      {model_type}_resonance_probe_heatmap.png
    """
    os.makedirs(plot_dir, exist_ok=True)

    # ── Compute derived quantities from true params ────────────────────────────
    Y_phys = _inverse_transform(Y_scaled_test, y_scalers, targets, log_idx)
    derived = _compute_derived_quantities(Y_phys, targets, model_type)
    print(f"\n[interp2-C] Resonance probing  (derived quantities: "
          f"{list(derived.keys())}) …")

    if not derived:
        print("  No derived quantities could be computed.  Skipping.")
        return {}

    # ── Collect hidden activations ─────────────────────────────────────────────
    print("  Collecting hidden layer activations …")
    acts = _collect_activations(model, model_type, loader, device)
    layer_names = sorted(acts.keys())
    print(f"  Layers hooked: {layer_names} "
          f"  dims={[acts[k].shape[1] for k in layer_names]}")

    # ── Probe training ─────────────────────────────────────────────────────────
    N   = min(len(list(acts.values())[0]), len(Y_phys))
    rng = np.random.default_rng(seed)
    split_idx = rng.permutation(N)
    tr_i = split_idx[:int(0.8 * N)]
    te_i = split_idx[int(0.8 * N):]

    r2_linear    = {}   # derived_qty → list of R² per layer
    r2_nonlinear = {}

    for qty_name, qty_vals in derived.items():
        y_qty = qty_vals[:N]
        y_tr, y_te = y_qty[tr_i], y_qty[te_i]
        r2_lin_layers, r2_mlp_layers = [], []

        for lname in layer_names:
            X_l  = acts[lname][:N]
            # Standardise activations for stable probe training
            sc   = _SS(); sc.fit(X_l[tr_i])
            X_tr = sc.transform(X_l[tr_i]);  X_te = sc.transform(X_l[te_i])

            # Linear probe (Ridge with fixed α)
            lin = Ridge(alpha=1.0).fit(X_tr, y_tr)
            r2_lin_layers.append(r2_score(y_te, lin.predict(X_te)))

            # Nonlinear probe (small MLP — 2 hidden layers)
            mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                                random_state=seed, early_stopping=True,
                                validation_fraction=0.1)
            mlp.fit(X_tr, y_tr)
            r2_mlp_layers.append(r2_score(y_te, mlp.predict(X_te)))

        r2_linear[qty_name]    = r2_lin_layers
        r2_nonlinear[qty_name] = r2_mlp_layers
        print(f"  {qty_name:30s}  "
              f"linear  R²: {[f'{v:.3f}' for v in r2_lin_layers]}  |  "
              f"MLP R²: {[f'{v:.3f}' for v in r2_mlp_layers]}")

    # ── Fig 1: R² vs layer depth per derived quantity ─────────────────────────
    n_qty    = len(derived)
    n_layers = len(layer_names)
    layer_xs = np.arange(n_layers)
    qty_colors = plt.cm.tab10(np.linspace(0, 1, n_qty))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for qi, (qty_name, clr) in enumerate(zip(derived.keys(), qty_colors)):
        axes[0].plot(layer_xs, r2_linear[qty_name],    'o-', color=clr,
                     lw=1.8, ms=6, label=qty_name)
        axes[1].plot(layer_xs, r2_nonlinear[qty_name], 's--', color=clr,
                     lw=1.8, ms=6, label=qty_name)

    for ax, title in zip(axes, ['Linear probe (Ridge)', 'Nonlinear probe (MLP)']):
        ax.axhline(0, color='gray', lw=0.8, linestyle=':')
        ax.set_xticks(layer_xs)
        ax.set_xticklabels([ln.replace('enc_', '') for ln in layer_names],
                            fontsize=8, rotation=20)
        ax.set_xlabel('Encoder layer', fontsize=10)
        ax.set_ylabel('R²  (probe accuracy on held-out test)', fontsize=10)
        ax.set_title(f'{title}\n'
                     f'Rising R² = layer progressively encodes physics quantity',
                     fontsize=9)
        ax.legend(fontsize=7, ncol=1)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.1, 1.05)

    plt.suptitle(f'Part C — Resonance Probing: Do Hidden Layers Encode Physics?\n'
                 f'({model_type.upper()})',
                 fontsize=10, fontweight='bold')
    out1 = os.path.join(plot_dir, f"{model_type}_resonance_probe_r2_vs_depth.png")
    # — CSV: R² per derived quantity × layer
    lin_df = pd.DataFrame(r2_linear,  index=layer_names).T.reset_index().rename(columns={'index':'qty'})
    mlp_df = pd.DataFrame(r2_nonlinear, index=layer_names).T.reset_index().rename(columns={'index':'qty'})
    csv_save(lin_df, out1, suffix='__linear')
    csv_save(mlp_df, out1, suffix='__mlp')
    plt.tight_layout(); plt.savefig(out1, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [C] R² vs depth → '{out1}'")

    # ── Fig 2: R² heatmap (derived_qty × layer) ───────────────────────────────
    mat_lin = np.array([r2_linear[q]    for q in derived.keys()])
    mat_mlp = np.array([r2_nonlinear[q] for q in derived.keys()])
    qty_labels = list(derived.keys())

    fig2, axes2 = plt.subplots(1, 2, figsize=(max(10, n_layers * 1.6), max(5, n_qty * 0.9)))
    for ax, mat, title in zip(axes2,
                               [mat_lin, mat_mlp],
                               ['Linear probe R²', 'MLP probe R²']):
        im = ax.imshow(mat, cmap='RdYlGn', vmin=-0.1, vmax=1.0, aspect='auto')
        ax.set_xticks(layer_xs)
        ax.set_xticklabels([ln.replace('enc_', '') for ln in layer_names],
                            fontsize=8, rotation=20)
        ax.set_yticks(range(n_qty))
        ax.set_yticklabels(qty_labels, fontsize=9)
        plt.colorbar(im, ax=ax, label='R²', fraction=0.05)
        ax.set_title(f'{title}', fontsize=10)
        ax.set_xlabel('Encoder layer', fontsize=9)
        for i in range(n_qty):
            for j in range(n_layers):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=8,
                        color='white' if mat[i,j] > 0.7 or mat[i,j] < -0.05 else 'black')

    plt.suptitle(f'Part C — Resonance Probe R² Heatmap\n'
                 f'Green = physics quantity is linearly/nonlinearly encoded at this layer',
                 fontsize=10, fontweight='bold')
    out2 = os.path.join(plot_dir, f"{model_type}_resonance_probe_heatmap.png")
    # — CSV: linear and MLP R² matrices
    csv_save(pd.DataFrame(mat_lin, index=qty_labels,
                          columns=[ln.replace('enc_','') for ln in layer_names]), out2, suffix='__linear')
    csv_save(pd.DataFrame(mat_mlp, index=qty_labels,
                          columns=[ln.replace('enc_','') for ln in layer_names]), out2, suffix='__mlp')
    plt.tight_layout(); plt.savefig(out2, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [C] Heatmap → '{out2}'")

    return {'linear': r2_linear, 'nonlinear': r2_nonlinear,
            'layer_names': layer_names}


# ─────────────────────────────────────────────────────────────────────────────
# ── PART D: WEIGHT MATRIX SVD ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def _effective_rank(singular_vals: np.ndarray) -> float:
    """
    Effective rank via the entropy of the normalised singular value distribution.
    r_eff = exp(H(p))  where  p_i = σ_i / Σσ_j
    A rank-r matrix has r_eff = r; diffuse spectra have high r_eff.
    """
    sv  = np.abs(singular_vals)
    sv  = sv[sv > 1e-12]
    if len(sv) == 0:
        return 0.0
    p   = sv / sv.sum()
    H   = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(H))


def _collect_linear_layers(model: nn.Module, model_type: str):
    """
    Returns list of (name, weight_np) for all Linear layers in encoder + decoder.
    """
    layers = []

    # Encoder
    enc    = model.encoder
    net    = enc.net if model_type == 'piae' else enc.shared
    for i, layer in enumerate(net):
        if isinstance(layer, nn.Linear):
            layers.append((f'enc_L{i}', layer.weight.detach().cpu().numpy()))
    if model_type == 'cvae':
        layers.append(('enc_fc_mu',     enc.fc_mu.detach().cpu().numpy()
                        if hasattr(enc.fc_mu, 'detach')
                        else enc.fc_mu.weight.detach().cpu().numpy()))
        layers.append(('enc_fc_logvar', enc.fc_logvar.weight.detach().cpu().numpy()))

    # Decoder
    dec = model.decoder
    net_dec = dec.net if hasattr(dec, 'net') else dec
    for i, layer in enumerate(net_dec):
        if isinstance(layer, nn.Linear):
            layers.append((f'dec_L{i}', layer.weight.detach().cpu().numpy()))

    return layers


def run_weight_svd(model: nn.Module, model_type: str,
                   targets: list, plot_dir: str):
    """
    Part D — Weight Matrix SVD.

    For every Linear layer in the encoder and decoder:
      1. Computes the SVD W = U Σ V^T.
      2. Plots the singular value spectrum (log scale).
      3. Computes the effective rank r_eff.
      4. For the FIRST encoder layer: aligns the right singular vectors
         (in PCA space) with the standard basis to show which PCA
         components each input mode emphasises most.

    Produces:
      {model_type}_weight_svd_spectra.png
      {model_type}_weight_svd_first_layer_alignment.png
    """
    os.makedirs(plot_dir, exist_ok=True)
    named_layers = _collect_linear_layers(model, model_type)
    print(f"\n[interp2-D] Weight SVD ({model_type.upper()}, "
          f"{len(named_layers)} Linear layers) …")

    svd_results = {}
    for name, W in named_layers:
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        r_eff = _effective_rank(S)
        svd_results[name] = {'U': U, 'S': S, 'Vt': Vt, 'r_eff': r_eff,
                              'shape': W.shape}
        print(f"  {name:20s}  shape={W.shape}  "
              f"rank_eff={r_eff:.1f}  σ_max={S[0]:.4f}  σ_min={S[-1]:.4e}")

    # ── Fig 1: Singular value spectra (all layers in one figure) ─────────────
    n_layers = len(named_layers)
    n_cols   = min(n_layers, 4)
    n_rows   = math.ceil(n_layers / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 3.5 * n_rows))
    axes_flat = np.array(axes).flatten()

    for ax_i, (name, _) in enumerate(named_layers):
        ax  = axes_flat[ax_i]
        res = svd_results[name]
        S   = res['S']
        ax.semilogy(S, 'o-', ms=4, lw=1.5, color='steelblue')
        ax.set_title(f'{name}\n'
                     f'shape={res["shape"]}  r_eff={res["r_eff"]:.1f}',
                     fontsize=8)
        ax.set_xlabel('Singular value index', fontsize=8)
        ax.set_ylabel('σ (log scale)', fontsize=8)
        ax.grid(True, alpha=0.25, which='both')
        # Annotate effective rank
        r_int = max(1, int(round(res['r_eff'])))
        if r_int < len(S):
            ax.axvline(r_int, color='crimson', lw=1, linestyle='--', alpha=0.7,
                       label=f'r_eff≈{res["r_eff"]:.1f}')
            ax.legend(fontsize=7)

    for ax in axes_flat[n_layers:]:
        ax.set_visible(False)

    # Summary bar chart in top-right corner
    if n_layers < len(axes_flat):
        ax_sum = axes_flat[n_layers]
        ax_sum.set_visible(True)
        ax_sum.bar(range(n_layers),
                   [svd_results[n]['r_eff'] for n, _ in named_layers],
                   color='steelblue', alpha=0.8, edgecolor='k', lw=0.6)
        ax_sum.set_xticks(range(n_layers))
        ax_sum.set_xticklabels([n for n, _ in named_layers],
                                rotation=45, ha='right', fontsize=7)
        ax_sum.set_ylabel('Effective rank r_eff', fontsize=9)
        ax_sum.set_title('Effective rank summary\n'
                         'Higher = more distributed representation', fontsize=8)
        ax_sum.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Part D — Weight Matrix Singular Value Spectra\n'
                 f'({model_type.upper()})  Rapid drop-off = low-rank structure',
                 fontsize=10, fontweight='bold')
    out1 = os.path.join(plot_dir, f"{model_type}_weight_svd_spectra.png")
    # — CSV: effective rank and top singular values per layer
    rows = [{'layer': name,
             'shape': str(res['shape']),
             'effective_rank': res['r_eff'],
             'sigma_max': res['S'][0],
             'sigma_min': res['S'][-1]}
            for name, res in svd_results.items()]
    csv_save(pd.DataFrame(rows), out1, suffix='__summary')
    # — also export full singular value spectra
    sv_dict = {'sv_index': list(range(max(len(res['S']) for res in svd_results.values())))}
    for name, res in svd_results.items():
        s_padded = np.pad(res['S'],
                          (0, len(sv_dict['sv_index']) - len(res['S'])),
                          constant_values=np.nan)
        sv_dict[name] = s_padded
    csv_save(sv_dict, out1, suffix='__spectra')
    plt.tight_layout(); plt.savefig(out1, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [D] SVD spectra → '{out1}'")

    # ── Fig 2: First encoder layer — right singular vector alignment ──────────
    # The first encoder linear layer maps (n_pca) → (hidden_dim).
    # Its right singular vectors Vt (shape: min(d_in,d_out), d_in) are in PCA space.
    # We look at which PCA component index each singular mode emphasises most.
    enc    = model.encoder
    net    = enc.net if model_type == 'piae' else enc.shared
    first_linear_W = None
    for layer in net:
        if isinstance(layer, nn.Linear):
            first_linear_W = layer.weight.detach().cpu().numpy()
            break

    if first_linear_W is not None:
        _, S_first, Vt_first = np.linalg.svd(first_linear_W, full_matrices=False)
        n_pca     = Vt_first.shape[1]
        n_show    = min(n_pca, 10)   # show top-n_show singular modes
        pc_labels = [f'PC{i+1}' for i in range(n_pca)]

        # Alignment matrix: |Vt[mode_k, pc_j]|  — which PC does mode k point to?
        align_mat = np.abs(Vt_first[:n_show])   # (n_show, n_pca)

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

        im = axes2[0].imshow(align_mat, cmap='YlOrRd', aspect='auto',
                              vmin=0, vmax=align_mat.max())
        axes2[0].set_xticks(range(n_pca))
        axes2[0].set_xticklabels(pc_labels, fontsize=8, rotation=45)
        axes2[0].set_yticks(range(n_show))
        axes2[0].set_yticklabels([f'SV{i+1}' for i in range(n_show)], fontsize=9)
        plt.colorbar(im, ax=axes2[0], label='|component|')
        axes2[0].set_title('First encoder layer: |right singular vector|\n'
                            'Each row = one input mode; bright = which PC it uses',
                            fontsize=9)
        for i in range(n_show):
            for j in range(n_pca):
                axes2[0].text(j, i, f'{align_mat[i,j]:.2f}', ha='center',
                              va='center', fontsize=7,
                              color='white' if align_mat[i,j] > 0.6 else 'black')

        # Dominant PC per singular mode
        dominant_pc = np.argmax(align_mat, axis=1)
        bar_vals    = align_mat[np.arange(n_show), dominant_pc]
        axes2[1].barh(range(n_show), bar_vals, color='steelblue', alpha=0.8)
        axes2[1].set_yticks(range(n_show))
        axes2[1].set_yticklabels([f'SV{i+1} → PC{dominant_pc[i]+1}'
                                   for i in range(n_show)], fontsize=9)
        axes2[1].set_xlabel('|alignment| with dominant PC', fontsize=10)
        axes2[1].set_title('Dominant PCA component per singular mode\n'
                            'High alignment = encoder first layer ≈ PC selector',
                            fontsize=9)
        axes2[1].grid(axis='x', alpha=0.3)
        axes2[1].set_xlim(0, 1)
        axes2[1].invert_yaxis()

        plt.suptitle(f'Part D — First Encoder Layer SVD: PCA Alignment\n'
                     f'({model_type.upper()}, layer shape={first_linear_W.shape})\n'
                     f'Sparse alignment = first layer selects specific PCs',
                     fontsize=10, fontweight='bold')
        out2 = os.path.join(plot_dir,
                             f"{model_type}_weight_svd_first_layer_alignment.png")
        # — CSV: alignment matrix (SV × PC)
        csv_save(pd.DataFrame(align_mat,
                              index=[f'SV{i+1}' for i in range(n_show)],
                              columns=pc_labels), out2)
        plt.tight_layout(); plt.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [D] First-layer alignment → '{out2}'")
    return svd_results


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner  (mirrors interpretability.run())
# ─────────────────────────────────────────────────────────────────────────────
def run(model_path:  str,
        model_type:  str,
        data_dir:    str = "data/raw",
        pca_dir:     str = "data/pca_artifacts",
        proc_dir:    str = "data/processed",
        splits_dir:  str = "data/splits",
        plot_dir:    str = "outputs/plots/interp2",
        n_rounds:    int = 10,
        seed:        int = 42):

    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt      = torch.load(model_path, map_location=device, weights_only=False)
    dim_pca   = ckpt["dim_pca"]
    targets   = ckpt["target_cols"]
    y_scalers = ckpt["y_scalers"]
    log_idx   = [targets.index(t) for t in ckpt.get("log_targets", [])]

    # ── Rebuild model ─────────────────────────────────────────────────────────
    from piae_model import PIAE
    from cvae_model import PhysicsSupervisedCVAE

    sp    = np.load(os.path.join(pca_dir, "pca_score_scaler_params.npz"))
    V_np  = np.load(os.path.join(pca_dir, "V_pca_bridge.npy")).astype(np.float32)
    mu_np = np.load(os.path.join(pca_dir, "mu_scaler_bridge.npy")).astype(np.float32)
    fs_np = np.load(os.path.join(pca_dir, "std_scaler_bridge.npy")).astype(np.float32)
    fm_np = np.load(os.path.join(pca_dir, "scaler_mean_bridge.npy")).astype(np.float32)
    sm_np = sp["score_mean"].astype(np.float32)
    ss_np = sp["score_std"].astype(np.float32)
    buffers = (torch.tensor(V_np), torch.tensor(mu_np), torch.tensor(sm_np),
               torch.tensor(ss_np), torch.tensor(fm_np), torch.tensor(fs_np))

    model = (PIAE(*buffers, dim_pca=dim_pca) if model_type == 'piae'
             else PhysicsSupervisedCVAE(*buffers, dim_pca=dim_pca)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"\n[interp2] Loaded {model_type.upper()} | DIM_PCA={dim_pca} | "
          f"targets={targets}")

    # ── Load PCA-space data ────────────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler as _SS2, MinMaxScaler as _MMS
    df_pca   = pd.read_csv(os.path.join(proc_dir, "ssec_pca_final_v2.csv"))
    pc_cols  = [f"PC{i+1}" for i in range(dim_pca)]
    X_pca    = df_pca[pc_cols].values.astype(np.float32)
    Y_phys_all = df_pca[targets].values.astype(np.float32)
    test_idx   = np.load(os.path.join(splits_dir, "split_test_idx.npy"))

    X_pca_test = X_pca[test_idx]
    Y_phys_test = Y_phys_all[test_idx].astype(np.float64)

    # Rebuild scaled Y for test set (mirrors training cell)
    Y_proc = Y_phys_test.copy()
    for i in log_idx:
        Y_proc[:, i] = np.log10(np.abs(Y_proc[:, i]) + 1e-30)
    Y_scaled_test = np.zeros_like(Y_proc, dtype=np.float32)
    for i, t in enumerate(targets):
        Y_scaled_test[:, i] = y_scalers[t].transform(
            Y_proc[:, i:i+1]).ravel()

    # DataLoader over test PCA scores
    class _DS(Dataset):
        def __init__(self, X):
            self.X = torch.tensor(X, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], 0

    loader = DataLoader(
        Subset(_DS(X_pca), test_idx.tolist()),
        batch_size=256, shuffle=False,
        pin_memory=True, num_workers=2, persistent_workers=True)

    # ── Part A: Counterfactual interventions ──────────────────────────────────
    run_counterfactual_analysis(
        model, model_type, loader, device, targets,
        plot_dir=os.path.join(plot_dir, "A_counterfactual"))

    # ── Part B: Round-trip decay ──────────────────────────────────────────────
    run_roundtrip_decay(
        model, model_type,
        X_pca_test, Y_scaled_test,
        y_scalers, log_idx, targets,
        device, plot_dir=os.path.join(plot_dir, "B_roundtrip"),
        n_rounds=n_rounds, seed=seed)

    # ── Part C: Resonance probing ─────────────────────────────────────────────
    run_resonance_probing(
        model, model_type, loader, device,
        targets, y_scalers, log_idx, Y_scaled_test,
        plot_dir=os.path.join(plot_dir, "C_resonance"))

    # ── Part D: Weight SVD ────────────────────────────────────────────────────
    run_weight_svd(
        model, model_type, targets,
        plot_dir=os.path.join(plot_dir, "D_svd"))

    print(f"\n[interp2] ✅  All Part-2 interpretability analyses complete.")
    print(f"  Plots written to subdirectories of: {plot_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SSEC Interpretability Part 2")
    p.add_argument("--model-path",  required=True)
    p.add_argument("--model-type",  required=True, choices=["piae", "cvae"])
    p.add_argument("--data-dir",    default="data/raw")
    p.add_argument("--pca-dir",     default="data/pca_artifacts")
    p.add_argument("--proc-dir",    default="data/processed")
    p.add_argument("--splits-dir",  default="data/splits")
    p.add_argument("--plot-dir",    default="outputs/plots/interp2")
    p.add_argument("--n-rounds",    type=int, default=10)
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    run(model_path  = args.model_path,
        model_type  = args.model_type,
        data_dir    = args.data_dir,
        pca_dir     = args.pca_dir,
        proc_dir    = args.proc_dir,
        splits_dir  = args.splits_dir,
        plot_dir    = args.plot_dir,
        n_rounds    = args.n_rounds,
        seed        = args.seed)