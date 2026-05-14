# -*- coding: utf-8 -*-
"""
interpretability.py
===================
Interpretability methods adapted for the 2610-dim (S11+S21+S22) configuration.
Supports both PIAE and CVAE models.

Key methods:
  1. Encoder Crosstalk Matrix (Jacobian of decoder)
  2. Poincare / Smith Chart Locus (Traversing the latent space)
  3. Integrated Gradients (Raw frequency attribution)
"""

import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NF = 435
DIM_RAW = 2610

# Note: The model itself knows its targets, but we provide defaults.
PIAE_TARGETS = ['Rbv', 'Cbv', 'Rdv', 'Cdv', 'Rav']
CVAE_TARGETS = ['Rbv', 'Cbv', 'Rdv', 'Ldv', 'Cdv', 'Rav']

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
class ForwardWrapper(torch.nn.Module):
    """Wraps model to accept raw 2610-dim data, returning bottleneck params."""
    def __init__(self, model, model_type="piae"):
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, x_raw):
        # Apply PCA inside PyTorch graph
        # V_pca shape: (DIM_RAW, n_pca) → forward proj: (x_std - mu_pca) @ V_pca
        x_std        = (x_raw - self.model.feature_mean) / self.model.feature_std
        x_pca_unnorm = (x_std - self.model.mu_pca) @ self.model.V_pca   # (B, n_pca)
        x_pca        = (x_pca_unnorm - self.model.score_mean) / self.model.score_std

        if self.model_type == "piae":
            _, y_bott, _, _ = self.model(x_pca)
            return y_bott
        elif self.model_type == "cvae":
            mu, _ = self.model.encoder(x_pca)
            return mu
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Encoder Crosstalk (Jacobian)
# ─────────────────────────────────────────────────────────────────────────────
def compute_crosstalk_matrix(model, model_type, loader, device, targets, plot_dir):
    """Computes and plots the Jacobian of the bottleneck w.r.t PCA inputs."""
    model.eval()
    dim_phys = len(targets)
    dim_pca = model.encoder.net[0].in_features if model_type == "piae" else model.encoder.shared[0].in_features
    J_acc = np.zeros((dim_phys, dim_pca))
    n_samples = 0

    print("\n[interp] Computing Jacobian for Crosstalk Matrix...")
    for x_b, _ in loader:
        x_b = x_b.to(device).requires_grad_(True)
        if model_type == "piae":
            y_b = model.encoder(x_b)
        else:
            y_b, _ = model.encoder(x_b)

        for i in range(dim_phys):
            grad = torch.autograd.grad(
                outputs=y_b[:, i].sum(), inputs=x_b,
                retain_graph=True, create_graph=False)[0]
            J_acc[i] += torch.abs(grad).sum(dim=0).cpu().numpy()
        n_samples += len(x_b)

    J_mean = J_acc / n_samples
    J_norm = J_mean / J_mean.max(axis=1, keepdims=True)

    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    import seaborn as sns
    sns.heatmap(J_norm, annot=True, cmap='Blues', fmt=".2f",
                xticklabels=[f'PC{i+1}' for i in range(dim_pca)],
                yticklabels=targets)
    plt.title(f'Encoder Sensitivity to Principal Components ({model_type.upper()})')
    plt.ylabel('Bottleneck Parameter')
    plt.xlabel('Input PC')
    out = os.path.join(plot_dir, f"{model_type}_crosstalk_matrix.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] Crosstalk matrix → '{out}'")

# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Integrated Gradients
# ─────────────────────────────────────────────────────────────────────────────
class IntegratedGradientsRaw:
    def __init__(self, wrapper_model):
        self.model = wrapper_model
        self.model.eval()

    def generate(self, inputs, baseline, target_idx, steps=50):
        self.model.zero_grad()
        inputs = inputs.clone().detach().requires_grad_(False)
        baseline = baseline.clone().detach().requires_grad_(False)

        alphas = torch.linspace(0.0, 1.0, steps=steps, device=inputs.device).view(-1, 1)
        path = baseline + alphas * (inputs - baseline)
        path.requires_grad_(True)

        outputs = self.model(path)
        target_outputs = outputs[:, target_idx]

        self.model.zero_grad()
        target_outputs.sum().backward()
        grads = path.grad

        avg_grads = torch.mean(grads[:-1] + grads[1:], dim=0) / 2.0
        ig = (inputs - baseline) * avg_grads
        return ig

def plot_integrated_gradients(model, model_type, X_raw, Y_raw, device, targets, plot_dir):
    """Computes IG w.r.t the raw 2610-D input for one representative sample."""
    wrapper = ForwardWrapper(model, model_type).to(device)
    ig_explainer = IntegratedGradientsRaw(wrapper)

    # Pick a random sample near the median
    idx = len(X_raw) // 2
    x_sample = torch.tensor(X_raw[idx:idx+1], dtype=torch.float32, device=device)
    baseline = torch.zeros_like(x_sample)

    print(f"\n[interp] Computing Integrated Gradients for {model_type.upper()}...")
    attributions = []
    for i, t in enumerate(targets):
        ig = ig_explainer.generate(x_sample, baseline, target_idx=i, steps=50)
        attributions.append(ig[0].cpu().detach().numpy())
    attributions = np.array(attributions)

    fq = np.linspace(0.04, 43.5, NF)
    fig, axes = plt.subplots(len(targets), 6, figsize=(24, 2.5 * len(targets)))
    panels = [('S11 Re', slice(0, NF)), ('S11 Im', slice(NF, 2*NF)),
              ('S21 Re', slice(2*NF, 3*NF)), ('S21 Im', slice(3*NF, 4*NF)),
              ('S22 Re', slice(4*NF, 5*NF)), ('S22 Im', slice(5*NF, DIM_RAW))]

    for row, target in enumerate(targets):
        attr = attributions[row]
        vmax = np.max(np.abs(attr)) + 1e-9
        for col, (title, sl) in enumerate(panels):
            ax = axes[row, col]
            ax.bar(fq, attr[sl], color='teal', width=0.1)
            ax.set_ylim(-vmax, vmax)
            if row == 0: ax.set_title(title)
            if col == 0: ax.set_ylabel(f'Attr → {target}', fontsize=9)

    plt.suptitle(f'Integrated Gradients Frequency Attribution ({model_type.upper()})', fontsize=12)
    out = os.path.join(plot_dir, f"{model_type}_integrated_gradients.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] Integrated Gradients → '{out}'")

# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA as sklearn_PCA

# ─────────────────────────────────────────────────────────────────────────────
# A.0  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def poincare_distance(g1, g2):
    num   = np.abs(g1 - g2)
    denom = np.abs(1.0 - np.conj(g1) * g2)
    ratio = np.clip(num / (denom + 1e-12), 0.0, 1.0 - 1e-9)
    return 2.0 * np.arctanh(ratio)


def draw_smith_background(ax, title=""):
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1.2, alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.3)
    for r in [0.2, 0.5, 1.0, 2.0]:
        cx, rad = r/(1+r), 1/(1+r)
        ax.add_patch(plt.Circle((cx, 0), rad, fill=False, color='lightgray', lw=0.5))
    for x in [0.5, 1.0, 2.0]:
        cy, rad = 1/x, 1/x
        for sign in [1, -1]:
            th = np.linspace(0, np.pi, 200)
            xs = 1 + rad * np.cos(th)
            ys = sign * cy + rad * np.sin(th) * sign
            mask = xs**2 + ys**2 <= 1.01
            ax.plot(xs[mask], ys[mask], color='lightgray', lw=0.5)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"Re($\Gamma$)", fontsize=8)
    ax.set_ylabel(r"Im($\Gamma$)", fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# A.1  Smith Chart Traversal
# ─────────────────────────────────────────────────────────────────────────────
def _bott_to_X2610(model, bott_vec, device):
    """
    Forward pass the decoder from a bottleneck vector → 2610-dim spectrum.
    Works for PIAE (uses model.decoder + pca_invert).
    For CVAE mu is used directly as z.
    """
    with torch.no_grad():
        b = torch.tensor(bott_vec, dtype=torch.float32).unsqueeze(0).to(device)
        z = model.decoder(b)
        X = model.pca_invert(z)
    return X.cpu().numpy()[0]   # (2610,)


def traverse_one_param(model, device, param_idx, nominal_bott, n_trav=100):
    """Sweep param_idx from -2→2 (normalised bottleneck), hold others at nominal."""
    sweep_vals = np.linspace(-2.0, 2.0, n_trav)
    S11  = np.zeros((n_trav, NF), dtype=complex)
    S21  = np.zeros((n_trav, NF), dtype=complex)
    S22  = np.zeros((n_trav, NF), dtype=complex)
    for i, sv in enumerate(sweep_vals):
        bott            = nominal_bott.copy()
        bott[param_idx] = sv
        X = _bott_to_X2610(model, bott, device)
        S11[i] = X[0:NF]          + 1j * X[NF:2*NF]
        S21[i] = X[2*NF:3*NF]     + 1j * X[3*NF:4*NF]
        S22[i] = X[4*NF:5*NF]     + 1j * X[5*NF:]
    return sweep_vals, S11, S21, S22


def plot_smith_traversal(model, model_type, loader, device, targets, plot_dir):
    """3A-i: single-frequency centre traversal + 3A-ii: full-spectrum loci."""
    os.makedirs(plot_dir, exist_ok=True)
    model.eval()

    # Compute nominal bottleneck = test-set median
    bott_list = []
    with torch.no_grad():
        for x_b, _ in loader:
            x_b = x_b.to(device)
            if model_type == "piae":
                _, yh, _, _ = model(x_b)
            else:
                yh, _ = model.encoder(x_b)
            bott_list.append(yh.cpu().numpy())
    bott_all     = np.concatenate(bott_list)
    nominal_bott = np.median(bott_all, axis=0).astype(np.float32)

    FREQ_IDX = NF // 2
    N_TRAV   = 100
    n_params = len(targets)
    traversals   = {}
    arc_lengths  = {}

    # ── 3A-i: centre-frequency traversal ─────────────────────────────────────
    n_cols = min(n_params, 3)
    n_rows = math.ceil(n_params / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5.5*n_rows))
    axes_flat  = np.array(axes).flatten()

    X_nom      = _bott_to_X2610(model, nominal_bott, device)
    nom_gamma  = complex(X_nom[FREQ_IDX], X_nom[NF + FREQ_IDX])

    print("[interp] Smith Chart traversal …")
    for p_idx, pname in enumerate(targets):
        print(f"  Sweeping {pname} …", end=" ", flush=True)
        sweep_vals, S11, S21, S22 = traverse_one_param(
            model, device, p_idx, nominal_bott, N_TRAV)
        traversals[pname] = (sweep_vals, S11, S21, S22)

        gammas_cf = S11[:, FREQ_IDX]
        arc = sum(poincare_distance(gammas_cf[i], gammas_cf[i+1])
                  for i in range(len(gammas_cf)-1))
        arc_lengths[pname] = float(arc)

        ax = axes_flat[p_idx]
        draw_smith_background(ax, title=f"{pname}  Dgeo={arc:.4f}")

        cmap_colors = cm.plasma(np.linspace(0.1, 0.9, N_TRAV - 1))
        for i in range(N_TRAV - 1):
            ax.plot([gammas_cf[i].real, gammas_cf[i+1].real],
                    [gammas_cf[i].imag, gammas_cf[i+1].imag],
                    color=cmap_colors[i], lw=2.2, solid_capstyle='round')

        ax.scatter(gammas_cf[0].real,  gammas_cf[0].imag,  c='steelblue', s=70, zorder=6, label='min')
        ax.scatter(gammas_cf[-1].real, gammas_cf[-1].imag, c='crimson',   s=70, zorder=6, label='max')
        ax.scatter(nom_gamma.real,     nom_gamma.imag,     c='limegreen', s=80, marker='*', zorder=7, label='nominal')

        violating = np.sum(np.abs(gammas_cf) > 1.0)
        if violating:
            ax.text(0.02, 0.02, f"⚠ {violating}/{N_TRAV} outside |Γ|=1",
                    transform=ax.transAxes, fontsize=7, color='red')
        ax.legend(fontsize=7, loc='lower right')
        print(f"arc={arc:.4f}")

    # Hide unused panels
    for ax in axes_flat[n_params:]:
        ax.set_visible(False)

    sm = cm.ScalarMappable(cmap=cm.plasma, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes_flat[n_params-1], label='Sweep (min → max)')

    plt.suptitle(f"Smith Chart — Latent Manifold Traversal (S11 @ centre freq.)\n"
                 f"({model_type.upper()}, 2610-dim)", fontsize=12)
    out = os.path.join(plot_dir, f"{model_type}_smith_traversal_center_freq.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] Smith traversal (centre freq) → '{out}'")

    # ── 3A-ii: full-spectrum loci ─────────────────────────────────────────────
    LOCUS_STEPS  = [0, N_TRAV//4, N_TRAV//2, 3*N_TRAV//4, N_TRAV-1]
    LOCUS_LABELS = ['min', '25%', '50%', '75%', 'max']
    LOCUS_COLORS = ['steelblue', 'teal', 'gold', 'darkorange', 'crimson']

    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5.5*n_rows))
    axes2_flat  = np.array(axes2).flatten()

    for p_idx, pname in enumerate(targets):
        _, S11, _, _ = traversals[pname]
        ax = axes2_flat[p_idx]
        draw_smith_background(ax, title=f"{pname} — full spectrum locus")
        for step_i, label, color in zip(LOCUS_STEPS, LOCUS_LABELS, LOCUS_COLORS):
            gc = S11[step_i]
            ax.plot(gc.real, gc.imag, color=color, lw=1.5, label=label, alpha=0.9)
            ax.scatter(gc[FREQ_IDX].real, gc[FREQ_IDX].imag, color=color, s=40, zorder=5)
        ax.legend(fontsize=7, title=pname, title_fontsize=8)

    for ax in axes2_flat[n_params:]:
        ax.set_visible(False)

    plt.suptitle(f"Smith Chart — Full S11 Spectrum Loci ({model_type.upper()})", fontsize=12)
    out2 = os.path.join(plot_dir, f"{model_type}_smith_full_locus.png")
    plt.tight_layout(); plt.savefig(out2, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] Smith full locus → '{out2}'")

    # ── 3A-iii: arc-length summary ────────────────────────────────────────────
    print("\n── Geodesic Arc Length Summary (centre freq.) ───────────────────")
    print(f"  {'Param':>6}  {'Dgeo':>10}  Bar")
    max_arc = max(arc_lengths.values()) + 1e-9
    for pname, arc in sorted(arc_lengths.items(), key=lambda kv: -kv[1]):
        bar = '█' * int(20 * arc / max_arc)
        print(f"  {pname:>6}  {arc:>10.4f}  {bar}")
    print("  Larger arc → stronger S11 sensitivity to this parameter.")

    return arc_lengths


# ─────────────────────────────────────────────────────────────────────────────
# B  Averaged Integrated Gradients (all three sub-plots)
# ─────────────────────────────────────────────────────────────────────────────
def _raw_to_encoder_output(model, model_type, x_raw_batch):
    """Differentiable: X_raw (2610) → bottleneck output. Runs on CPU."""
    model_cpu = model.cpu()
    feat_mean = model_cpu.feature_mean
    feat_std  = model_cpu.feature_std
    V_pca     = model_cpu.V_pca
    mu_pca    = model_cpu.mu_pca
    sm        = model_cpu.score_mean
    ss        = model_cpu.score_std

    x_std    = (x_raw_batch - feat_mean) / feat_std
    # PCA forward: x_std → PCA scores
    # mu_pca here is the PCA component mean in standardised space
    x_pca_un = (x_std - mu_pca) @ V_pca          # (B, n_pca)
    x_pca_n  = (x_pca_un - sm) / ss

    if model_type == "piae":
        return model_cpu.encoder(x_pca_n)
    else:
        mu, _ = model_cpu.encoder(x_pca_n)
        return mu


def compute_ig_averaged(model, model_type, X_raw_test, targets, device,
                        n_steps=50, n_samples=200, seed=0):
    """
    Compute Integrated Gradients averaged over n_samples test samples.
    Returns freq_attrs: (n_params, NF)  — averaged |IG| per frequency (collapsed over channels).
    """
    model.eval()
    dim_phys = len(targets)
    rng      = np.random.default_rng(seed)
    samp_idx = rng.choice(len(X_raw_test), min(n_samples, len(X_raw_test)), replace=False)
    feat_mean_np = model.feature_mean.cpu().numpy()
    baseline_np  = feat_mean_np.astype(np.float32)   # zero in standardised space

    freq_attrs = np.zeros((dim_phys, NF), dtype=np.float64)
    # Also store full 2610-dim IG for channel breakdown
    chan_attrs  = np.zeros((dim_phys, DIM_RAW), dtype=np.float64)

    print(f"[interp] IG over {len(samp_idx)} samples × {dim_phys} params × {n_steps} steps …")

    # Move model to CPU once for IG
    model.cpu()

    for s_i, samp_i in enumerate(samp_idx):
        x = X_raw_test[samp_i].astype(np.float32)
        for p_idx in range(dim_phys):
            alphas = np.linspace(0.0, 1.0, n_steps)
            grads  = []
            for alpha in alphas:
                interp_np = baseline_np + alpha * (x - baseline_np)
                interp_t  = torch.tensor(interp_np, dtype=torch.float32).unsqueeze(0)
                interp_t.requires_grad_(True)
                out = _raw_to_encoder_output(model, model_type, interp_t)
                out[0, p_idx].backward()
                grads.append(interp_t.grad.detach().squeeze(0).numpy())

            avg_grad = np.stack(grads).mean(axis=0)
            ig_raw   = (x - baseline_np) * avg_grad   # (2610,)
            chan_attrs[p_idx] += np.abs(ig_raw)

            # Collapse 6 channels → per-frequency mean
            ig_s11_re = np.abs(ig_raw[0:NF])
            ig_s11_im = np.abs(ig_raw[NF:2*NF])
            ig_s21_re = np.abs(ig_raw[2*NF:3*NF])
            ig_s21_im = np.abs(ig_raw[3*NF:4*NF])
            ig_s22_re = np.abs(ig_raw[4*NF:5*NF])
            ig_s22_im = np.abs(ig_raw[5*NF:])
            freq_attrs[p_idx] += (ig_s11_re + ig_s11_im + ig_s21_re +
                                  ig_s21_im + ig_s22_re + ig_s22_im) / 6.0

        if (s_i + 1) % 50 == 0:
            print(f"    {s_i+1}/{len(samp_idx)} samples done …")

    freq_attrs /= len(samp_idx)
    chan_attrs  /= len(samp_idx)

    # Move model back to original device
    model.to(device)
    return freq_attrs, chan_attrs


def plot_ig_heatmap(freq_attrs, targets, model_type, plot_dir):
    """3B-i: IG heatmap (params × frequency)."""
    os.makedirs(plot_dir, exist_ok=True)
    freq_attrs_norm = freq_attrs / (freq_attrs.max(axis=1, keepdims=True) + 1e-30)

    fig, ax = plt.subplots(figsize=(14, max(4, len(targets)*0.8)))
    im = ax.imshow(freq_attrs_norm, aspect='auto', cmap='inferno',
                   origin='upper', vmin=0, vmax=1)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=11)
    ax.set_xlabel("Frequency index  (0 = 0.04 GHz → 434 = 43.5 GHz)", fontsize=10)
    ax.set_title(f"Integrated Gradients — Frequency Attribution ({model_type.upper()})\n"
                 "(row-normalised; brighter = more informative for that parameter)", fontsize=11)
    plt.colorbar(im, ax=ax, label='Normalised |IG|')
    out = os.path.join(plot_dir, f"{model_type}_ig_heatmap.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] IG heatmap → '{out}'")


def plot_ig_per_param(freq_attrs, targets, model_type, plot_dir):
    """3B-ii: per-parameter IG curves with peak annotation."""
    os.makedirs(plot_dir, exist_ok=True)
    PARAM_COLORS = ['royalblue','darkorange','seagreen','crimson','purple','teal']
    fq = np.linspace(0.04, 43.5, NF)

    fig, axes = plt.subplots(len(targets), 1, figsize=(14, 2.8*len(targets)), sharex=True)
    for p_idx, (pname, color) in enumerate(zip(targets, PARAM_COLORS)):
        ax  = axes[p_idx]
        att = freq_attrs[p_idx]
        ax.fill_between(fq, att, alpha=0.35, color=color)
        ax.plot(fq, att, color=color, lw=1.5, label=pname)
        peak_f = fq[np.argmax(att)]
        ax.axvline(peak_f, color=color, lw=1.0, linestyle='--', alpha=0.7)
        ax.text(peak_f + 0.5, att.max() * 0.85, f"peak @ {peak_f:.1f} GHz",
                fontsize=8, color=color)
        ax.set_ylabel(f"{pname}\n|IG|", fontsize=9)
        ax.set_xlim(fq[0], fq[-1])
        ax.grid(axis='x', alpha=0.3)

    axes[-1].set_xlabel("Frequency (GHz)", fontsize=10)
    plt.suptitle(f"IG — Per-parameter Frequency Attribution ({model_type.upper()})\n"
                 f"averaged over test samples; end-to-end w.r.t. raw S-params", fontsize=12)
    out = os.path.join(plot_dir, f"{model_type}_ig_per_param.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] IG per-param curves → '{out}'")

    # Print summary
    print("\n── IG Peak Frequency per Parameter ─────────────────────────────")
    for p_idx, pname in enumerate(targets):
        peak_f = fq[np.argmax(freq_attrs[p_idx])]
        band   = 'low' if peak_f < 14.5 else ('mid' if peak_f < 29.0 else 'high')
        print(f"  {pname:>6s}: {peak_f:.1f} GHz  ({band} band)")


def plot_ig_channel_breakdown(chan_attrs, targets, model_type, plot_dir):
    """3B-iii: per-channel (S11/S21/S22) and low/mid/high band breakdown."""
    os.makedirs(plot_dir, exist_ok=True)
    PARAM_COLORS = ['royalblue','darkorange','seagreen','crimson','purple','teal']
    fq = np.linspace(0.04, 43.5, NF)
    low_idx, mid_idx = NF // 3, 2 * NF // 3

    fig, axes = plt.subplots(len(targets), 1, figsize=(14, 2.8*len(targets)), sharex=True)
    for p_idx, (pname, color) in enumerate(zip(targets, PARAM_COLORS)):
        ax  = axes[p_idx]
        # Collapse all 6 raw channels to per-freq for band annotation
        att = (chan_attrs[p_idx][0:NF]       + chan_attrs[p_idx][NF:2*NF]   +
               chan_attrs[p_idx][2*NF:3*NF]  + chan_attrs[p_idx][3*NF:4*NF] +
               chan_attrs[p_idx][4*NF:5*NF]  + chan_attrs[p_idx][5*NF:]) / 6.0

        ax.axvspan(fq[0],       fq[low_idx],  alpha=0.08, color='blue',  label='low band')
        ax.axvspan(fq[low_idx], fq[mid_idx],  alpha=0.08, color='green', label='mid band')
        ax.axvspan(fq[mid_idx], fq[-1],       alpha=0.08, color='red',   label='high band')
        ax.plot(fq, att, color=color, lw=1.5, label=pname)

        low_m  = att[:low_idx].sum()
        mid_m  = att[low_idx:mid_idx].sum()
        high_m = att[mid_idx:].sum()
        total  = low_m + mid_m + high_m + 1e-30
        ax.set_ylabel(pname, fontsize=9)
        ax.text(0.99, 0.82,
                f"low {100*low_m/total:.1f}%   mid {100*mid_m/total:.1f}%   high {100*high_m/total:.1f}%",
                transform=ax.transAxes, ha='right', fontsize=8, color='dimgray')
        if p_idx == 0:
            ax.legend(fontsize=7, ncol=4, loc='upper left')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(fq[0], fq[-1])

    axes[-1].set_xlabel("Frequency (GHz)", fontsize=10)
    plt.suptitle(f"IG — Spectral Band Energy Breakdown ({model_type.upper()})\n"
                 "low / mid / high: thirds of frequency range", fontsize=12)
    out = os.path.join(plot_dir, f"{model_type}_ig_band_breakdown.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] IG band breakdown → '{out}'")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level call — add this block inside interpretability.run() after the
# existing compute_crosstalk_matrix() and plot_integrated_gradients() calls
# ─────────────────────────────────────────────────────────────────────────────
def run_part1(model, model_type, X_raw, loader, device, targets, plot_dir):
    """
    Call this from interpretability.run() to execute Part 1 additions.
    X_raw : (N, 2610) float32  — full raw dataset (all splits)
    loader: DataLoader over PCA-encoded test set
    """
    # Smith Chart
    arc_lengths = plot_smith_traversal(model, model_type, loader, device, targets, plot_dir)

    # Averaged IG
    test_X_raw = X_raw   # pass only test-set rows if you prefer
    freq_attrs, chan_attrs = compute_ig_averaged(
        model, model_type, test_X_raw, targets, device,
        n_steps=50, n_samples=200)

    plot_ig_heatmap(freq_attrs, targets, model_type, plot_dir)
    plot_ig_per_param(freq_attrs, targets, model_type, plot_dir)
    plot_ig_channel_breakdown(chan_attrs, targets, model_type, plot_dir)

    return arc_lengths, freq_attrs, chan_attrs

warnings.filterwarnings("ignore")

NF = 435

# ── Section 1: Percentile-split Encoder Crosstalk ─────────────────────────────
def plot_encoder_crosstalk(y_bott_np, Y_test_phys, targets, model_type, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    D = len(targets)
    crosstalk_raw  = np.zeros((D, D))
    crosstalk_norm = np.zeros((D, D))

    for ks in range(D):
        lo = np.percentile(Y_test_phys[:, ks], 20)
        hi = np.percentile(Y_test_phys[:, ks], 80)
        lo_mask = Y_test_phys[:, ks] <= lo
        hi_mask = Y_test_phys[:, ks] >= hi
        delta = y_bott_np[hi_mask].mean(axis=0) - y_bott_np[lo_mask].mean(axis=0)
        crosstalk_raw[:, ks] = delta
        self_change = abs(delta[ks]) + 1e-9
        crosstalk_norm[:, ks] = delta / self_change

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(crosstalk_raw, cmap='RdBu_r', aspect='auto',
                         vmin=-np.abs(crosstalk_raw).max(), vmax=np.abs(crosstalk_raw).max())
    axes[0].set_xticks(range(D)); axes[0].set_yticks(range(D))
    axes[0].set_xticklabels([f'↑{n}' for n in targets], rotation=30, ha='right', fontsize=9)
    axes[0].set_yticklabels([f'Δ{n}' for n in targets], fontsize=10)
    plt.colorbar(im0, ax=axes[0], label='Mean Δ bottleneck')
    axes[0].set_title('Encoder Crosstalk — raw Δ\nDiagonal=self  Off-diag=leakage', fontsize=9)
    axes[0].set_xlabel('Source parameter'); axes[0].set_ylabel('Effect on output')
    for i in range(D):
        for j in range(D):
            axes[0].text(j, i, f'{crosstalk_raw[i,j]:+.3f}', ha='center', va='center',
                         fontsize=8, fontweight='bold' if i==j else 'normal')

    im1 = axes[1].imshow(np.abs(crosstalk_norm), cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(range(D)); axes[1].set_yticks(range(D))
    axes[1].set_xticklabels([f'↑{n}' for n in targets], rotation=30, ha='right', fontsize=9)
    axes[1].set_yticklabels([f'Δ{n}' for n in targets], fontsize=10)
    plt.colorbar(im1, ax=axes[1], label='|Δ effect| / |Δ self|')
    axes[1].set_title('Encoder Crosstalk — normalised\nOff-diagonal = fraction of self leaked', fontsize=9)
    for i in range(D):
        for j in range(D):
            v = abs(crosstalk_norm[i, j])
            axes[1].text(j, i, f'{v:.2f}', ha='center', va='center',
                         fontsize=9, color='white' if v > 0.6 else 'black',
                         fontweight='bold' if i==j else 'normal')

    plt.suptitle('Encoder Crosstalk Matrix\nIdeal: diagonal=1, off-diagonal≈0', fontsize=10)
    out = os.path.join(plot_dir, f"{model_type}_encoder_crosstalk.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] Crosstalk matrix → '{out}'")

    print("\n  Worst crosstalk pairs (|norm. leakage| > 0.15):")
    for ke in range(D):
        for ks in range(D):
            if ke != ks and abs(crosstalk_norm[ke, ks]) > 0.15:
                print(f"    ↑{targets[ks]:>6s} leaks {abs(crosstalk_norm[ke,ks])*100:.1f}% into {targets[ke]}")
    return crosstalk_norm


# ── Section 2: Fidelity by Parameter Regime ────────────────────────────────────
def plot_fidelity_by_regime(X_dec_np, X_true_np, Y_pred_phys, Y_test_phys,
                             targets, model_type, plot_dir, n_bins=8):
    os.makedirs(plot_dir, exist_ok=True)
    D = len(targets)
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728','#17becf']
    freqs_GHz = np.linspace(0.04, 43.5, NF)

    # Per-sample metrics
    recon_mse = np.mean((X_dec_np - X_true_np)**2, axis=1)
    pct_err   = np.abs(Y_pred_phys - Y_test_phys) / (np.abs(Y_test_phys) + 1e-8) * 100.

    # Passivity: check S11+S21 and S22+S21
    R11d = X_dec_np[:, :NF];        I11d = X_dec_np[:, NF:2*NF]
    R21d = X_dec_np[:, 2*NF:3*NF]; I21d = X_dec_np[:, 3*NF:4*NF]
    R22d = X_dec_np[:, 4*NF:5*NF]; I22d = X_dec_np[:, 5*NF:]
    passiv_viol = (
        np.mean(np.maximum(0, R11d**2 + I11d**2 + R21d**2 + I21d**2 - 1), axis=1) +
        np.mean(np.maximum(0, R22d**2 + I22d**2 + R21d**2 + I21d**2 - 1), axis=1)
    ) / 2.0

    fig, axes = plt.subplots(3, D, figsize=(5*D, 11), sharey='row')
    for k, name in enumerate(targets):
        param_vals = Y_test_phys[:, k]
        qs = np.quantile(param_vals, np.linspace(0, 1, n_bins + 1))
        bc, rm, mm, pm = [], [], [], []
        for b in range(n_bins):
            mask = (param_vals >= qs[b]) & (param_vals < qs[b+1])
            if mask.sum() < 5: continue
            bc.append(0.5 * (qs[b] + qs[b+1]))
            rm.append(recon_mse[mask].mean())
            mm.append(pct_err[mask, k].mean())
            pm.append(passiv_viol[mask].mean())
        c = colors[k % len(colors)]
        axes[0, k].plot(bc, rm, 'o-', color=c, lw=1.5, ms=5)
        axes[0, k].set_title(name, fontsize=10, fontweight='bold')
        axes[0, k].grid(True, alpha=0.25)
        if k == 0: axes[0, k].set_ylabel('Mean Recon. MSE', fontsize=9)
        axes[1, k].plot(bc, mm, 's-', color=c, lw=1.5, ms=5)
        axes[1, k].grid(True, alpha=0.25)
        if k == 0: axes[1, k].set_ylabel('Mean MAPE (%)', fontsize=9)
        axes[2, k].plot(bc, pm, '^-', color=c, lw=1.5, ms=5)
        axes[2, k].set_xlabel(f'{name} (true)', fontsize=9)
        axes[2, k].grid(True, alpha=0.25)
        if k == 0: axes[2, k].set_ylabel('Mean Passivity Viol.', fontsize=9)

    plt.suptitle('Reconstruction & Extraction Quality Across Parameter Range\n'
                 'Row 1: decoder MSE  |  Row 2: MAPE  |  Row 3: passivity violation',
                 fontsize=10, fontweight='bold')
    out = os.path.join(plot_dir, f"{model_type}_fidelity_by_regime.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] Fidelity by regime → '{out}'")


# ── Section 3: Residual Spectrum Analysis ──────────────────────────────────────
def plot_residual_analysis(X_dec_np, X_true_np, Y_pred_phys, Y_test_phys,
                            targets, model_type, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    D = len(targets)
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728','#17becf']
    freqs_GHz = np.linspace(0.04, 43.5, NF)
    pct_err   = np.abs(Y_pred_phys - Y_test_phys) / (np.abs(Y_test_phys) + 1e-8) * 100.
    delta_X   = X_dec_np - X_true_np

    # 3A: Mean residual bias per channel (S11_Re, S11_Im, S21_Re, S21_Im, S22_Re, S22_Im)
    panels = [
        ('Re(S11)', slice(None, NF)),        ('Im(S11)', slice(NF, 2*NF)),
        ('Re(S21)', slice(2*NF, 3*NF)),      ('Im(S21)', slice(3*NF, 4*NF)),
        ('Re(S22)', slice(4*NF, 5*NF)),      ('Im(S22)', slice(5*NF, None)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (title, sl) in zip(axes.flatten(), panels):
        mean_ch = delta_X[:, sl].mean(axis=0)
        std_ch  = delta_X[:, sl].std(axis=0)
        ax.plot(freqs_GHz, mean_ch, lw=1.5, color='steelblue', label='Mean bias')
        ax.fill_between(freqs_GHz, mean_ch - std_ch, mean_ch + std_ch,
                        alpha=0.25, color='steelblue', label='±1 std')
        ax.axhline(0, color='red', lw=0.8, linestyle='--', alpha=0.6)
        ax.set_title(f'ΔX — {title}', fontsize=10)
        ax.set_xlabel('Frequency (GHz)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        peak_f = freqs_GHz[np.argmax(np.abs(mean_ch))]
        peak_v = mean_ch[np.argmax(np.abs(mean_ch))]
        ax.annotate(f'{peak_v:+.4f}\n@ {peak_f:.1f} GHz', xy=(peak_f, peak_v),
                    fontsize=7, color='red', xytext=(peak_f + 2, peak_v * 0.6),
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    plt.suptitle('Section 3A — Frequency-Resolved Residual Bias\n'
                 'Blue: mean (Decoded − True) ± 1σ  |  Flat at 0 = unbiased',
                 fontsize=10, fontweight='bold')
    out3a = os.path.join(plot_dir, f"{model_type}_residual_bias.png")
    plt.tight_layout(); plt.savefig(out3a, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] Residual bias → '{out3a}'")

    # 3B: Residual–prediction error correlation (Spearman)
    N_FREQ_SUB   = 87
    freq_sub_idx = np.arange(0, NF, NF // N_FREQ_SUB)
    freqs_sub    = freqs_GHz[freq_sub_idx]
    resid_err_corr = np.zeros((D, N_FREQ_SUB))

    print("[interp] Computing residual-error Spearman correlations …")
    for k in range(D):
        err_k = pct_err[:, k]
        for fi, f_idx in enumerate(freq_sub_idx):
            abs_res_f = (np.abs(delta_X[:, f_idx]) + np.abs(delta_X[:, NF+f_idx]) +
                         np.abs(delta_X[:, 2*NF+f_idx]) + np.abs(delta_X[:, 3*NF+f_idx]) +
                         np.abs(delta_X[:, 4*NF+f_idx]) + np.abs(delta_X[:, 5*NF+f_idx])) / 6.0
            r, _ = spearmanr(err_k, abs_res_f)
            resid_err_corr[k, fi] = 0.0 if np.isnan(r) else r

    fig, axes = plt.subplots(D, 1, figsize=(13, 3*D), sharex=True)
    for k, (ax, name) in enumerate(zip(axes, targets)):
        c = colors[k % len(colors)]
        ax.fill_between(freqs_sub, 0, resid_err_corr[k], alpha=0.55, color=c)
        ax.plot(freqs_sub, resid_err_corr[k], lw=1.2, color=c)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel('Spearman r', fontsize=9)
        ax.set_title(f'{name}: misprediction ↔ |ΔSpectrum(f)|', fontsize=9)
        ax.set_ylim(-0.3, 0.8); ax.grid(True, alpha=0.2)
        pf = freqs_sub[np.argmax(resid_err_corr[k])]
        ax.axvline(pf, color=c, lw=0.8, linestyle=':', alpha=0.7)
        ax.text(pf + 0.3, 0.55, f'{pf:.1f} GHz', fontsize=7, color=c)
    axes[-1].set_xlabel('Frequency (GHz)', fontsize=10)
    plt.suptitle('Section 3B — Where Does Misprediction Hurt the Spectrum?\n'
                 'High r at freq f = mispredicting param corrupts S at f',
                 fontsize=9, fontweight='bold')
    out3b = os.path.join(plot_dir, f"{model_type}_residual_error_correlation.png")
    plt.tight_layout(); plt.savefig(out3b, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] Residual-error correlation → '{out3b}'")

    # 3C: PCA of residuals
    pca_resid = sklearn_PCA(n_components=min(10, len(delta_X)-1), svd_solver='full')
    pca_resid.fit(delta_X)
    evr_resid = pca_resid.explained_variance_ratio_
    cum_resid = np.cumsum(evr_resid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, len(evr_resid)+1), evr_resid*100, color='steelblue', alpha=0.75)
    axes[0].set_xlabel('Residual PC', fontsize=10)
    axes[0].set_ylabel('Explained variance (%)', fontsize=10)
    axes[0].set_title('PCA of Residuals ΔX\nFlat = unstructured (random)', fontsize=9)
    axes[0].grid(True, alpha=0.25)

    resid_scores_pc1 = pca_resid.transform(delta_X)[:, 0]
    for k, name in enumerate(targets):
        r_val, _ = pearsonr(resid_scores_pc1, pct_err[:, k])
        axes[1].bar(k, abs(r_val), color=colors[k % len(colors)], alpha=0.8,
                    label=f'{name} |r|={abs(r_val):.2f}')
    axes[1].set_xticks(range(D)); axes[1].set_xticklabels(targets, fontsize=10)
    axes[1].set_ylabel('|Pearson r| with Residual PC1', fontsize=9)
    axes[1].set_title('Which param error drives dominant residual structure?', fontsize=9)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.25); axes[1].set_ylim(0, 1)

    plt.suptitle(f'Section 3C — Residual Space Structure\n'
                 f'PC1 explains {evr_resid[0]*100:.1f}% of residual variance',
                 fontsize=10, fontweight='bold')
    out3c = os.path.join(plot_dir, f"{model_type}_residual_pca.png")
    plt.tight_layout(); plt.savefig(out3c, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] Residual PCA → '{out3c}'")
    print(f"  Residual PC1: {evr_resid[0]*100:.1f}%   first 3: {cum_resid[2]*100:.1f}%")
    return evr_resid


# ── Section 4: Input Noise Robustness ──────────────────────────────────────────
def plot_noise_robustness(model, model_type, X_true_np, Y_test_phys,
                           y_scalers, log_idx, targets, device, plot_dir,
                           noise_levels=None, n_trials=5):
    os.makedirs(plot_dir, exist_ok=True)
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    D = len(targets)
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728','#17becf']

    X_true_t = torch.tensor(X_true_np, dtype=torch.float32).to(device)
    rms_all  = float(X_true_t.pow(2).mean().sqrt())
    print(f"[interp] Noise robustness — signal RMS: {rms_all:.4f}")

    # Helper: physical spectrum → PCA → encoder output
    def _encode_from_phys(X_phys_t):
        x_std    = (X_phys_t - model.feature_mean) / model.feature_std
        x_pca_un = (x_std - model.mu_pca) @ model.V_pca
        x_pca_n  = (x_pca_un - model.score_mean) / model.score_std
        x_pca_n  = torch.clamp(x_pca_n, -5.0, 5.0)
        if model_type == "piae":
            _, yh, _, _ = model(x_pca_n)
        else:
            yh, _ = model.encoder(x_pca_n)
        return yh

    # Need _inverse locally
    def _inv(Y_sc):
        import numpy as np
        Y_out = np.zeros_like(Y_sc, dtype=np.float64)
        for i, t in enumerate(targets):
            Y_out[:, i] = y_scalers[t].inverse_transform(Y_sc[:, i:i+1]).ravel()
        for i in log_idx:
            Y_out[:, i] = 10.0 ** Y_out[:, i]
        return Y_out

    mape_vs_noise  = {n: [] for n in targets}
    mape_std_noise = {n: [] for n in targets}

    model.eval()
    with torch.no_grad():
        for sigma_rel in noise_levels:
            sigma_abs = sigma_rel * rms_all
            trial_mapes = {n: [] for n in targets}
            for _ in range(n_trials):
                noise   = sigma_abs * torch.randn_like(X_true_t)
                X_noisy = X_true_t + noise
                yh      = _encode_from_phys(X_noisy)
                y_pred  = _inv(yh.cpu().numpy())
                for k, name in enumerate(targets):
                    mape_k = np.mean(np.abs(y_pred[:, k] - Y_test_phys[:, k]) /
                                     (np.abs(Y_test_phys[:, k]) + 1e-8)) * 100.
                    trial_mapes[name].append(mape_k)
            for name in targets:
                mape_vs_noise[name].append(np.mean(trial_mapes[name]))
                mape_std_noise[name].append(np.std(trial_mapes[name]))
            mean_all = np.mean([np.mean(trial_mapes[n]) for n in targets])
            print(f"  σ_rel={sigma_rel:.3f}  mean_MAPE={mean_all:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for k, name in enumerate(targets):
        means = np.array(mape_vs_noise[name])
        stds  = np.array(mape_std_noise[name])
        c = colors[k % len(colors)]
        ax.plot(noise_levels, means, 'o-', color=c, lw=1.8, ms=5, label=name)
        ax.fill_between(noise_levels, means - stds, means + stds, alpha=0.15, color=c)
    ax.set_xlabel('Relative noise σ/RMS', fontsize=10)
    ax.set_ylabel('MAPE (%)', fontsize=10)
    ax.set_title('Extraction MAPE vs Input Noise\n(noise in physical domain, re-encoded via PCA)', fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    ax.set_xscale('symlog', linthresh=0.001)

    ax2 = axes[1]
    mean_all_noise = np.array([np.mean([mape_vs_noise[n][i] for n in targets])
                                for i in range(len(noise_levels))])
    baseline_mape  = mean_all_noise[0]
    ax2.plot(noise_levels, mean_all_noise, 'ko-', lw=2, ms=6, label='Mean MAPE')
    ax2.axhline(baseline_mape * 1.5, color='orange', lw=1, linestyle='--',
                label=f'+50% = {baseline_mape*1.5:.1f}%')
    ax2.axhline(baseline_mape * 2.0, color='red',    lw=1, linestyle='-.',
                label=f'×2 = {baseline_mape*2.0:.1f}%')

    doubled_idx = np.searchsorted(mean_all_noise, baseline_mape * 2.0)
    if doubled_idx < len(noise_levels):
        snr_floor = noise_levels[doubled_idx]
        ax2.axvline(snr_floor, color='red', lw=1.2, linestyle=':',
                    label=f'SNR floor ≈ {snr_floor:.3f}')
        print(f"  SNR floor (MAPE doubles): σ_rel ≈ {snr_floor:.3f}")

    ax2.set_xlabel('Relative noise σ/RMS', fontsize=10)
    ax2.set_ylabel('Mean MAPE (%)', fontsize=10)
    ax2.set_title('Operational SNR Floor', fontsize=9)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)
    ax2.set_xscale('symlog', linthresh=0.001)

    plt.suptitle(f'Section 4 — Input Noise Robustness ({model_type.upper()})\n'
                 f'Baseline (zero noise) MAPE = {baseline_mape:.2f}%',
                 fontsize=10, fontweight='bold')
    out = os.path.join(plot_dir, f"{model_type}_noise_robustness.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[interp] Noise robustness → '{out}'")
    return mape_vs_noise


# ── Top-level call — add inside interpretability.run() ─────────────────────────
def run_part2(model, model_type, X_dec_np, X_true_np,
              y_bott_np, Y_pred_phys, Y_test_phys,
              y_scalers, log_idx, targets, device, plot_dir):
    """
    Call from interpretability.run() after gathering inference arrays.

    y_bott_np   : (N_test, dim_phys) — raw bottleneck/mu outputs (scaled)
    X_dec_np    : (N_test, 2610)     — decoded spectra
    X_true_np   : (N_test, 2610)     — true physical spectra (PCA-inverted from input)
    Y_pred_phys : (N_test, D)        — predicted physical params (un-scaled)
    Y_test_phys : (N_test, D)        — true physical params (un-scaled)
    """
    plot_encoder_crosstalk(y_bott_np, Y_test_phys, targets, model_type, plot_dir)
    plot_fidelity_by_regime(X_dec_np, X_true_np, Y_pred_phys, Y_test_phys,
                             targets, model_type, plot_dir)
    evr = plot_residual_analysis(X_dec_np, X_true_np, Y_pred_phys, Y_test_phys,
                                  targets, model_type, plot_dir)
    plot_noise_robustness(model, model_type, X_true_np, Y_test_phys,
                           y_scalers, log_idx, targets, device, plot_dir)
    return evr

# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (merged — calls all analysis sections)
# ─────────────────────────────────────────────────────────────────────────────
def run(model_path: str, model_type: str, data_dir: str = "data/raw",
        pca_dir: str = "data/pca_artifacts", proc_dir: str = "data/processed",
        splits_dir: str = "data/splits", plot_dir: str = "outputs/plots/interp"):

    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt      = torch.load(model_path, map_location=device, weights_only=False)
    dim_pca   = ckpt["dim_pca"]
    targets   = ckpt["target_cols"]
    y_scalers = ckpt["y_scalers"]
    log_idx   = [targets.index(t) for t in ckpt.get("log_targets", targets)]

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

    if model_type == "piae":
        model = PIAE(*buffers, dim_pca=dim_pca).to(device)
    else:
        model = PhysicsSupervisedCVAE(*buffers, dim_pca=dim_pca).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"\n[interp] Loaded {model_type.upper()} from {model_path}")

    # ── PCA-space data & DataLoader ───────────────────────────────────────────
    df_pca   = pd.read_csv(os.path.join(proc_dir, "ssec_pca_final_v2.csv"))
    pc_cols  = [f"PC{i+1}" for i in range(dim_pca)]
    X_pca    = df_pca[pc_cols].values.astype(np.float32)
    test_idx = np.load(os.path.join(splits_dir, "split_test_idx.npy"))
    Y_phys_all = df_pca[targets].values.astype(np.float32)

    class _DS(Dataset):
        def __init__(self, X): self.X = torch.tensor(X, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], 0

    loader = DataLoader(Subset(_DS(X_pca), test_idx.tolist()), batch_size=256, shuffle=False)

    # ── Raw 2610-dim data (for IG and noise robustness) ───────────────────────
    archive = np.load(os.path.join(data_dir, "data.npz"), allow_pickle=True)
    df_raw  = pd.DataFrame(archive["matrix"], columns=archive["headers"])
    def _fv(col): return float(col.split("_")[-1].replace("GHz", ""))
    feat_cols = (sorted([c for c in df_raw.columns if c.startswith("S11_Re")], key=_fv) +
                 sorted([c for c in df_raw.columns if c.startswith("S11_Im")], key=_fv) +
                 sorted([c for c in df_raw.columns if c.startswith("S21_Re")], key=_fv) +
                 sorted([c for c in df_raw.columns if c.startswith("S21_Im")], key=_fv) +
                 sorted([c for c in df_raw.columns if c.startswith("S22_Re")], key=_fv) +
                 sorted([c for c in df_raw.columns if c.startswith("S22_Im")], key=_fv))
    X_raw = df_raw[feat_cols].values.astype(np.float32)

    # ── Inference: gather arrays for Part 2 ───────────────────────────────────
    bott_list, Xd_list, Xt_list = [], [], []
    model.eval()
    with torch.no_grad():
        for x_b, _ in loader:
            x_b = x_b.to(device)
            if model_type == "piae":
                _, yh, Xd, Xt = model(x_b)
            else:
                yh, _ = model.encoder(x_b)
                _, _, _, Xd, Xt = model(x_b)
            bott_list.append(yh.cpu().numpy())
            Xd_list.append(Xd.cpu().numpy())
            Xt_list.append(Xt.cpu().numpy())

    y_bott_np = np.concatenate(bott_list)   # (N_test, dim_phys)
    X_dec_np  = np.concatenate(Xd_list)     # (N_test, 2610)
    X_true_np = np.concatenate(Xt_list)     # (N_test, 2610)

    # Inverse-transform to physical space
    def _inv(Y_sc):
        Y_out = np.zeros_like(Y_sc, dtype=np.float64)
        for i, t in enumerate(targets):
            Y_out[:, i] = y_scalers[t].inverse_transform(Y_sc[:, i:i+1]).ravel()
        for i in log_idx:
            Y_out[:, i] = 10.0 ** Y_out[:, i]
        return Y_out

    Y_proc_test = Y_phys_all[test_idx].copy().astype(np.float64)
    for i in log_idx:
        Y_proc_test[:, i] = np.log10(np.abs(Y_proc_test[:, i]) + 1e-30)
    Y_scaled_test = np.zeros_like(Y_proc_test, dtype=np.float32)
    for i, t in enumerate(targets):
        Y_scaled_test[:, i] = y_scalers[t].transform(Y_proc_test[:, i:i+1]).ravel()

    Y_test_phys = _inv(Y_scaled_test)
    Y_pred_phys = _inv(y_bott_np)

    # ── Section 0: Jacobian crosstalk heatmap (original) ─────────────────────
    compute_crosstalk_matrix(model, model_type, loader, device, targets, plot_dir)

    # ── Section 0b: Single-sample Integrated Gradients (original) ────────────
    plot_integrated_gradients(model, model_type, X_raw, None, device, targets, plot_dir)

    # ── Part 1: Smith Chart traversal + averaged IG ───────────────────────────
    run_part1(model, model_type, X_raw[test_idx], loader, device, targets, plot_dir)

    # ── Part 2: percentile crosstalk / fidelity / residuals / noise ──────────
    run_part2(model, model_type, X_dec_np, X_true_np,
              y_bott_np, Y_pred_phys, Y_test_phys,
              y_scalers, log_idx, targets, device, plot_dir)

    print(f"\n[interp] ✅  All interpretability analyses complete.")
