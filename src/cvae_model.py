# -*- coding: utf-8 -*-
"""
cvae_model.py
=============
Physics-Supervised Conditional VAE (CVAE v5) — updated for S11 + S21 + S22 (2610-dim).

Key features:
  - DIM_RAW = 2610
  - Retains Ldv in the latent space (DIM_PHYS = 6).
  - Implements the STOP-GRADIENT fix on z_Ld to correctly model the null space.
  - Per-dimension KL weights (Ldv gets higher weight).
  - Free bits on observable dimensions to prevent sigma collapse.
  - Checkpoints based on 5-param SMAPE (excluding Ldv).

All artefacts (model checkpoint, plots) are written to the paths
passed in via run().
"""

import math, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NF      = 435
DIM_RAW = 2610

TARGET_COLS = ['Rbv', 'Cbv', 'Rdv', 'Ldv', 'Cdv', 'Rav']
LOG_TARGETS = ['Rbv', 'Cbv', 'Rdv', 'Ldv']  # Cdv and Rav kept in linear space
DIM_PHYS    = len(TARGET_COLS)

LDV_IDX = TARGET_COLS.index('Ldv')
OBS_IDX = [i for i in range(DIM_PHYS) if i != LDV_IDX]

# Weights
PHYS_WEIGHTS  = [1.0, 1.0, 1.0, 0.0, 1.0, 5.0]   # Ldv weight = 0.0 (unobservable)
AUG_SIGMA     = 0.05   # std-dev of Gaussian noise added to normalised PCA scores during training
BETA_KL       = 0.3
BETA_LDV_MULT = 10.0
LAMBDA1       = 15.0
LAMBDA_PASSIV = 5.0
FREE_BITS     = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_data(pca_dir: str, proc_dir: str, splits_dir: str):
    sp      = np.load(os.path.join(pca_dir, "pca_score_scaler_params.npz"))
    dim_pca = int(sp['n_components'])
    pc_cols = [f'PC{i+1}' for i in range(dim_pca)]
    log_idx = [TARGET_COLS.index(t) for t in LOG_TARGETS]

    df_pca = pd.read_csv(os.path.join(proc_dir, "ssec_pca_final_v2.csv"))
    X_np   = df_pca[pc_cols].values.astype(np.float32)
    Y_np   = df_pca[TARGET_COLS].values.astype(np.float32)

    train_idx = np.load(os.path.join(splits_dir, "split_train_idx.npy"))
    val_idx   = np.load(os.path.join(splits_dir, "split_val_idx.npy"))
    test_idx  = np.load(os.path.join(splits_dir, "split_test_idx.npy"))

    Y_proc = Y_np.copy().astype(np.float64)
    for i in log_idx:
        Y_proc[:, i] = np.log10(np.abs(Y_proc[:, i]) + 1e-30)

    y_scalers = {}
    Y_scaled  = np.zeros_like(Y_proc, dtype=np.float32)
    for i, t in enumerate(TARGET_COLS):
        sc = StandardScaler()
        sc.fit(Y_proc[train_idx, i:i+1])
        Y_scaled[:, i] = sc.transform(Y_proc[:, i:i+1]).ravel()
        y_scalers[t]   = sc

    return (X_np, Y_scaled, y_scalers, log_idx,
            train_idx, val_idx, test_idx, dim_pca, sp)

def _inverse(Y_sc: np.ndarray, y_scalers, log_idx) -> np.ndarray:
    Y_out = np.zeros_like(Y_sc, dtype=np.float64)
    for i, t in enumerate(TARGET_COLS):
        Y_out[:, i] = y_scalers[t].inverse_transform(Y_sc[:, i:i+1]).ravel()
    for i in log_idx:
        Y_out[:, i] = 10.0 ** Y_out[:, i]
    return Y_out

def _make_loaders(X_np, Y_scaled, train_idx, val_idx, test_idx, batch=256):
    class DS(Dataset):
        def __init__(self, X, Y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.Y = torch.tensor(Y, dtype=torch.float32)
        def __len__(self):        return len(self.X)
        def __getitem__(self, i): return self.X[i], self.Y[i]

    ds = DS(X_np, Y_scaled)
    kw = dict(batch_size=batch, pin_memory=True, num_workers=2, persistent_workers=True)
    return (DataLoader(Subset(ds, train_idx.tolist()), shuffle=True,  **kw),
            DataLoader(Subset(ds, val_idx.tolist()),   shuffle=False, **kw),
            DataLoader(Subset(ds, test_idx.tolist()),  shuffle=False, **kw))

def _load_pca_buffers(pca_dir: str, sp):
    V_np  = np.load(os.path.join(pca_dir, "V_pca_bridge.npy")).astype(np.float32)
    mu_np = np.load(os.path.join(pca_dir, "mu_scaler_bridge.npy")).astype(np.float32)
    fs_np = np.load(os.path.join(pca_dir, "std_scaler_bridge.npy")).astype(np.float32)
    fm_np = np.load(os.path.join(pca_dir, "scaler_mean_bridge.npy")).astype(np.float32)
    sm_np = sp['score_mean'].astype(np.float32)
    ss_np = sp['score_std'].astype(np.float32)
    return (torch.tensor(V_np), torch.tensor(mu_np),
            torch.tensor(sm_np), torch.tensor(ss_np),
            torch.tensor(fm_np), torch.tensor(fs_np))

# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────
class CVAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=DIM_PHYS, hidden_dims=(128, 256, 256, 128)):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        self.shared    = nn.Sequential(*layers)
        self.fc_mu     = nn.Linear(d, latent_dim)
        self.fc_logvar = nn.Linear(d, latent_dim)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, x):
        h      = self.shared(x)
        mu     = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=4.0)
        return mu, logvar

class CVAEDecoder(nn.Module):
    def __init__(self, latent_dim=DIM_PHYS, output_dim=None, hidden_dims=(128, 256, 256, 128)):
        super().__init__()
        layers, d = [], latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)

class PhysicsSupervisedCVAE(nn.Module):
    def __init__(self, V_pca, mu_pca, score_mean, score_std, feature_mean, feature_std, dim_pca):
        super().__init__()
        self.encoder = CVAEEncoder(input_dim=dim_pca)
        self.decoder = CVAEDecoder(output_dim=dim_pca)
        for name, buf in [('V_pca', V_pca), ('mu_pca', mu_pca),
                          ('score_mean', score_mean), ('score_std', score_std),
                          ('feature_mean', feature_mean), ('feature_std', feature_std)]:
            self.register_buffer(name, buf)

    def pca_invert(self, z_norm):
        z_unnorm = z_norm * self.score_std + self.score_mean
        X_std    = z_unnorm @ self.V_pca.T + self.mu_pca
        return X_std * self.feature_std + self.feature_mean

    def reparameterise(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x_pca):
        mu, logvar = self.encoder(x_pca)
        z          = self.reparameterise(mu, logvar)

        # Stop gradient on Ldv
        z_for_decoder = torch.cat([
            z[:, :LDV_IDX],
            z[:, LDV_IDX:LDV_IDX + 1].detach(),
            z[:, LDV_IDX + 1:]
        ], dim=1)

        z_recon        = self.decoder(z_for_decoder)
        X_2610_decoded = self.pca_invert(z_recon)
        X_2610_true    = self.pca_invert(x_pca)
        return z_recon, mu, logvar, X_2610_decoded, X_2610_true

# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────
def passivity_loss(X_2610, Nf=NF):
    R11 = X_2610[..., :Nf];           I11 = X_2610[..., Nf:2*Nf]
    R21 = X_2610[..., 2*Nf:3*Nf];    I21 = X_2610[..., 3*Nf:4*Nf]
    R22 = X_2610[..., 4*Nf:5*Nf];    I22 = X_2610[..., 5*Nf:]

    mag_S11_sq = R11**2 + I11**2
    mag_S22_sq = R22**2 + I22**2
    mag_S21_sq = R21**2 + I21**2

    loss_port1 = torch.mean(F.relu(mag_S11_sq + mag_S21_sq - 1.0))
    loss_port2 = torch.mean(F.relu(mag_S22_sq + mag_S21_sq - 1.0))

    det_Re = R11*R22 - I11*I22 - (R21**2 - I21**2)
    det_Im = R11*I22 + I11*R22 - 2.0*R21*I21
    mag_det_sq    = det_Re**2 + det_Im**2
    det_violation = mag_S11_sq + mag_S22_sq + 2.0*mag_S21_sq - mag_det_sq - 1.0
    return loss_port1 + loss_port2 + torch.mean(F.relu(det_violation))

def get_annealed_weight(epoch, max_epoch=200):
    if epoch >= max_epoch: return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * epoch / max_epoch))

def cvae_loss(z_recon, x_pca, mu, logvar, y_gt, X_2610_decoded, epoch, pw):
    ramp = get_annealed_weight(epoch)
    L_recon = F.mse_loss(z_recon, x_pca)

    sigma2      = torch.exp(logvar)
    kl_per_dim  = 0.5 * (sigma2 + (mu - y_gt)**2 - 1.0 - logvar)

    beta_weights = torch.ones(DIM_PHYS, dtype=torch.float32, device=mu.device) * BETA_KL
    beta_weights[LDV_IDX] = BETA_KL * BETA_LDV_MULT

    thresholds = torch.zeros(DIM_PHYS, device=mu.device)
    thresholds[OBS_IDX] = FREE_BITS

    kl_clamped  = torch.max(kl_per_dim, thresholds.unsqueeze(0))
    kl_weighted = kl_clamped * beta_weights.unsqueeze(0)
    L_kl        = kl_weighted.mean()

    L_phys   = torch.mean(pw * (mu - y_gt)**2)
    L_passiv = passivity_loss(X_2610_decoded)

    L_total = L_recon + L_kl + LAMBDA1 * L_phys + LAMBDA_PASSIV * L_passiv * ramp
    return L_total, {
        'L_total' : L_total.item(), 'L_recon' : L_recon.item(),
        'L_kl'    : L_kl.item(),    'L_phys'  : L_phys.item(),
        'L_passiv': L_passiv.item()
    }

# ─────────────────────────────────────────────────────────────────────────────
# Training-time Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def _augment_pca_scores(x_pca: torch.Tensor, sigma: float = AUG_SIGMA) -> torch.Tensor:
    """
    Add i.i.d. Gaussian noise to normalised PCA scores during training.
    Implements a denoising-autoencoder objective:
      - Model input : x_pca + noise  (corrupted)
      - Recon target: x_pca          (clean)
    sigma=0.05 ≈ 5% of each dim's std. Applied only during training.
    """
    return x_pca + sigma * torch.randn_like(x_pca)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def _smape_obs(y_pred_phys, y_true_phys):
    p = y_pred_phys[:, OBS_IDX]
    t = y_true_phys[:, OBS_IDX]
    return np.mean(200.0 * np.abs(p - t) / (np.abs(t) + np.abs(p) + 1e-8))

def train_cvae(model, train_loader, val_loader, y_scalers, log_idx, device, epochs=500, lr=1e-3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    pw        = torch.tensor(PHYS_WEIGHTS, dtype=torch.float32, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    keys    = ['total', 'recon', 'kl', 'phys', 'passiv']
    history = {f'train_{k}': [] for k in keys}
    history.update({'val_total': [], 'val_smape_obs': [], 'val_smape_all': [], 'sigma_ld': [], 'sigma_obs_mean': []})

    best_val_smape, best_state, best_epoch = float('inf'), None, -1

    for epoch in range(1, epochs + 1):
        model.train()
        accum = {k: 0.0 for k in keys}; n_seen = 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            x_b_noisy = _augment_pca_scores(x_b)          # corrupted input
            optimizer.zero_grad()
            z_recon, mu, logvar, X_dec, _ = model(x_b_noisy)              # forward on noisy
            L_total, comp = cvae_loss(z_recon, x_b, mu, logvar, y_b, X_dec, epoch, pw)  # recon target = clean x_b
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = len(x_b); n_seen += bs
            for k in keys: accum[k] += comp[f'L_{k}'] * bs
        for k in keys: accum[k] /= n_seen; history[f'train_{k}'].append(accum[k])

        model.eval()
        vt = so_acc = sa_acc = sld_acc = sobs_acc = vs = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                z_recon, mu, logvar, X_dec, _ = model(x_b)
                Lt, _ = cvae_loss(z_recon, x_b, mu, logvar, y_b, X_dec, epoch, pw)
                vt += Lt.item() * len(x_b)

                sigma = torch.exp(0.5 * logvar)
                sld_acc  += sigma[:, LDV_IDX].mean().item() * len(x_b)
                sobs_acc += sigma[:, OBS_IDX].mean().item() * len(x_b)

                yp = _inverse(mu.cpu().numpy(), y_scalers, log_idx)
                yt = _inverse(y_b.cpu().numpy(), y_scalers, log_idx)

                so_acc += _smape_obs(yp, yt) * len(x_b)
                sa_acc += np.mean(200.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-8)) * len(x_b)
                vs += len(x_b)

        history['val_total'].append(vt / vs)
        history['val_smape_obs'].append(so_acc / vs)
        history['val_smape_all'].append(sa_acc / vs)
        history['sigma_ld'].append(sld_acc / vs)
        history['sigma_obs_mean'].append(sobs_acc / vs)
        scheduler.step()

        val_smape_obs = so_acc / vs
        if val_smape_obs < best_val_smape:
            best_val_smape = val_smape_obs
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch     = epoch

        if epoch % 20 == 0 or epoch == 1:
            ratio = (sld_acc/vs) / ((sobs_acc/vs) + 1e-8)
            print(f"Ep {epoch:>3d} | rec={accum['recon']:.5f} kl={accum['kl']:.5f} "
                  f"phy={accum['phys']:.5f} pas={accum['passiv']:.2e} || "
                  f"SMAPE(obs)={val_smape_obs:.3f}% ratio={ratio:.2f}×")

    model.load_state_dict(best_state)
    print(f"\n✅ CVAE best weights restored (val_SMAPE_obs={best_val_smape:.4f}% @ ep {best_epoch})")
    return model, history, best_epoch

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation and Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_cvae(model, loader, y_scalers, log_idx, device, split_name='Test',
                  plot_dir: str = None):
    model.eval()
    mu_list, y_list = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            _, mu, _, _, _ = model(x_b.to(device))
            mu_list.append(mu.cpu().numpy())
            y_list.append(y_b.numpy())

    yp = _inverse(np.concatenate(mu_list), y_scalers, log_idx)
    yt = _inverse(np.concatenate(y_list), y_scalers, log_idx)

    print(f"\n── Per-parameter Metrics [{split_name}] ──────────────────────────")
    smapes, r2s = {}, {}
    for k, name in enumerate(TARGET_COLS):
        t, p  = yt[:, k], yp[:, k]
        r2    = r2_score(t, p)
        smape = np.mean(200.0 * np.abs(t - p) / (np.abs(t) + np.abs(p) + 1e-8))
        smapes[name], r2s[name] = smape, r2
        flag = " [null-space — poor expected]" if name == 'Ldv' else ""
        qual = "✅ Excellent" if r2 >= 0.95 else "✅ Good" if r2 >= 0.90 else "⚠️ Moderate" if r2 >= 0.70 else "❌ Poor"
        print(f"  {name:>6s}  R²={r2:>8.4f}  SMAPE={smape:>7.3f}%  {qual}{flag}")

    # Parity + residual scatter plots
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        n = len(TARGET_COLS)
        COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728','#17becf']
        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
        for k, name in enumerate(TARGET_COLS):
            t, p = yt[:, k], yp[:, k]
            c = COLORS[k % len(COLORS)]
            ax = axes[0, k]
            lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
            ax.scatter(t, p, alpha=0.2, s=4, color=c)
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
            lbl = f'{name}  R²={r2s[name]:.3f}'
            if name == 'Ldv': lbl += ' [null]'
            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel('True', fontsize=8); ax.set_ylabel('Predicted', fontsize=8)
            ax2 = axes[1, k]
            ax2.scatter(t, p - t, alpha=0.2, s=4, color=c)
            ax2.axhline(0, color='red', lw=1, linestyle='--')
            ax2.set_title(f'{name} residuals  SMAPE={smapes[name]:.2f}%', fontsize=9)
            ax2.set_xlabel('True', fontsize=8); ax2.set_ylabel('Pred − True', fontsize=8)
        plt.suptitle(f'CVAE — Parity & Residual Plots [{split_name}]', fontsize=11)
        out = os.path.join(plot_dir, f"cvae_parity_{split_name.lower()}.png")
        plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"[cvae] Parity plots → '{out}'")

    return smapes, r2s

def ld_unobservability_report(model, loader, device):
    model.eval()
    sigma_all = []
    with torch.no_grad():
        for x_b, _ in loader:
            _, _, logvar, _, _ = model(x_b.to(device))
            sigma_all.append(torch.exp(0.5 * logvar).cpu().numpy())
    sigma_all = np.concatenate(sigma_all)
    mean_sigma = sigma_all.mean(axis=0)
    mean_obs = mean_sigma[OBS_IDX].mean()
    ratio = mean_sigma[LDV_IDX] / (mean_obs + 1e-8)

    print("\n── Posterior σ per Parameter (Ld Unobservability) ─────")
    for i, name in enumerate(TARGET_COLS):
        print(f"  {name:>6s}  {mean_sigma[i]:>10.4f}")
    print(f"  σ_Ldv={mean_sigma[LDV_IDX]:.4f}  σ̄_obs={mean_obs:.4f}  Ratio={ratio:.2f}×")
    return sigma_all

def bottleneck_report(model, loader, device):
    model.eval()
    mu_all = []
    with torch.no_grad():
        for x_b, _ in loader:
            _, mu, _, _, _ = model(x_b.to(device))
            mu_all.append(mu.cpu().numpy())
    mu_all = np.concatenate(mu_all)
    print("\n── Posterior Mean (μ) Saturation Diagnostic ─────────────────────")
    for i, name in enumerate(TARGET_COLS):
        col = mu_all[:, i]
        lo = (col < -3.0).mean() * 100
        hi = (col >  3.0).mean() * 100
        risk = "⚠️  HIGH" if (lo > 5 or hi > 5) else "✅  OK"
        print(f"  {name:>6s}  min={col.min():>8.4f}  max={col.max():>8.4f}  <-3={lo:>6.2f}%  >3={hi:>6.2f}%  {risk}")

def passivity_report(model, loader, device):
    model.eval()
    total_p = n = 0
    with torch.no_grad():
        for x_b, _ in loader:
            _, _, _, X_dec, _ = model(x_b.to(device))
            total_p += passivity_loss(X_dec).item() * len(x_b)
            n += len(x_b)
    mean_p = total_p / n
    print(f"\nPassivity Report (test): {mean_p:.4e}  [expected ~0]")
    return mean_p

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(28, 4))
    ep = range(1, len(history['train_total']) + 1)

    colours = {'recon':'steelblue','kl':'darkorange','phys':'green','passiv':'crimson'}
    for k, c in colours.items():
        axes[0].plot(ep, history[f'train_{k}'], label=f'L_{k}', color=c, lw=0.9)
    axes[0].set_yscale('log'); axes[0].set_title('Loss Components'); axes[0].legend()

    axes[1].plot(ep, history['train_total'], label='Train'); axes[1].plot(ep, history['val_total'], label='Val', ls='--')
    axes[1].set_title('Total Loss'); axes[1].legend()

    axes[2].plot(ep, history['val_smape_obs'], color='purple', label='5-param (obs)')
    axes[2].plot(ep, history['val_smape_all'], color='gray',   label='6-param (all)', ls='--')
    axes[2].set_title('Val SMAPE (%)'); axes[2].legend()

    axes[3].plot(ep, history['train_kl'], color='darkorange'); axes[3].set_title('KL Divergence')

    axes[4].plot(ep, history['sigma_ld'],       color='#d65f5f', lw=1.5, label='σ_Ldv')
    axes[4].plot(ep, history['sigma_obs_mean'], color='#4878d0', lw=1.5, label='σ̄_observable')
    axes[4].axhline(1.0, color='green', linestyle=':', lw=1.0, label='KL equil. (σ=1)')
    axes[4].set_title('σ over Training'); axes[4].legend()

    out = os.path.join(plot_dir, "cvae_v5_training_curves.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[cvae] Training curves → '{out}'")

def plot_decoder_validation(model, loader, device, plot_dir, n_samples=3):
    os.makedirs(plot_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        x_b, _ = next(iter(loader))
        _, _, _, X_dec, X_true = model(x_b[:n_samples].to(device))
    Xd, Xt = X_dec.cpu().numpy(), X_true.cpu().numpy()
    fq = np.linspace(0.04, 43.5, NF)
    panels = [('S11 Re', slice(None, NF)),   ('S11 Im', slice(NF, 2*NF)),
              ('S21 Re', slice(2*NF, 3*NF)), ('S21 Im', slice(3*NF, 4*NF)),
              ('S22 Re', slice(4*NF, 5*NF)), ('S22 Im', slice(5*NF, None))]
    fig, axes = plt.subplots(n_samples, 6, figsize=(26, 3.5*n_samples))
    for row in range(n_samples):
        for col, (title, sl) in enumerate(panels):
            ax = axes[row, col]
            ax.plot(fq, Xt[row, sl], 'k-',  lw=1.5, label='True')
            ax.plot(fq, Xd[row, sl], 'r--', lw=1.0, label='CVAE')
            mse = np.mean((Xt[row, sl] - Xd[row, sl])**2)
            ax.set_title(f'S{row+1} — {title}\nMSE={mse:.2e}', fontsize=7)
            if col == 0: ax.set_ylabel('S-param', fontsize=7)
            if row == 0 and col == 0: ax.legend(fontsize=7)
    out = os.path.join(plot_dir, "cvae_v5_decoder_validation.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[cvae] Decoder validation → '{out}'")

def posterior_uncertainty_plot(sigma_all, plot_dir, seed=42):
    os.makedirs(plot_dir, exist_ok=True)
    mean_sigma_obs = sigma_all[:, OBS_IDX].mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    parts = ax.violinplot([sigma_all[:, i] for i in range(DIM_PHYS)],
                          positions=range(DIM_PHYS), showmeans=True, showmedians=True)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor('#d65f5f' if i == LDV_IDX else '#4878d0')
        body.set_alpha(0.65)
    ax.axhline(mean_sigma_obs, color='gray', linestyle='--', lw=1.2,
               label=f'Mean σ (obs) = {mean_sigma_obs:.3f}')
    ax.axhline(1.0, color='green', linestyle=':', lw=1.0, label='KL equilibrium (σ=1)')
    ax.set_xticks(range(DIM_PHYS)); ax.set_xticklabels(TARGET_COLS, fontsize=11)
    ax.set_title('Posterior σ per Parameter (violin)')
    ax.set_ylabel('σ (posterior std dev)'); ax.legend(fontsize=8)

    ax2 = axes[1]
    rng   = np.random.default_rng(seed)          # seeded for reproducibility
    n_show = min(1000, len(sigma_all))
    idx   = rng.choice(len(sigma_all), n_show, replace=False)
    ratio = sigma_all[idx, LDV_IDX] / (sigma_all[idx][:, OBS_IDX].mean(axis=1) + 1e-8)
    ax2.hist(ratio, bins=50, color='#d65f5f', alpha=0.7, edgecolor='white')
    ax2.axvline(1.0, color='k', linestyle='--', lw=1.2, label='ratio=1')
    ax2.axvline(ratio.mean(), color='red', linestyle='-', lw=1.5,
                label=f'mean ratio = {ratio.mean():.2f}×')
    ax2.set_xlabel('σ_Ldv / σ̄_obs'); ax2.set_ylabel('Count')
    ax2.set_title('Ldv Unobservability Ratio Distribution')
    ax2.legend(fontsize=9)
    plt.suptitle('CVAE v5 — Posterior Uncertainty (UQ)', fontsize=11)
    out = os.path.join(plot_dir, "cvae_v5_posterior_uncertainty.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[cvae] Posterior UQ plot → '{out}'")

# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────
def run(pca_dir:    str = "data/pca_artifacts",
        proc_dir:   str = "data/processed",
        splits_dir: str = "data/splits",
        plot_dir:   str = "outputs/plots/cvae",
        ckpt_dir:   str = "outputs/checkpoints",
        epochs:     int = 500,
        lr:         float = 1e-3,
        seed:       int = 42):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[cvae] Device: {device}")

    (X_np, Y_scaled, y_scalers, log_idx,
     train_idx, val_idx, test_idx, dim_pca, sp) = _load_data(pca_dir, proc_dir, splits_dir)

    train_loader, val_loader, test_loader = _make_loaders(
        X_np, Y_scaled, train_idx, val_idx, test_idx)

    buffers = _load_pca_buffers(pca_dir, sp)
    model   = PhysicsSupervisedCVAE(*buffers, dim_pca=dim_pca).to(device)

    print(f"[cvae] DIM_PCA={dim_pca}  DIM_PHYS={DIM_PHYS}  epochs={epochs}")
    model, history, best_ep = train_cvae(
        model, train_loader, val_loader, y_scalers, log_idx, device, epochs, lr, seed)

    plot_training_curves(history, plot_dir)
    evaluate_cvae(model, test_loader, y_scalers, log_idx, device, plot_dir=plot_dir)
    sigma_all = ld_unobservability_report(model, test_loader, device)
    bottleneck_report(model, test_loader, device)
    plot_decoder_validation(model, test_loader, device, plot_dir)
    posterior_uncertainty_plot(sigma_all, plot_dir)
    passivity_report(model, test_loader, device)

    ckpt_path = os.path.join(ckpt_dir, "cvae_v5_physics_supervised_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'y_scalers':        y_scalers,
        'log_targets':      LOG_TARGETS,
        'target_cols':      TARGET_COLS,
        'best_epoch':       best_ep,
        'total_epochs':     epochs,
        'history':          history,
        'dim_pca':          dim_pca,
        'dim_phys':         DIM_PHYS,
        'beta_kl':          BETA_KL,
    }, ckpt_path)
    print(f"\n[cvae] ✅ Checkpoint saved → '{ckpt_path}'  (best ep {best_ep}/{epochs})")
    return ckpt_path
