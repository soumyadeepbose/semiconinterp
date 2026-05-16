# -*- coding: utf-8 -*-
"""
piae_model.py
=============
Physics-Informed Autoencoder (PIAE v3) — updated for S11 + S21 + S22 (2610-dim).

Key changes from the old (S11+S22 only) version:
  - DIM_RAW = 2610  (was 1740)
  - Passivity loss uses the full 3-port conditions including S21 and the
    strict determinant condition.
  - PIAE forward includes a small non-linear residual_net branch.
  - Ld is excluded from the bottleneck (DIM_PHYS = 5).
  - TARGET_COLS = ['Rbv', 'Cbv', 'Rdv', 'Cdv', 'Rav']

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from csv_utils import csv_save
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constants  (must mirror data_processing.py)
# ─────────────────────────────────────────────────────────────────────────────
NF       = 435
DIM_RAW  = 2610      # 6 × NF
DIM_PHYS = 5         # Ld excluded

TARGET_COLS  = ['Rbv', 'Cbv', 'Rdv', 'Cdv', 'Rav']
LOG_TARGETS  = ['Rbv', 'Cbv', 'Rdv']        # Cdv and Rav stay in linear space
PHYS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 5.0]   # Rav gets extra weight
AUG_SIGMA    = 0.05   # std-dev of Gaussian noise added to normalised PCA scores during training


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_data(pca_dir: str, proc_dir: str, splits_dir: str):
    sp        = np.load(os.path.join(pca_dir, "pca_score_scaler_params.npz"))
    dim_pca   = int(sp['n_components'])
    pc_cols   = [f'PC{i+1}' for i in range(dim_pca)]
    log_idx   = [TARGET_COLS.index(t) for t in LOG_TARGETS]

    df_pca   = pd.read_csv(os.path.join(proc_dir, "ssec_pca_final_v2.csv"))
    X_np     = df_pca[pc_cols].values.astype(np.float32)
    Y_np     = df_pca[TARGET_COLS].values.astype(np.float32)

    train_idx = np.load(os.path.join(splits_dir, "split_train_idx.npy"))
    val_idx   = np.load(os.path.join(splits_dir, "split_val_idx.npy"))
    test_idx  = np.load(os.path.join(splits_dir, "split_test_idx.npy"))

    # Log-transform + MinMaxScale targets to [0, 1] (matches Sigmoid encoder output)
    Y_proc = Y_np.copy().astype(np.float64)
    for i in log_idx:
        Y_proc[:, i] = np.log10(np.abs(Y_proc[:, i]) + 1e-30)

    y_scalers = {}
    Y_scaled  = np.zeros_like(Y_proc, dtype=np.float32)
    for i, t in enumerate(TARGET_COLS):
        sc = MinMaxScaler(feature_range=(0.0, 1.0))
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

    ds   = DS(X_np, Y_scaled)
    kw   = dict(batch_size=batch, pin_memory=True, num_workers=2, persistent_workers=True)
    return (DataLoader(Subset(ds, train_idx.tolist()), shuffle=True,  **kw),
            DataLoader(Subset(ds, val_idx.tolist()),   shuffle=False, **kw),
            DataLoader(Subset(ds, test_idx.tolist()),  shuffle=False, **kw))


def _load_pca_buffers(pca_dir: str, sp):
    V_np    = np.load(os.path.join(pca_dir, "V_pca_bridge.npy")).astype(np.float32)
    mu_np   = np.load(os.path.join(pca_dir, "mu_scaler_bridge.npy")).astype(np.float32)
    fs_np   = np.load(os.path.join(pca_dir, "std_scaler_bridge.npy")).astype(np.float32)
    fm_np   = np.load(os.path.join(pca_dir, "scaler_mean_bridge.npy")).astype(np.float32)
    sm_np   = sp['score_mean'].astype(np.float32)
    ss_np   = sp['score_std'].astype(np.float32)
    return (torch.tensor(V_np), torch.tensor(mu_np),
            torch.tensor(sm_np), torch.tensor(ss_np),
            torch.tensor(fm_np), torch.tensor(fs_np))


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim=DIM_PHYS, hidden=(128, 256, 128)):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, output_dim), nn.Sigmoid()]  # bound to [0,1] to match MinMaxScaler targets
        self.net = nn.Sequential(*layers)
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)


class Decoder(nn.Module):
    def __init__(self, input_dim=DIM_PHYS, output_dim=None, hidden=(128, 256, 128)):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, y): return self.net(y)


class PIAE(nn.Module):
    """Physics-Informed Autoencoder v3 (S11+S21+S22, 2610-dim)."""
    def __init__(self, V_pca, mu_pca, score_mean, score_std, feature_mean, feature_std, dim_pca):
        super().__init__()
        self.encoder      = Encoder(input_dim=dim_pca, output_dim=DIM_PHYS)
        self.decoder      = Decoder(input_dim=DIM_PHYS, output_dim=dim_pca)
        self.residual_net = nn.Sequential(
            nn.Linear(DIM_RAW, 512), nn.ReLU(), nn.Linear(512, DIM_RAW))
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)
        for name, buf in [('V_pca', V_pca), ('mu_pca', mu_pca),
                          ('score_mean', score_mean), ('score_std', score_std),
                          ('feature_mean', feature_mean), ('feature_std', feature_std)]:
            self.register_buffer(name, buf)

    def pca_invert(self, z_norm):
        z_u  = z_norm * self.score_std + self.score_mean
        X_s  = z_u @ self.V_pca.T + self.mu_pca
        return X_s * self.feature_std + self.feature_mean

    def forward(self, x_pca):
        y_bott         = self.encoder(x_pca)
        z_recon        = self.decoder(y_bott)
        X_pca_only     = self.pca_invert(z_recon)
        X_true         = self.pca_invert(x_pca)
        X_decoded      = X_pca_only + 0.01 * self.residual_net(X_pca_only)
        return z_recon, y_bott, X_decoded, X_true


# ─────────────────────────────────────────────────────────────────────────────
# Physics losses
# ─────────────────────────────────────────────────────────────────────────────
def passivity_loss(X: torch.Tensor, Nf: int = NF) -> torch.Tensor:
    R11, I11 = X[..., :Nf],       X[..., Nf:2*Nf]
    R21, I21 = X[..., 2*Nf:3*Nf], X[..., 3*Nf:4*Nf]
    R22, I22 = X[..., 4*Nf:5*Nf], X[..., 5*Nf:]
    m11 = R11**2 + I11**2;  m21 = R21**2 + I21**2;  m22 = R22**2 + I22**2
    l1  = torch.mean(F.relu(m11 + m21 - 1.0))
    l2  = torch.mean(F.relu(m22 + m21 - 1.0))
    dr  = R11*R22 - I11*I22 - (R21**2 - I21**2)
    di  = R11*I22 + I11*R22 - 2*R21*I21
    l3  = torch.mean(F.relu(m11 + m22 + 2*m21 - (dr**2 + di**2) - 1.0))
    return l1 + l2 + l3


def _ramp(epoch, max_epoch=200):
    if epoch >= max_epoch: return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * epoch / max_epoch))


def quad_loss(z_recon, x_pca, y_bott, y_gt, X_dec, epoch, pw):
    ramp     = _ramp(epoch)
    L_recon  = F.mse_loss(z_recon, x_pca)
    L_phys   = torch.mean(pw * (y_bott - y_gt)**2)
    L_passiv = passivity_loss(X_dec)
    L_total  = 1.0*L_recon + 15.0*L_phys + 5.0*L_passiv*ramp
    return L_total, {'L_recon': L_recon.item(), 'L_phys': L_phys.item(),
                     'L_passiv': L_passiv.item(), 'L_smooth': 0.0,
                     'L_total': L_total.item()}


# ─────────────────────────────────────────────────────────────────────────────
# Training-time Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def _augment_pca_scores(x_pca: torch.Tensor, sigma: float = AUG_SIGMA) -> torch.Tensor:
    """
    Add i.i.d. Gaussian noise to normalised PCA scores during training.
    Implements a denoising-autoencoder objective:
      - Model input : x_pca + noise  (corrupted)
      - Recon target: x_pca          (clean)
    Noise lives in PCA-score space (zero-mean, unit-std per dim),
    so sigma=0.05 ≈ 5% of each dimension's std — mild but effective.
    Applied only during training; validation/test use clean scores.
    """
    return x_pca + sigma * torch.randn_like(x_pca)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_piae(model, train_loader, val_loader, y_scalers, log_idx,
               device, epochs=500, lr=1e-3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    pw        = torch.tensor(PHYS_WEIGHTS, dtype=torch.float32, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    keys    = ['total', 'recon', 'phys', 'passiv', 'smooth']
    history = {f'train_{k}': [] for k in keys}
    history.update({'val_total': [], 'val_mape': []})

    best_mape, best_state, best_ep = float('inf'), None, -1

    for ep in range(1, epochs+1):
        model.train()
        acc = {k: 0.0 for k in keys}; n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb_noisy = _augment_pca_scores(xb)     # corrupted input
            optimizer.zero_grad()
            zr, yh, Xd, _ = model(xb_noisy)        # forward on noisy input
            Lt, comp = quad_loss(zr, xb, yh, yb, Xd, ep, pw)  # recon target = clean xb
            Lt.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = len(xb); n += bs
            for k in keys: acc[k] += comp[f'L_{k}'] * bs
        for k in keys: acc[k] /= n; history[f'train_{k}'].append(acc[k])

        model.eval(); vt = vm = vs = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                zr, yh, Xd, _ = model(xb)
                Lt, _ = quad_loss(zr, xb, yh, yb, Xd, ep, pw)
                vt += Lt.item() * len(xb)
                yp = _inverse(yh.cpu().numpy(), y_scalers, log_idx)
                yt = _inverse(yb.cpu().numpy(), y_scalers, log_idx)
                vm += np.mean(np.abs(yp - yt) / (np.abs(yt) + 1e-8)) * 100. * len(xb)
                vs += len(xb)
        val_mape = vm / vs
        history['val_total'].append(vt / vs)
        history['val_mape'].append(val_mape)
        scheduler.step()

        if val_mape < best_mape:
            best_mape  = val_mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep    = ep

        if ep % 20 == 0 or ep == 1:
            print(f"Ep {ep:>3d} | rec={acc['recon']:.5f} phy={acc['phys']:.5f} "
                  f"pas={acc['passiv']:.2e} | val_MAPE={val_mape:.3f}%")

    model.load_state_dict(best_state)
    print(f"\n✅ PIAE best weights restored  (val_MAPE={best_mape:.4f}% @ ep {best_ep})")
    return model, history, best_ep


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, y_scalers, log_idx, device, split_name='Test',
             plot_dir: str = None):
    model.eval()
    yp_list, yt_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            _, yh, _, _ = model(xb.to(device))
            yp_list.append(yh.cpu().numpy())
            yt_list.append(yb.numpy())
    yp = _inverse(np.concatenate(yp_list), y_scalers, log_idx)
    yt = _inverse(np.concatenate(yt_list), y_scalers, log_idx)

    print(f"\n── Per-parameter Metrics [{split_name}] ──────────────────────────")
    mapes, r2s = {}, {}
    for k, name in enumerate(TARGET_COLS):
        t, p  = yt[:, k], yp[:, k]
        r2    = r2_score(t, p)
        mape  = np.mean(np.abs(t-p)/(np.abs(t)+1e-8))*100.
        mapes[name], r2s[name] = mape, r2
        st = "✅ Excellent" if r2>=0.95 else "✅ Good" if r2>=0.90 else "⚠️ Moderate" if r2>=0.70 else "❌ Poor"
        print(f"  {name:>6s}  R²={r2:>8.4f}  MAPE={mape:>7.3f}%  {st}")
    print(f"  {'Mean':>6s}  R²={np.mean(list(r2s.values())):>8.4f}  "
          f"MAPE={np.mean(list(mapes.values())):>7.3f}%")

    # Parity scatter plots
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        n = len(TARGET_COLS)
        COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728']
        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
        for k, name in enumerate(TARGET_COLS):
            t, p = yt[:, k], yp[:, k]
            c = COLORS[k % len(COLORS)]
            # Parity plot
            ax = axes[0, k]
            lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
            ax.scatter(t, p, alpha=0.2, s=4, color=c)
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
            ax.set_title(f'{name}  R²={r2s[name]:.3f}', fontsize=9)
            ax.set_xlabel('True', fontsize=8); ax.set_ylabel('Predicted', fontsize=8)
            # Residual plot
            ax2 = axes[1, k]
            ax2.scatter(t, p - t, alpha=0.2, s=4, color=c)
            ax2.axhline(0, color='red', lw=1, linestyle='--')
            ax2.set_title(f'{name} residuals  MAPE={mapes[name]:.2f}%', fontsize=9)
            ax2.set_xlabel('True', fontsize=8); ax2.set_ylabel('Pred − True', fontsize=8)
        plt.suptitle(f'PIAE — Parity & Residual Plots [{split_name}]', fontsize=11)
        out = os.path.join(plot_dir, f"piae_parity_{split_name.lower()}.png")
        # — CSV export: true & predicted values per parameter
        rec = {'sample_idx': np.arange(len(yt))}
        for k, name in enumerate(TARGET_COLS):
            rec[f'{name}_true'] = yt[:, k]
            rec[f'{name}_pred'] = yp[:, k]
        csv_save(rec, out)
        plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"[piae] Parity plots → '{out}'")

    return mapes, r2s


def bottleneck_report(model, loader, device):
    model.eval(); bv = []
    with torch.no_grad():
        for xb, _ in loader:
            _, yh, _, _ = model(xb.to(device))
            bv.append(yh.cpu().numpy())
    bv = np.concatenate(bv)
    print("\n── Bottleneck Saturation Diagnostic ─────────────────────────────")
    for i, name in enumerate(TARGET_COLS):
        col = bv[:, i]
        lo  = (col < -3.0).mean()*100; hi = (col > 3.0).mean()*100
        risk = "⚠️  HIGH" if (lo>5 or hi>5) else "✅  OK"
        print(f"  {name:>6s}  min={col.min():>7.4f}  max={col.max():>7.4f}  "
              f"<-3={lo:>6.2f}%  >3={hi:>6.2f}%  {risk}")


def passivity_report(model, loader, device):
    model.eval(); tp = n = 0
    with torch.no_grad():
        for xb, _ in loader:
            _, _, Xd, _ = model(xb.to(device))
            tp += passivity_loss(Xd).item() * len(xb); n += len(xb)
    print(f"\nPassivity (test): mean violation = {tp/n:.4e}  [expected ~0]")
    return tp/n


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    ep = range(1, len(history['train_total'])+1)
    for k, c in [('recon','steelblue'),('phys','darkorange'),('passiv','green'),('smooth','red')]:
        axes[0].plot(ep, history[f'train_{k}'], label=f'L_{k}', color=c, lw=0.9)
    axes[0].set_yscale('log'); axes[0].set_title('Loss Components (train)'); axes[0].legend(fontsize=7)
    axes[1].plot(ep, history['train_total'], label='train')
    axes[1].plot(ep, history['val_total'], label='val', linestyle='--')
    axes[1].set_title('Total Loss'); axes[1].legend()
    axes[2].plot(ep, history['val_mape'], color='purple')
    axes[2].set_title('Validation MAPE (%)')
    out = os.path.join(plot_dir, "piae_training_curves.png")
    # — CSV export
    hist_df = pd.DataFrame({'epoch': list(ep)})
    for k in ['total', 'recon', 'phys', 'passiv', 'smooth']:
        hist_df[f'train_{k}'] = history[f'train_{k}']
    hist_df['val_total'] = history['val_total']
    hist_df['val_mape']  = history['val_mape']
    csv_save(hist_df, out)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[piae] Training curves → '{out}'")


def plot_decoder_validation(model, loader, device, plot_dir, n_samples=3):
    os.makedirs(plot_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        xb, _ = next(iter(loader))
        _, _, Xd, Xt = model(xb[:n_samples].to(device))
    Xd = Xd.cpu().numpy(); Xt = Xt.cpu().numpy()
    fq = np.linspace(FREQ_START_GHZ := 0.04, FREQ_STOP_GHZ := 43.5, NF)
    panels = [('S11 Re', slice(None, NF)),     ('S11 Im', slice(NF, 2*NF)),
              ('S21 Re', slice(2*NF, 3*NF)),   ('S21 Im', slice(3*NF, 4*NF)),
              ('S22 Re', slice(4*NF, 5*NF)),   ('S22 Im', slice(5*NF, None))]
    fig, axes = plt.subplots(n_samples, 6, figsize=(24, 3.5*n_samples))
    for row in range(n_samples):
        for col, (title, sl) in enumerate(panels):
            ax = axes[row, col]
            ax.plot(fq, Xt[row, sl], 'k-',  lw=1.5, label='True')
            ax.plot(fq, Xd[row, sl], 'r--', lw=1.0, label='Decoded')
            mse = np.mean((Xt[row, sl] - Xd[row, sl])**2)
            ax.set_title(f'Sample {row+1} — {title}  [MSE={mse:.2e}]', fontsize=8)
            ax.set_xlabel('Freq (GHz)', fontsize=7); ax.tick_params(labelsize=7)
            if col==0: ax.set_ylabel('S-param', fontsize=7)
            if row==0 and col==0: ax.legend(fontsize=7)
    plt.suptitle('PIAE Decoder Reconstruction Quality', fontsize=10)
    out = os.path.join(plot_dir, "piae_decoder_validation.png")
    # — CSV export: one sub-CSV per sample
    for row in range(n_samples):
        rec = {'freq_GHz': fq}
        for col, (title, sl) in enumerate(panels):
            rec[f'{title.replace(" ","_")}_true']    = Xt[row, sl]
            rec[f'{title.replace(" ","_")}_decoded'] = Xd[row, sl]
        csv_save(rec, out, suffix=f'__sample{row+1}')
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[piae] Decoder validation → '{out}'")

# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────
def run(pca_dir:    str = "data/pca_artifacts",
        proc_dir:   str = "data/processed",
        splits_dir: str = "data/splits",
        plot_dir:   str = "outputs/plots/piae",
        ckpt_dir:   str = "outputs/checkpoints",
        epochs:     int = 500,
        lr:         float = 1e-3,
        seed:       int = 42):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[piae] Device: {device}")

    (X_np, Y_scaled, y_scalers, log_idx,
     train_idx, val_idx, test_idx, dim_pca, sp) = _load_data(pca_dir, proc_dir, splits_dir)

    train_loader, val_loader, test_loader = _make_loaders(
        X_np, Y_scaled, train_idx, val_idx, test_idx)

    buffers = _load_pca_buffers(pca_dir, sp)
    model   = PIAE(*buffers, dim_pca=dim_pca).to(device)

    print(f"[piae] DIM_PCA={dim_pca}  DIM_PHYS={DIM_PHYS}  epochs={epochs}")
    model, history, best_ep = train_piae(
        model, train_loader, val_loader, y_scalers, log_idx, device, epochs, lr, seed)

    plot_training_curves(history, plot_dir)
    evaluate(model, test_loader, y_scalers, log_idx, device, plot_dir=plot_dir)
    bottleneck_report(model, test_loader, device)
    plot_decoder_validation(model, test_loader, device, plot_dir)
    passivity_report(model, test_loader, device)

    ckpt_path = os.path.join(ckpt_dir, "piae_v3_s21_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'y_scalers':        y_scalers,
        'log_targets':      LOG_TARGETS,
        'target_cols':      TARGET_COLS,
        'best_epoch':       best_ep,
        'total_epochs':     epochs,
        'history':          history,
        'dim_pca':          dim_pca,
    }, ckpt_path)
    print(f"\n[piae] ✅ Checkpoint saved → '{ckpt_path}'  (best ep {best_ep}/{epochs})")
    return ckpt_path
