"""
Near-Field Antenna Sweep — Autoencoder-Based Anomaly Detection
==============================================================
HYBRID APPROACH  (Gold Standard for ML on hardware data)

Instead of purely synthetic cosines (academically weak) or running
thousands of CST simulations (computationally prohibitive), this script:

  1.  Loads ONE accurate CST simulation  (Simulated_NF_Data.txt)
  2.  Extracts the real antenna physics   (main lobe, side lobes, nulls)
  3.  Generates 1000s of augmented copies (Data Augmentation)
  4.  Trains the Autoencoder on REAL physics in ~30 seconds
  5.  Deploys for anomaly detection + smart interpolation

Usage
-----
    python nf_autoencoder.py                             # train + demo
    python nf_autoencoder.py --infer raw_sweep.csv       # inference only
    python nf_autoencoder.py --cst MyCleanSweep.csv      # custom CST file
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# 1.  MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════
class NearFieldAutoencoder(nn.Module):
    """
    Lightweight 1-D convolutional autoencoder for 360-point sweeps.

    Architecture
    ------------
    Encoder:  360 → 180 → 90 → 45   (latent)
    Decoder:  45  → 90  → 180 → 360 (reconstruction)

    The bottleneck forces the network to learn only the smooth,
    physically meaningful structure of the radiation pattern.
    Sharp motor-jitter spikes cannot be encoded, so the reconstruction
    error spikes at exactly those corrupted coordinates.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2,
                               padding=3, output_padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ═══════════════════════════════════════════════════════════════════════════
# 2.  CST SIMULATION LOADER
# ═══════════════════════════════════════════════════════════════════════════
def load_cst_simulation(filepath: str, n_points: int = 360) -> np.ndarray:
    """
    Load a CST simulation and extract a clean 1-D magnitude profile.

    Handles TWO formats automatically:

    A)  **CST planar export** (``Simulated_NF_Data.txt``):
        Whitespace-separated, 2 header lines.
        Columns: x, y, z, ExRe, ExIm, EyRe, EyIm, EzRe, EzIm.
        → Extracts center row (y = 0) of |Ey|, maps x-position to
          observation angle via θ = arctan(x / z), and interpolates
          to a full 360° sweep with noise-floor padding.

    B)  **Simple CSV** (angle, mag_dB, phase_deg):
        → Uses column 1 (magnitude) directly.

    Returns
    -------
    clean_base : np.ndarray, shape (n_points,) — magnitude in dB
    """
    with open(filepath, "r") as f:
        first_line = f.readline()

    is_cst_planar = ("ExRe" in first_line or "x [mm]" in first_line)

    if is_cst_planar:
        # ── Format A: CST planar near-field export ──────────────────
        data = np.loadtxt(filepath, skiprows=2)

        x     = data[:, 0]   # mm
        y     = data[:, 1]   # mm
        z     = data[:, 2]   # mm  (constant, e.g. 115)
        EyRe  = data[:, 5]
        EyIm  = data[:, 6]

        # Centre row  (y ≈ 0, tolerance 0.5 mm)
        mask = np.abs(y) < 0.5
        x_c  = x[mask]
        Ey_c = EyRe[mask] + 1j * EyIm[mask]
        z_c  = z[mask][0]

        # Sort by x
        order = np.argsort(x_c)
        x_c   = x_c[order]
        Ey_c  = Ey_c[order]

        # |Ey| → dB  (normalised: peak = 0 dB)
        mag    = np.abs(Ey_c)
        mag_dB = 20.0 * np.log10(mag / mag.max() + 1e-12)

        # Map spatial x to observation angle (boresight = 180°)
        theta_deg = np.degrees(np.arctan2(x_c, z_c)) + 180.0

        # ── Build full 360° with smooth taper to noise floor ────────
        noise_floor = -50.0
        taper       = 15.0   # degrees of cosine roll-off each side

        t_lo = theta_deg.min()
        t_hi = theta_deg.max()

        # Anchor points beyond the measured range
        anchors_a = np.array([0.0,
                              t_lo - taper,
                              t_lo - taper * 0.3,
                              t_hi + taper * 0.3,
                              t_hi + taper,
                              359.0])
        anchors_v = np.array([noise_floor,
                              noise_floor,
                              noise_floor + 5.0,
                              noise_floor + 5.0,
                              noise_floor,
                              noise_floor])

        all_a = np.concatenate([theta_deg, anchors_a])
        all_v = np.concatenate([mag_dB,    anchors_v])

        order  = np.argsort(all_a)
        all_a  = all_a[order]
        all_v  = all_v[order]

        query      = np.linspace(0, 359, n_points)
        clean_base = np.interp(query, all_a, all_v).astype(np.float32)

        print(f"  [OK] Loaded CST planar data  ({filepath})")
        print(f"    Centre-row |Ey| extracted -- {len(x_c)} spatial points")
        print(f"    Measured angular range : {t_lo:.1f} deg - {t_hi:.1f} deg")
        print(f"    Interpolated to {n_points} points (full 360 deg)")

    else:
        # ── Format B: Simple CSV ────────────────────────────────────
        df  = pd.read_csv(filepath)
        mag = df.iloc[:, 1].values.astype(np.float32)

        if len(mag) != n_points:
            query      = np.linspace(0, 359, n_points)
            angles_src = np.linspace(0, 359, len(mag))
            clean_base = np.interp(query, angles_src, mag).astype(np.float32)
        else:
            clean_base = mag.copy()

        # Normalise peak to 0 dB
        clean_base -= clean_base.max()

        print(f"  [OK] Loaded CSV sweep  ({filepath}, {len(mag)} pts -> {n_points})")

    return clean_base


# ═══════════════════════════════════════════════════════════════════════════
# 3.  DATA AUGMENTATION  (Hybrid Approach)
# ═══════════════════════════════════════════════════════════════════════════
def generate_cst_augmented_sweeps(clean_base: np.ndarray,
                                  batch_size: int = 64) -> torch.Tensor:
    """
    Generate a batch of augmented training sweeps from a SINGLE CST base.

    Each copy is perturbed with realistic measurement imperfections
    that do NOT constitute anomalies:

      - Gaussian noise    -- VNA thermal noise          (sigma in 0.1-1.5 dB)
      - Amplitude scaling -- cable-loss / gain drift    (x0.95 ... x1.05)
      - Angular jitter    -- slight motor offset        (+/-2 sample points)
      - Baseline drift    -- temperature-induced shift  (+/-0.5 dB linear)

    Returns
    -------
    torch.Tensor — shape (batch_size, 1, n_points)
    """
    n = len(clean_base)
    sweeps = np.zeros((batch_size, 1, n), dtype=np.float32)

    for i in range(batch_size):
        s = clean_base.copy()

        # A -- Amplitude scaling  (cable loss / gain drift)
        s *= np.random.uniform(0.95, 1.05)

        # B -- Gaussian thermal noise  (VNA noise floor)
        noise_std = np.random.uniform(0.1, 1.5)
        s += np.random.normal(0, noise_std, n).astype(np.float32)

        # C -- Angular jitter  (small rotational offset from the motor)
        shift = np.random.randint(-2, 3)
        if shift:
            s = np.roll(s, shift)

        # D -- Baseline drift  (temperature-induced DC offset)
        drift = np.random.uniform(-0.5, 0.5)
        s += np.linspace(-drift, drift, n, dtype=np.float32)

        # E -- Normalise to [-1, 1]  (from dB range ~[-50, 0])
        s = (s + 25.0) / 25.0

        sweeps[i, 0, :] = s

    return torch.from_numpy(sweeps)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
def train_model(model: nn.Module,
                clean_base: np.ndarray,
                epochs: int = 500,
                batch_size: int = 64,
                lr: float = 1e-3,
                save_path: str = "models/nf_autoencoder.pth",
                device: str = "cpu") -> list:
    """
    Train the autoencoder on augmented CST sweeps.

    A FRESH augmented batch is generated every epoch, so the model
    effectively sees  epochs × batch_size  unique training samples
    without ever storing them all in memory.
    """
    model.to(device).train()
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                             T_max=epochs)
    criterion  = nn.MSELoss()
    loss_hist  = []

    print(f"\n{'=' * 62}")
    print(f"  Training NearFieldAutoencoder")
    print(f"  Epochs: {epochs}  |  Batch: {batch_size}  |"
          f"  Effective samples: {epochs * batch_size:,}")
    print(f"  Data source: CST-augmented sweeps  (hybrid approach)")
    print(f"{'=' * 62}")

    for epoch in range(1, epochs + 1):
        batch = generate_cst_augmented_sweeps(clean_base, batch_size).to(device)

        optimiser.zero_grad()
        recon = model(batch)
        loss  = criterion(recon, batch)
        loss.backward()
        optimiser.step()
        scheduler.step()

        lv = loss.item()
        loss_hist.append(lv)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{epochs}  |  Loss: {lv:.6f}  |"
                  f"  LR: {scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), save_path)
    print(f"\n  [OK] Model weights saved -> {save_path}\n")
    return loss_hist


# ═══════════════════════════════════════════════════════════════════════════
# 5.  ANOMALY DETECTION   (Adaptive Threshold: μ + 3σ)
# ═══════════════════════════════════════════════════════════════════════════
def detect_anomalies(model: nn.Module,
                     sweep: np.ndarray,
                     device: str = "cpu"):
    """
    Run inference on a raw sweep and flag anomalies.

    The adaptive threshold  μ + 3σ  is computed from the per-point
    reconstruction errors.  This is robust across varying SNR levels
    in different anechoic-chamber environments.

    Returns
    -------
    anomaly_indices : np.ndarray of flagged indices
    errors          : np.ndarray of per-point |reconstruction error|
    reconstructed   : np.ndarray of the model's smooth output
    threshold       : float — the adaptive threshold used
    """
    model.to(device).eval()

    # Normalise input to [-1, 1] (same as training)
    sweep_norm = (sweep.astype(np.float32) + 25.0) / 25.0
    t_in = torch.from_numpy(sweep_norm).view(1, 1, -1).to(device)

    with torch.no_grad():
        t_out = model(t_in)

    recon_norm = t_out.squeeze().cpu().numpy()

    # Denormalise back to dB
    recon = recon_norm * 25.0 - 25.0

    errors = np.abs(sweep - recon)

    mu        = errors.mean()
    sigma     = errors.std()
    threshold = mu + 3.0 * sigma

    anomaly_indices = np.where(errors > threshold)[0]
    return anomaly_indices, errors, recon, threshold


# ═══════════════════════════════════════════════════════════════════════════
# 6.  SMART INTERPOLATION   (fix ONLY the flagged points)
# ═══════════════════════════════════════════════════════════════════════════
def smart_interpolate(angles: np.ndarray,
                      magnitudes: np.ndarray,
                      phases: np.ndarray,
                      anomaly_indices: np.ndarray):
    """
    Linearly interpolate magnitude and phase at ONLY the anomalous
    coordinates, using the nearest clean neighbours as anchors.
    """
    mag_fix = magnitudes.copy()
    pha_fix = phases.copy()

    clean_mask = np.ones(len(angles), dtype=bool)
    clean_mask[anomaly_indices] = False
    clean_idx = np.where(clean_mask)[0]

    if len(clean_idx) < 2:
        print("  [WARN] Too few clean points -- returning raw data.")
        return mag_fix, pha_fix

    mag_fix[anomaly_indices] = np.interp(
        angles[anomaly_indices], angles[clean_idx], magnitudes[clean_idx])
    pha_fix[anomaly_indices] = np.interp(
        angles[anomaly_indices], angles[clean_idx], phases[clean_idx])

    return mag_fix, pha_fix


# ═══════════════════════════════════════════════════════════════════════════
# 7.  DEMO:  Inject Known Anomalies into the CST Base
# ═══════════════════════════════════════════════════════════════════════════
def generate_raw_sweep_with_anomalies(clean_base: np.ndarray,
                                      n_anomalies: int = 8):
    """
    Start from the REAL CST-derived pattern and inject known anomalies
    to demonstrate + validate the detection -> interpolation pipeline.

    Returns
    -------
    angles, mag_raw, phase_raw, mag_clean, phase_clean, truth_indices
    """
    n      = len(clean_base)
    angles = np.arange(n, dtype=np.float64)

    # Magnitude: start from real CST physics
    mag_clean = clean_base.copy()

    # Phase: realistic ramp (linear + curvature)
    phase_clean = (120.0 * np.sin(np.radians(angles - 180.0))
                   + 5.0 * np.sin(2.0 * np.radians(angles - 180.0)))

    # Add mild measurement noise  (NOT anomalous)
    mag_raw   = mag_clean   + np.random.normal(0, 0.3, n).astype(np.float32)
    phase_raw = phase_clean + np.random.normal(0, 1.0, n)

    # ── Inject anomalies  (motor jitter / cable spikes) ─────────────
    rng   = np.random.default_rng(42)
    truth = rng.choice(np.arange(20, 340), size=n_anomalies, replace=False)
    truth.sort()

    for idx in truth:
        mag_raw[idx]   += rng.choice([-1, 1]) * rng.uniform(8, 15)
        phase_raw[idx] += rng.choice([-1, 1]) * rng.uniform(30, 90)

    return angles, mag_raw, phase_raw, mag_clean, phase_clean, truth


# ═══════════════════════════════════════════════════════════════════════════
# 8.  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════
def plot_results(angles, mag_raw, mag_recon, mag_cleaned,
                 errors, threshold, anom_det, anom_truth,
                 loss_history):
    """Four-panel diagnostic plot: loss, raw+flags, error, before/after."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Near-Field Autoencoder — Anomaly Detection & Repair\n"
        "Trained on CST-Augmented Data  (Hybrid Approach)",
        fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: Training loss ──────────────────────────────────────
    ax = axes[0, 0]
    ax.semilogy(loss_history, color="#2196F3", linewidth=1.2)
    ax.set_title("Training Loss (MSE)")
    ax.set_xlabel("Epoch");  ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Raw sweep + anomaly flags ──────────────────────────
    ax = axes[0, 1]
    ax.plot(angles, mag_raw, color="#78909C", lw=0.9, label="Raw sweep")
    ax.scatter(angles[anom_det], mag_raw[anom_det],
               color="#F44336", s=60, zorder=5,
               edgecolors="k", linewidths=0.5, label="Detected anomalies")
    if len(anom_truth):
        ax.scatter(angles[anom_truth], mag_raw[anom_truth],
                   facecolors="none", edgecolors="#4CAF50",
                   s=120, linewidths=2, zorder=4,
                   label="Ground-truth anomalies")
    ax.set_title("Raw Sweep + Anomaly Flags")
    ax.set_xlabel("Angle (°)");  ax.set_ylabel("Magnitude (dB)")
    ax.legend(fontsize=8);       ax.grid(True, alpha=0.3)

    # ── Panel 3: Reconstruction error + threshold ───────────────────
    ax = axes[1, 0]
    ax.fill_between(angles, 0, errors, color="#FF9800", alpha=0.4)
    ax.plot(angles, errors, color="#E65100", lw=0.8)
    ax.axhline(threshold, color="#F44336", ls="--", lw=1.5,
               label=f"Threshold  (μ+3σ = {threshold:.2f} dB)")
    ax.set_title("Point-Wise Reconstruction Error")
    ax.set_xlabel("Angle (°)");  ax.set_ylabel("|Error| (dB)")
    ax.legend(fontsize=8);       ax.grid(True, alpha=0.3)

    # ── Panel 4: Before vs After ────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(angles, mag_raw, color="#BDBDBD", lw=0.8,
            label="Raw (corrupted)", alpha=0.7)
    ax.plot(angles, mag_recon, color="#2196F3", lw=1.0,
            ls="--", label="Autoencoder reconstruction")
    ax.plot(angles, mag_cleaned, color="#4CAF50", lw=1.5,
            label="Cleaned (smart interpolation)")
    ax.set_title("Before vs After Repair")
    ax.set_xlabel("Angle (°)");  ax.set_ylabel("Magnitude (dB)")
    ax.legend(fontsize=8);       ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = "results/nf_anomaly_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [OK] Diagnostic plot saved -> {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 9.  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="NF Autoencoder anomaly detector  (CST hybrid approach)")
    ap.add_argument("--cst", default="data/raw/Simulated_NF_Data.txt",
                    help="Path to CST simulation file  (default: %(default)s)")
    ap.add_argument("--infer", default=None,
                    help="CSV with columns: angle, mag_dB, phase_deg")
    ap.add_argument("--weights", default="models/nf_autoencoder.pth",
                    help="Model weights path  (default: %(default)s)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-plot", action="store_true",
                    help="Suppress matplotlib window")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")

    # ── 1. Load the CST base pattern ────────────────────────────────
    print(f"\n  Loading CST simulation: {args.cst}")
    clean_base = load_cst_simulation(args.cst)

    # ── 2. Build model ──────────────────────────────────────────────
    model = NearFieldAutoencoder()

    # ── 3. Train  (or skip if weights exist and --infer is given) ───
    need_train = (args.infer is None) or (not os.path.exists(args.weights))
    if need_train:
        loss_history = train_model(
            model, clean_base,
            epochs=args.epochs, batch_size=args.batch_size,
            save_path=args.weights, device=device)
    else:
        loss_history = []

    # ── 4. Load weights ─────────────────────────────────────────────
    model.load_state_dict(
        torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"  [OK] Weights loaded <- {args.weights}")

    # ── 5. Prepare input sweep ──────────────────────────────────────
    if args.infer and os.path.exists(args.infer):
        # Real measurement data
        df        = pd.read_csv(args.infer)
        angles    = df.iloc[:, 0].values
        mag_raw   = df.iloc[:, 1].values
        phase_raw = df.iloc[:, 2].values
        mag_clean = phase_clean = None
        anom_truth = np.array([], dtype=int)
    else:
        # Demo mode — inject anomalies into the CST base
        print("\n  >> Demo mode: injecting 8 anomalies into CST base...")
        (angles, mag_raw, phase_raw,
         mag_clean, phase_clean, anom_truth) = \
            generate_raw_sweep_with_anomalies(clean_base)

    # ── 6. Detect anomalies ─────────────────────────────────────────
    anom_idx, errors, mag_recon, threshold = detect_anomalies(
        model, mag_raw, device=device)

    print(f"\n  {'-' * 44}")
    print(f"  Adaptive threshold : {threshold:.3f} dB   (mean + 3*std)")
    print(f"  Anomalies detected : {len(anom_idx)}   "
          f"at angles {anom_idx.tolist()}")

    if len(anom_truth):
        det  = set(anom_idx.tolist())
        tru  = set(anom_truth.tolist())
        print(f"  Ground-truth count : {len(anom_truth)}")
        print(f"  True positives     : {len(det & tru)}")
        print(f"  False positives    : {len(det - tru)}")
        print(f"  Missed             : {len(tru - det)}")
    print(f"  {'-' * 44}")

    # ── 7. Smart interpolation ──────────────────────────────────────
    mag_fix, pha_fix = smart_interpolate(
        angles, mag_raw, phase_raw, anom_idx)

    # ── 8. Save cleaned output ──────────────────────────────────────
    out_csv = "data/processed/cleaned_sweep.csv"
    pd.DataFrame({
        "angle_deg": angles,
        "mag_dB":    mag_fix,
        "phase_deg": pha_fix,
    }).to_csv(out_csv, index=False)
    print(f"\n  [OK] Cleaned data saved -> {out_csv}")

    # ── 9. Plot ─────────────────────────────────────────────────────
    if not args.no_plot:
        plot_results(
            angles, mag_raw, mag_recon, mag_fix,
            errors, threshold, anom_idx, anom_truth,
            loss_history if loss_history else [0])


if __name__ == "__main__":
    main()
