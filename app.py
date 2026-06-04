"""
NF-FF Antenna Metrology Dashboard
==================================
Streamlit-based lab benchtop interface for the automated near-field
to far-field measurement pipeline.

Launch
------
    streamlit run app.py

Author : NF Metrology Pipeline
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Resolve imports ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src" / "python"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import MetrologyPipeline, PipelineConfig, PipelineStage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ═══════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG & GLOBAL STYLES
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NF-FF Antenna Metrology",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* ── Import premium font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0f172a 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #151d2e 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(59,130,246,0.06) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99,102,241,0.5);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 500;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700;
        font-size: 1.6rem;
    }

    /* ── Headers ── */
    .main-header {
        background: linear-gradient(90deg, rgba(99,102,241,0.12) 0%, transparent 100%);
        border-left: 3px solid #6366f1;
        padding: 12px 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e2e8f0, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        margin: 4px 0 0 0;
        color: #64748b;
        font-size: 0.85rem;
    }

    /* ── Stage status badges ── */
    .stage-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .stage-done {
        background: rgba(34,197,94,0.15);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .stage-running {
        background: rgba(99,102,241,0.15);
        color: #818cf8;
        border: 1px solid rgba(99,102,241,0.3);
    }
    .stage-pending {
        background: rgba(100,116,139,0.1);
        color: #64748b;
        border: 1px solid rgba(100,116,139,0.2);
    }

    /* ── Tabs ── */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(99,102,241,0.15) !important;
    }

    /* ── Success / info boxes ── */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 2.  SIDEBAR — File Upload & Configuration
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📡 NF-FF Metrology")
    st.caption("Automated Antenna Measurement Pipeline")
    st.divider()

    # File uploader
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload VNA / CST measurement file",
        type=["csv", "txt"],
        help="Accepts CST planar exports (.txt) or VNA CSV files (.csv)",
    )

    use_demo = st.checkbox(
        "🎯 Use demo data (CST simulation)",
        value=uploaded_file is None,
        help="Process the included Simulated_NF_Data.txt",
    )

    st.divider()

    # Configuration
    st.markdown("### ⚙️ Configuration")
    freq_ghz = st.number_input(
        "Frequency (GHz)", min_value=0.1, max_value=100.0,
        value=10.0, step=0.5, format="%.1f",
    )
    step_mm = st.number_input(
        "Step Size (mm)", min_value=0.1, max_value=100.0,
        value=10.0, step=1.0, format="%.1f",
    )
    r_probe_m = st.number_input(
        "Probe Radius (m)", min_value=0.01, max_value=10.0,
        value=0.5, step=0.05, format="%.2f",
        help="Measurement radius for circular scans (used by CME transform)",
    )
    n_fft = st.select_slider(
        "FFT Size", options=[64, 128, 256, 512, 1024],
        value=256,
    )

    st.divider()

    # Run button
    run_clicked = st.button(
        "🚀  Run Full Pipeline",
        use_container_width=True,
        type="primary",
    )

    # Sidebar footer
    st.divider()
    st.caption(
        "Built for near-field antenna measurement research at 10 GHz.\n\n"
        "Pipeline: Ingestion → Autoencoder → LightGBM → PWS FFT"
    )


# ═══════════════════════════════════════════════════════════════════
# 3.  MAIN CANVAS — Header
# ═══════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="main-header">'
    "<h1>Near-Field → Far-Field Dashboard</h1>"
    "<p>Automated anomaly detection, phase calibration, and "
    "Plane-Wave Spectrum transformation</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
# 4.  PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════

PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=40, r=40, t=50, b=40),
)


def make_3d_radiation_pattern(ff: dict) -> go.Figure:
    """Build an interactive 3D far-field radiation pattern surface."""
    U = ff["U_grid"]
    V = ff["V_grid"]
    FF_dB = ff["farfield_Ey_dB"].copy()
    visible = ff["visible_region"]

    # Mask evanescent region
    FF_dB[~visible] = np.nan

    fig = go.Figure(data=[
        go.Surface(
            x=U, y=V, z=FF_dB,
            colorscale="Plasma",
            cmin=-40, cmax=0,
            colorbar=dict(
                title=dict(text="dB", font=dict(size=13)),
                thickness=15,
                len=0.7,
                tickfont=dict(size=11),
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=True,
                       highlightcolor="#6366f1", highlightwidth=1),
            ),
            lighting=dict(
                ambient=0.6, diffuse=0.5,
                specular=0.3, roughness=0.5,
            ),
            hovertemplate=(
                "U: %{x:.3f}<br>V: %{y:.3f}<br>"
                "FF: %{z:.1f} dB<extra></extra>"
            ),
        ),
    ])

    # Visible-region circle
    t = np.linspace(0, 2 * np.pi, 361)
    fig.add_trace(go.Scatter3d(
        x=np.cos(t), y=np.sin(t),
        z=np.full_like(t, -40),
        mode="lines",
        line=dict(color="#6366f1", width=3),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title=dict(
            text="3D Far-Field Radiation Pattern  |Ey|",
            font=dict(size=16, color="#e2e8f0"),
        ),
        scene=dict(
            xaxis=dict(title="U = sin θ cos φ", showgrid=True,
                       gridcolor="rgba(99,102,241,0.1)"),
            yaxis=dict(title="V = sin θ sin φ", showgrid=True,
                       gridcolor="rgba(99,102,241,0.1)"),
            zaxis=dict(title="Amplitude (dB)", range=[-50, 5],
                       showgrid=True,
                       gridcolor="rgba(99,102,241,0.1)"),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=0.9),
            ),
        ),
        height=600,
    )
    return fig


def make_uv_heatmap(ff: dict) -> go.Figure:
    """Build a 2-D U-V far-field heatmap."""
    FF_dB = ff["farfield_Ey_dB"].copy()
    visible = ff["visible_region"]
    FF_dB[~visible] = np.nan

    fig = go.Figure(data=[
        go.Heatmap(
            z=FF_dB,
            x=ff["u"], y=ff["v"],
            colorscale="Plasma",
            zmin=-40, zmax=0,
            colorbar=dict(title="dB", thickness=15),
            hovertemplate="U: %{x:.3f}<br>V: %{y:.3f}<br>%{z:.1f} dB<extra></extra>",
        ),
    ])

    # Visible-region circle
    t = np.linspace(0, 2 * np.pi, 361)
    fig.add_trace(go.Scatter(
        x=np.cos(t), y=np.sin(t),
        mode="lines",
        line=dict(color="#f8fafc", width=1.5, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="Far-Field |Ey|  — U-V Space",
        xaxis=dict(title="U", scaleanchor="y"),
        yaxis=dict(title="V"),
        height=520,
    )
    return fig


def make_theta_cuts(ff: dict) -> go.Figure:
    """Build principal-plane theta cuts at phi=0° and phi=90°."""
    n = ff["n_fft"]
    mid = n // 2
    FF_dB = ff["farfield_Ey_dB"]
    visible = ff["visible_region"]

    theta_grid = ff["theta"]
    U_grid = ff["U_grid"]
    V_grid = ff["V_grid"]

    # phi=0 cut (row at V≈0)
    theta_phi0 = np.degrees(theta_grid[mid, :])
    theta_signed_0 = np.where(U_grid[mid, :] >= 0, theta_phi0, -theta_phi0)
    mask0 = visible[mid, :]

    # phi=90 cut (column at U≈0)
    theta_phi90 = np.degrees(theta_grid[:, mid])
    theta_signed_90 = np.where(V_grid[:, mid] >= 0, theta_phi90, -theta_phi90)
    mask90 = visible[:, mid]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theta_signed_0[mask0], y=FF_dB[mid, mask0],
        mode="lines", name="φ = 0°",
        line=dict(color="#818cf8", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=theta_signed_90[mask90], y=FF_dB[mask90, mid],
        mode="lines", name="φ = 90°",
        line=dict(color="#f472b6", width=2.5, dash="dash"),
    ))

    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="Principal-Plane Cuts  |Ey|",
        xaxis=dict(title="θ (degrees)"),
        yaxis=dict(title="Normalised Amplitude (dB)", range=[-50, 5]),
        legend=dict(x=0.02, y=0.02, bgcolor="rgba(0,0,0,0.3)"),
        height=420,
    )
    return fig


def make_anomaly_plot(anom: dict) -> go.Figure:
    """Build anomaly detection diagnostic overlay."""
    angles = anom["angles"]
    mag_raw = anom["mag_raw"]
    mag_clean = anom["mag_clean"]
    recon = anom["reconstructed"]
    anom_idx = anom["anomaly_indices"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=angles, y=mag_raw,
        mode="lines", name="Raw sweep",
        line=dict(color="#64748b", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=angles, y=recon,
        mode="lines", name="Autoencoder output",
        line=dict(color="#3b82f6", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=angles, y=mag_clean,
        mode="lines", name="Cleaned",
        line=dict(color="#4ade80", width=2),
    ))
    if len(anom_idx) > 0:
        fig.add_trace(go.Scatter(
            x=angles[anom_idx], y=mag_raw[anom_idx],
            mode="markers", name="Anomalies",
            marker=dict(color="#ef4444", size=9, symbol="x",
                        line=dict(width=1.5, color="#fca5a5")),
        ))

    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="Anomaly Detection — Before vs After Repair",
        xaxis=dict(title="Angle (°)"),
        yaxis=dict(title="Magnitude (dB)"),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0.4)"),
        height=400,
    )
    return fig


def make_error_plot(anom: dict) -> go.Figure:
    """Build per-point reconstruction error chart."""
    angles = anom["angles"]
    errors = anom["errors"]
    threshold = anom["threshold"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=angles, y=errors,
        fill="tozeroy", mode="lines", name="|Error|",
        line=dict(color="#f97316", width=1),
        fillcolor="rgba(249,115,22,0.2)",
    ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#ef4444",
        annotation_text=f"μ+3σ = {threshold:.2f} dB",
        annotation_font_color="#fca5a5",
    )

    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="Per-Point Reconstruction Error",
        xaxis=dict(title="Angle (°)"),
        yaxis=dict(title="|Error| (dB)"),
        height=350,
    )
    return fig


# ── Circular far-field plot helpers ──────────────────────────────

def make_circular_polar(ff: dict) -> go.Figure:
    """Interactive polar plot of NF vs FF for circular transforms."""
    angles = ff["angles_deg"]
    theta = np.deg2rad(angles)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        theta=angles, r=ff["nf_mag_dB"],
        mode="lines", name="Near-Field",
        line=dict(color="#3b82f6", width=1.5),
    ))
    fig.add_trace(go.Scatterpolar(
        theta=angles, r=ff["ff_mag_dB"],
        mode="lines", name="Far-Field",
        line=dict(color="#f97316", width=2.5),
    ))

    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="Polar Radiation Pattern — NF vs FF",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[-40, 5], dtick=10,
                gridcolor="rgba(99,102,241,0.15)",
                color="#94a3b8",
            ),
            angularaxis=dict(
                gridcolor="rgba(99,102,241,0.15)",
                color="#94a3b8",
                dtick=30,
            ),
        ),
        legend=dict(x=0.02, y=1.0, bgcolor="rgba(0,0,0,0.3)"),
        height=600,
    )
    return fig


def make_circular_cartesian(ff: dict) -> go.Figure:
    """Cartesian overlay of NF and FF patterns."""
    angles = ff["angles_deg"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=angles, y=ff["nf_mag_dB"],
        mode="lines", name="Near-Field (measured)",
        line=dict(color="#3b82f6", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=angles, y=ff["ff_mag_dB"],
        mode="lines", name="Far-Field (CME transform)",
        line=dict(color="#f97316", width=2.5),
    ))

    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="NF vs FF — Cartesian Overlay",
        xaxis=dict(title="Azimuth (°)", dtick=30),
        yaxis=dict(title="Normalised Amplitude (dB)", range=[-50, 5]),
        legend=dict(x=0.02, y=0.02, bgcolor="rgba(0,0,0,0.3)"),
        height=450,
    )
    return fig


def make_mode_spectrum(ff: dict) -> go.Figure:
    """Cylindrical-mode spectrum before and after Hankel compensation."""
    n = ff["mode_indices"]
    nf_modes_dB = 20.0 * np.log10(np.abs(ff["modes_NF"]) + 1e-15)
    ff_modes_dB = 20.0 * np.log10(np.abs(ff["modes_FF"]) + 1e-15)
    max_mode = ff["max_mode"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n, y=nf_modes_dB,
        mode="lines", name="NF Modes",
        line=dict(color="#3b82f6", width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=n, y=ff_modes_dB,
        mode="lines", name="FF Modes (compensated)",
        line=dict(color="#f97316", width=2),
    ))
    # Mark evanescent cutoff
    fig.add_vrect(
        x0=-max_mode, x1=max_mode,
        fillcolor="rgba(99,102,241,0.08)",
        line_width=0,
        annotation_text=f"Propagating (|n| ≤ {max_mode})",
        annotation_position="top left",
        annotation_font_color="#818cf8",
    )

    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="Cylindrical-Mode Spectrum",
        xaxis=dict(title="Mode index n"),
        yaxis=dict(title="|Mode| (dB)", range=[-80, 0]),
        legend=dict(x=0.02, y=0.02, bgcolor="rgba(0,0,0,0.3)"),
        height=420,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# 5.  PIPELINE EXECUTION & STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None


def execute_pipeline(source, freq, step, nfft):
    """Run the pipeline with progress tracking."""
    config = PipelineConfig(
        freq_ghz=freq,
        step_mm=step,
        n_fft=nfft,
        r_probe_m=r_probe_m,
    )
    pipe = MetrologyPipeline(config)

    # Progress tracking widgets
    stage_labels = {
        PipelineStage.INGESTION: "Stage 1/4 — Ingesting Data",
        PipelineStage.ANOMALY: "Stage 2/4 — Scrubbing Anomalies",
        PipelineStage.CALIBRATION: "Stage 3/4 — Calibrating Phase",
        PipelineStage.TRANSFORM: "Stage 4/4 — Computing NF-FF Transform",
    }

    progress_bar = st.progress(0, text="Initialising pipeline …")
    status_text = st.empty()

    stage_order = list(PipelineStage)

    def progress_callback(stage: PipelineStage, frac: float, msg: str):
        idx = stage_order.index(stage)
        overall = (idx + frac) / len(stage_order)
        label = stage_labels.get(stage, str(stage))
        progress_bar.progress(
            min(overall, 1.0),
            text=f"**{label}** — {msg}" if msg else f"**{label}**",
        )

    with st.spinner("Running full metrology pipeline …"):
        results = pipe.run_full_sweep(source, progress_cb=progress_callback)

    progress_bar.progress(1.0, text="✅ **Pipeline complete**")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    return results


# ═══════════════════════════════════════════════════════════════════
# 6.  RUN & RENDER
# ═══════════════════════════════════════════════════════════════════

if run_clicked:
    # Determine data source
    if uploaded_file is not None:
        source = uploaded_file
    elif use_demo:
        demo_path = ROOT_DIR / "data" / "raw" / "Simulated_NF_Data.txt"
        if not demo_path.is_file():
            st.error(f"Demo file not found: `{demo_path}`")
            st.stop()
        source = str(demo_path)
    else:
        st.warning("Please upload a file or enable demo mode.")
        st.stop()

    try:
        st.session_state.results = execute_pipeline(
            source, freq_ghz, step_mm, n_fft
        )
    except ValueError as exc:
        st.error(f"**Pipeline Error:** {exc}")
        st.stop()

# ── Render results ───────────────────────────────────────────────

results = st.session_state.results

if results is None:
    # Empty state
    st.markdown(
        """
        <div style="text-align: center; padding: 80px 20px; color: #475569;">
            <p style="font-size: 3rem; margin-bottom: 0.5rem;">📡</p>
            <h3 style="color: #94a3b8; font-weight: 500;">
                Ready to Process
            </h3>
            <p style="max-width: 420px; margin: 0 auto; line-height: 1.6;">
                Upload a VNA measurement file or enable demo mode,
                then click <strong>Run Full Pipeline</strong> to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ── METRICS ROW ──────────────────────────────────────────────────

st.markdown("### 📊 Pipeline Metrics")

m = results["metrics"]
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Format Detected", m.get("input_format", "—").upper())
with col2:
    st.metric("Input Points", f"{m.get('input_rows', 0):,}")
with col3:
    n_anom = m.get("anomalies_detected", 0)
    st.metric("Anomalies Fixed", str(n_anom),
              delta=f"-{n_anom} spikes" if n_anom > 0 else "Clean ✓",
              delta_color="inverse" if n_anom > 0 else "off")
with col4:
    drift = m.get("phase_drift_corrected_deg", 0.0)
    st.metric("Phase Drift", f"{drift:.3f}°",
              delta=f"-{drift:.3f}° corrected" if drift > 0 else "—",
              delta_color="inverse" if drift > 0 else "off")
with col5:
    st.metric("Peak Gain", f"{m.get('peak_gain_dB', 0.0):.1f} dB")

st.divider()

# ── TIMING ROW ───────────────────────────────────────────────────

timings = results.get("timings", {})
if timings:
    total = sum(timings.values())
    cols = st.columns(len(timings) + 1)
    for i, (stage, t) in enumerate(timings.items()):
        with cols[i]:
            st.caption(f"⏱ {stage.capitalize()}")
            st.code(f"{t:.2f}s")
    with cols[-1]:
        st.caption("⏱ Total")
        st.code(f"{total:.2f}s")

st.divider()

# ── FAR-FIELD VISUALISATION ──────────────────────────────────────

ff = results["farfield"]
transform_mode = ff.get("transform_mode", "planar")

if transform_mode == "planar":
    st.markdown("### 🛰️ Far-Field Radiation Pattern (Planar PWS)")

    tab_3d, tab_uv, tab_cuts = st.tabs([
        "🌐 3D Surface", "🗺️ U-V Heatmap", "📈 Theta Cuts"
    ])

    with tab_3d:
        fig_3d = make_3d_radiation_pattern(ff)
        st.plotly_chart(fig_3d, use_container_width=True, key="ff_3d")

    with tab_uv:
        fig_uv = make_uv_heatmap(ff)
        st.plotly_chart(fig_uv, use_container_width=True, key="ff_uv")

    with tab_cuts:
        fig_cuts = make_theta_cuts(ff)
        st.plotly_chart(fig_cuts, use_container_width=True, key="ff_cuts")

else:
    st.markdown("### 🛰️ Far-Field Radiation Pattern (Circular CME)")
    st.caption(
        f"Cylindrical-mode expansion with Hankel compensation  |  "
        f"Max propagating mode: **{ff.get('max_mode', '—')}**"
    )

    tab_polar, tab_cart, tab_modes = st.tabs([
        "🎯 Polar Plot", "📈 Cartesian Overlay", "🔬 Mode Spectrum"
    ])

    with tab_polar:
        fig_polar = make_circular_polar(ff)
        st.plotly_chart(fig_polar, use_container_width=True, key="ff_polar")

    with tab_cart:
        fig_cart = make_circular_cartesian(ff)
        st.plotly_chart(fig_cart, use_container_width=True, key="ff_cart")

    with tab_modes:
        fig_modes = make_mode_spectrum(ff)
        st.plotly_chart(fig_modes, use_container_width=True, key="ff_modes")

st.divider()

# ── ANOMALY DETECTION DIAGNOSTICS ───────────────────────────────

st.markdown("### 🔍 Anomaly Detection Diagnostics")

anom = results["anomaly"]

col_a, col_b = st.columns(2)
with col_a:
    fig_anom = make_anomaly_plot(anom)
    st.plotly_chart(fig_anom, use_container_width=True, key="anom_plot")
with col_b:
    fig_err = make_error_plot(anom)
    st.plotly_chart(fig_err, use_container_width=True, key="err_plot")

# ── STAGE STATUS ────────────────────────────────────────────────

st.divider()

with st.expander("📋 Detailed Stage Status", expanded=False):
    stages = results["stages_completed"]
    all_stages = ["ingestion", "anomaly", "calibration", "transform"]
    labels = {
        "ingestion": "1. Data Ingestion",
        "anomaly": "2. Anomaly Detection (PyTorch Autoencoder)",
        "calibration": "3. Phase Calibration (LightGBM)",
        "transform": f"4. NF-FF Transform ({transform_mode.upper()})",
    }
    for s in all_stages:
        if s in stages:
            st.markdown(
                f'<span class="stage-badge stage-done">✓ DONE</span> '
                f"&nbsp; **{labels[s]}**  "
                f"({timings.get(s, 0):.2f}s)",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="stage-badge stage-pending">— SKIP</span> '
                f"&nbsp; {labels[s]}",
                unsafe_allow_html=True,
            )

    # Raw metrics dump
    st.markdown("#### All Metrics")
    st.json(m)
