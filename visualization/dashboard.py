"""
dashboard.py — Interactive Materials Prediction Dashboard
Streamlit-based UI for exploring material datasets and GNN predictions.
"""

import json
import math
import random
from pathlib import Path

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("[ERROR] Run: pip install streamlit plotly pandas numpy")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Page config ─────────────────────────────────────────────────────────────

if DEPS_AVAILABLE:
    st.set_page_config(
        page_title="Nano-AI Materials Predictor",
        page_icon="⚛️",
        layout="wide",
    )


def load_data(path: str = "data/sample_materials.json") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        # Generate on-the-fly if not present
        sys.path.insert(0, "data")
        from fetcher import generate_synthetic_samples
        generate_synthetic_samples(200)

    with open(p) as f:
        raw = json.load(f)
    materials = raw.get("materials", raw) if isinstance(raw, dict) else raw
    return pd.DataFrame(materials)


def plot_bandgap_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df.dropna(subset=["bandgap_ev"]),
        x="bandgap_ev",
        nbins=40,
        title="Bandgap Distribution (eV)",
        color_discrete_sequence=["#58a6ff"],
        labels={"bandgap_ev": "Bandgap (eV)"},
    )
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
        title_font_size=16,
    )
    return fig


def plot_formation_energy_vs_bandgap(df: pd.DataFrame) -> go.Figure:
    df_clean = df.dropna(subset=["bandgap_ev", "formation_energy_ev_atom"])
    fig = px.scatter(
        df_clean,
        x="bandgap_ev",
        y="formation_energy_ev_atom",
        color="is_stable",
        hover_data=["formula", "nsites"],
        title="Formation Energy vs Bandgap",
        labels={
            "bandgap_ev": "Bandgap (eV)",
            "formation_energy_ev_atom": "Formation Energy (eV/atom)",
        },
        color_discrete_map={True: "#3fb950", False: "#f85149"},
    )
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
    )
    return fig


def plot_elements_bar(df: pd.DataFrame) -> go.Figure:
    element_counts = {}
    for elems in df["elements"].dropna():
        if isinstance(elems, list):
            for el in elems:
                element_counts[el] = element_counts.get(el, 0) + 1

    sorted_elems = sorted(element_counts.items(), key=lambda x: -x[1])[:20]
    labels, counts = zip(*sorted_elems) if sorted_elems else ([], [])

    fig = go.Figure(go.Bar(
        x=list(labels),
        y=list(counts),
        marker_color="#58a6ff",
    ))
    fig.update_layout(
        title="Most Common Elements in Dataset",
        xaxis_title="Element",
        yaxis_title="Count",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
    )
    return fig


def plot_parity(predictions: list, actuals: list, target: str) -> go.Figure:
    min_val = min(min(predictions), min(actuals))
    max_val = max(max(predictions), max(actuals))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actuals, y=predictions,
        mode="markers",
        marker=dict(color="#58a6ff", size=6, opacity=0.7),
        name="Materials",
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color="#f85149", dash="dash"),
        name="Ideal (y=x)",
    ))
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    fig.update_layout(
        title=f"Predicted vs Actual {target} | MAE: {mae:.4f}",
        xaxis_title=f"Actual {target}",
        yaxis_title=f"Predicted {target}",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
    )
    return fig


def mock_predictions(df: pd.DataFrame, target: str = "bandgap_ev") -> tuple:
    """Generate mock predictions for demo (replace with real model inference)."""
    actuals = df[target].dropna().tolist()[:50]
    predictions = [v + random.gauss(0, 0.3) for v in actuals]
    return predictions, actuals


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    if not DEPS_AVAILABLE:
        print("Please install: pip install streamlit plotly pandas numpy")
        return

    # Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #161b22, #1c2333);
                padding: 32px; border-radius: 12px; margin-bottom: 24px;
                border: 1px solid #30363d;'>
        <h1 style='color: #58a6ff; margin: 0;'>⚛️ Nano-AI Materials Predictor</h1>
        <p style='color: #8b949e; margin-top: 8px;'>
            GNN-powered bandgap, formation energy & property prediction
            from crystal structure graphs.
        </p>
        <p style='color: #8b949e; font-size: 12px;'>
            UMBA YANGA IVON EXAUCE | UMBA Consulting Engineers
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Controls")
    data_path = st.sidebar.text_input("Data path", value="data/sample_materials.json")
    target = st.sidebar.selectbox(
        "Prediction target",
        ["bandgap_ev", "formation_energy_ev_atom"],
        index=0,
    )
    show_stable_only = st.sidebar.checkbox("Show stable materials only", value=False)

    # Load data
    try:
        df = load_data(data_path)
        if show_stable_only and "is_stable" in df.columns:
            df = df[df["is_stable"] == True]
        st.sidebar.success(f"✅ {len(df)} materials loaded")
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return

    # ── Row 1: Stats ──────────────────────────────────────────────────────────
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Materials", len(df))
    with col2:
        if "bandgap_ev" in df:
            st.metric("Avg Bandgap", f"{df['bandgap_ev'].mean():.2f} eV")
    with col3:
        if "formation_energy_ev_atom" in df:
            st.metric("Avg Formation Energy", f"{df['formation_energy_ev_atom'].mean():.2f} eV/atom")
    with col4:
        if "is_stable" in df:
            st.metric("Stable Materials", int(df["is_stable"].sum()))

    # ── Row 2: Distribution plots ─────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if "bandgap_ev" in df.columns:
            st.plotly_chart(plot_bandgap_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_elements_bar(df), use_container_width=True)

    # ── Row 3: Scatter ────────────────────────────────────────────────────────
    if "bandgap_ev" in df.columns and "formation_energy_ev_atom" in df.columns:
        st.plotly_chart(plot_formation_energy_vs_bandgap(df), use_container_width=True)

    # ── Row 4: Parity plot (mock predictions) ─────────────────────────────────
    st.subheader("🎯 GNN Predictions (Demo Mode)")
    st.info("ℹ️ Run `python prediction/trainer.py` to train the model. Showing mock predictions below.")

    if target in df.columns and df[target].notna().sum() >= 10:
        preds, actuals = mock_predictions(df, target)
        st.plotly_chart(plot_parity(preds, actuals, target), use_container_width=True)

    # ── Row 5: Data table ─────────────────────────────────────────────────────
    st.subheader("🔬 Material Records")
    display_cols = [c for c in ["formula", "material_id", "bandgap_ev",
                                 "formation_energy_ev_atom", "nsites",
                                 "is_stable", "density"] if c in df.columns]
    st.dataframe(
        df[display_cols].head(100),
        use_container_width=True,
        hide_index=True,
    )

    # Footer
    st.markdown("""
    <hr style='border-color: #30363d;'>
    <p style='text-align:center; color:#8b949e; font-size:12px;'>
        Nano-AI Materials Predictor | UMBA Consulting Engineers |
        umbaconsulting.com
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
