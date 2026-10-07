"""Interactive dashboard:  streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from viral_sim.analysis import critical_completion_rate, network_r0, region_crossing_days
from viral_sim.network import build_social_graph, graph_summary, to_arrays
from viral_sim.simulation import SimConfig, run_sir

st.set_page_config(page_title="Bio-Viral Trend Predictor", page_icon="🦠", layout="wide")
st.title("🦠 Bio-Viral: Trend Predictor & Social Graph Analyzer")
st.caption("An epidemiological SIR model of how a TikTok video spreads through a 5,000-user social graph.")


@st.cache_resource
def load_graph(n_users, n_regions, bridges):
    G = build_social_graph(n_users=n_users, n_regions=n_regions, cross_region_edges=bridges)
    A, regions = to_arrays(G)
    return G, A, regions


with st.sidebar:
    st.header("🎬 Video")
    completion = st.slider("Watch-completion rate", 0.05, 1.0, 0.65, 0.05)
    share = st.slider("Share probability (per friend, full watch)", 0.01, 0.30, 0.10, 0.01)
    decay = st.slider("Daily interest decay", 0.05, 0.9, 0.30, 0.05)
    st.header("🚀 Distribution")
    strategy = st.radio("Who posts it?", ["random", "influencer"], horizontal=True)
    boost = st.slider("For You Page push", 0.0, 3.0, 0.0, 0.25)
    st.header("🌐 Network")
    n_regions = st.slider("Regions", 1, 10, 5)
    bridges = st.slider("Cross-region connections", 0, 1000, 150, 50)
    seed = st.number_input("Random seed", 0, 10_000, 2)

G, A, regions = load_graph(5000, n_regions, bridges)
cfg = SimConfig(completion_rate=completion, base_share_prob=share, interest_decay=decay,
                algo_boost=boost, seed_strategy=strategy, seed=int(seed))
df = run_sir(A, regions, cfg)
n = A.shape[0]

r0 = network_r0(G, cfg.beta, decay)
c_crit = critical_completion_rate(G, share, decay)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Network R₀", f"{r0:.2f}", "above tipping point" if r0 > 1 else "below tipping point")
c2.metric("Final reach", f"{df['reach'].iloc[-1] / n:.1%}")
c3.metric("Peak day", int(df["infectious"].idxmax()))
c4.metric("Theoretical tipping point", f"{c_crit:.0%} completion" if c_crit <= 1 else "unreachable")

left, right = st.columns([3, 2])
with left:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(df["day"], df["infectious"], df["recovered"], df["susceptible"],
                 labels=["Sharing (I)", "Seen, lost interest (R)", "Not seen (S)"],
                 colors=["#EE1D52", "#69C9D0", "#e8e8e8"])
    ax.set(xlabel="Day", ylabel="Users", xlim=(0, cfg.days), ylim=(0, n), title="SIR state over time")
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)
with right:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    region_cols = [c for c in df.columns if c.startswith("region_")]
    ax.plot(df["day"], df[region_cols].values, lw=2)
    ax.set(xlabel="Day", ylabel="Reach", ylim=(0, 1), title="Reach by region")
    ax.legend([f"R{i}" for i in range(len(region_cols))], fontsize=7, ncol=2)
    st.pyplot(fig)

st.subheader("Day each region crossed 5% reach")
st.dataframe(region_crossing_days(df).to_frame("day").T, width="stretch")
with st.expander("Network stats & raw data"):
    st.json(graph_summary(G))
    st.dataframe(df, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False), "simulation.csv")
