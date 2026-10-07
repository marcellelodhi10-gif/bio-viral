"""Analytics: viral thresholds, parameter sweeps, and regional spread timing.

Key theory (network epidemiology, Newman 2002):
  - Transmissibility T = probability an active sharer passes the video to a
    given friend before losing interest.
  - The video goes viral (a "giant outbreak") only if T > T_c, where
        T_c = <k> / (<k^2> - <k>)
    depends ONLY on the network's degree distribution. Heavy-tailed networks
    (influencers!) have a large <k^2>, so their tipping point is very low.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .simulation import SimConfig, run_many


def transmissibility(beta: float, gamma: float) -> float:
    """T for discrete time: sharer attempts each day until recovering w.p. gamma."""
    return 1.0 - gamma * (1.0 - beta) / (1.0 - (1.0 - gamma) * (1.0 - beta))


def critical_transmissibility(G: nx.Graph) -> float:
    k = np.array([d for _, d in G.degree()], dtype=float)
    return k.mean() / ((k ** 2).mean() - k.mean())


def network_r0(G: nx.Graph, beta: float, gamma: float) -> float:
    """Average number of new viewers each sharer creates early on."""
    return transmissibility(beta, gamma) / critical_transmissibility(G)


def critical_completion_rate(G: nx.Graph, base_share_prob: float, gamma: float) -> float:
    """Invert T(beta) = T_c to get the watch-completion rate at the tipping point."""
    t_c = critical_transmissibility(G)
    x = (1 - t_c) / (gamma + (1 - t_c) * (1 - gamma))  # x = 1 - beta_c
    return (1 - x) / base_share_prob


def final_stats(runs: list[pd.DataFrame], n_users: int, viral_cutoff: float = 0.05) -> dict:
    final_reach = np.array([r["reach"].iloc[-1] / n_users for r in runs])
    viral = final_reach > viral_cutoff
    peak_day = [int(r["infectious"].idxmax()) for r, v in zip(runs, viral) if v]
    return {
        "mean_reach": final_reach.mean(),
        "p_viral": viral.mean(),
        "reach_if_viral": final_reach[viral].mean() if viral.any() else 0.0,
        "median_peak_day": float(np.median(peak_day)) if peak_day else np.nan,
    }


def threshold_sweep(G, A, regions, base_cfg: SimConfig, completion_rates, n_runs: int = 30) -> pd.DataFrame:
    rows = []
    for c in completion_rates:
        cfg = SimConfig(**{**base_cfg.__dict__, "completion_rate": float(c)})
        runs = run_many(A, regions, cfg, n_runs, base_seed=1000)
        rows.append({
            "completion_rate": c,
            "R0": network_r0(G, cfg.beta, cfg.interest_decay),
            **final_stats(runs, A.shape[0]),
        })
    return pd.DataFrame(rows)


def region_crossing_days(df: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
    """First day each region's reach exceeds `threshold` (NaN = never crossed)."""
    cols = [c for c in df.columns if c.startswith("region_")]
    out = {}
    for col in cols:
        hit = df.index[df[col] >= threshold]
        out[col.replace("_reach", "")] = int(df.loc[hit[0], "day"]) if len(hit) else np.nan
    return pd.Series(out)
