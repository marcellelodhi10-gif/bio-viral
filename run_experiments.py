"""Run every experiment and save figures to ./figures and data to ./results.

    python run_experiments.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from viral_sim.analysis import (critical_completion_rate, final_stats,
                                region_crossing_days, threshold_sweep)
from viral_sim.network import build_social_graph, graph_summary, to_arrays
from viral_sim.simulation import SimConfig, run_many, run_sir

FIG, RES = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
N_RUNS = 40


def pick_viral_run(A, regions, cfg, n_users):
    """First seed whose run actually goes viral (for clean single-run plots)."""
    for s in range(200):
        df = run_sir(A, regions, SimConfig(**{**cfg.__dict__, "seed": s}))
        if df["reach"].iloc[-1] > 0.05 * n_users:
            return df
    return df


def main():
    G = build_social_graph()
    A, regions = to_arrays(G)
    n = A.shape[0]
    base = SimConfig()
    summary = graph_summary(G)
    print("Graph:", summary)

    # 1. Viral curves at three watch-completion rates ------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for c, color in [(0.45, "#69C9D0"), (0.65, "#EE1D52"), (0.85, "#010101")]:
        df = pick_viral_run(A, regions, SimConfig(completion_rate=c), n)
        ax.plot(df["day"], df["infectious"], color=color, lw=2.5, label=f"Active sharers ({c:.0%} completion)")
        ax.plot(df["day"], df["reach"], color=color, lw=1.5, ls="--", label=f"Total reach ({c:.0%})")
    ax.set(title="How watch-completion rate shapes a viral curve", xlabel="Day", ylabel="Users")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "1_viral_curves.png", dpi=150)

    # 2. Tipping-point sweep: theory vs simulation ----------------------------
    rates = np.round(np.arange(0.1, 1.01, 0.05), 2)
    sweep = threshold_sweep(G, A, regions, base, rates, n_runs=N_RUNS)
    sweep.to_csv(RES / "threshold_sweep.csv", index=False)
    c_theory = critical_completion_rate(G, base.base_share_prob, base.interest_decay)
    c_sim = sweep.loc[sweep["p_viral"] >= 0.1, "completion_rate"].min()

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(sweep["completion_rate"], sweep["mean_reach"], "o-", color="#EE1D52", lw=2.5, label="Avg final reach")
    ax1.plot(sweep["completion_rate"], sweep["p_viral"], "s--", color="#69C9D0", lw=2, label="P(goes viral)")
    ax1.axvline(c_theory, color="gray", ls=":", lw=2, label=f"Mean-field theory tipping point ({c_theory:.0%})")
    ax1.axvline(c_sim, color="black", ls="-", lw=1.5, label=f"Observed tipping point ({c_sim:.0%})")
    ax1.set(title="Phase transition: the watch-completion tipping point",
            xlabel="Average watch-completion rate", ylabel="Fraction of users / probability")
    ax1.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "2_tipping_point.png", dpi=150)

    # 3. Regional spread heatmap ---------------------------------------------
    df = pick_viral_run(A, regions, SimConfig(completion_rate=0.75), n)
    region_cols = [c for c in df.columns if c.startswith("region_")]
    heat = df.set_index("day")[region_cols].T
    heat.index = [f"Region {i}" for i in range(len(region_cols))]
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(heat.iloc[:, :70], cmap="rocket_r", vmin=0, vmax=1, ax=ax, xticklabels=10, cbar_kws={"label": "Reach"})
    ax.set(title="A trend crossing regional clusters (75% completion)", xlabel="Day")
    fig.tight_layout()
    fig.savefig(FIG / "3_regional_spread.png", dpi=150)
    crossings = region_crossing_days(df, threshold=0.05)
    crossings.to_csv(RES / "region_crossing_days.csv")

    # 4. Seeding strategy x For You Page boost --------------------------------
    rows = []
    for strategy in ["random", "influencer"]:
        for boost in [0.0, 1.0]:
            cfg = SimConfig(completion_rate=0.5, seed_strategy=strategy, algo_boost=boost)
            stats = final_stats(run_many(A, regions, cfg, N_RUNS, base_seed=500), n)
            rows.append({"seed": strategy, "fyp_boost": boost, **stats})
    strat = pd.DataFrame(rows)
    strat.to_csv(RES / "seeding_strategy.csv", index=False)
    strat["scenario"] = strat["seed"].str.title() + " creator\n" + np.where(strat["fyp_boost"] > 0, "+ FYP push", "social graph only")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=strat, x="scenario", y="mean_reach", hue="scenario", legend=False, palette=["#69C9D0", "#69C9D0", "#EE1D52", "#EE1D52"], ax=ax)
    for p, pv in zip(ax.patches, strat["p_viral"]):
        ax.annotate(f"P(viral)={pv:.0%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set(title="Same video (50% completion), different distribution", xlabel="", ylabel="Avg final reach")
    fig.tight_layout()
    fig.savefig(FIG / "4_seeding_strategy.png", dpi=150)

    # Console summary ----------------------------------------------------------
    print(f"\nTheory tipping point: {c_theory:.0%}   Observed: {c_sim:.0%}")
    print("\nSweep:\n", sweep.round(3).to_string(index=False))
    print("\nRegion crossing days (5% reach):\n", crossings.to_string())
    print("\nSeeding:\n", strat.drop(columns="scenario").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
