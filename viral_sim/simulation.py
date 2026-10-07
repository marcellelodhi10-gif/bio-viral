"""Discrete-time SIR contagion on a network, vectorised with NumPy/SciPy.

Epidemiology  ->  TikTok
  Susceptible  ->  hasn't seen the video
  Infectious   ->  has seen it and is actively sharing / dueting / stitching
  Recovered    ->  has seen it but lost interest (no longer spreads it)

Each day, every Susceptible user with k Infectious friends gets the video with
probability 1 - (1 - beta)^k, where beta = base_share_prob * completion_rate.
Each Infectious user loses interest with probability ``interest_decay``.

Optional ``algo_boost`` models the For You Page: content is pushed to users
who are NOT connected to any sharer, in proportion to how hot it currently is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

S, I, R = 0, 1, 2


@dataclass
class SimConfig:
    completion_rate: float = 0.5   # avg fraction of the video watched (0-1)
    base_share_prob: float = 0.10  # P(share to a given friend | watched fully)
    interest_decay: float = 0.3    # P(an active sharer stops sharing) per day
    algo_boost: float = 0.0        # For You Page push strength (0 = social graph only)
    days: int = 90
    n_seeds: int = 1
    seed_strategy: str = "random"  # "random" or "influencer"
    seed: int | None = 0

    @property
    def beta(self) -> float:
        return float(np.clip(self.base_share_prob * self.completion_rate, 0, 1))


def choose_seeds(A: sp.csr_matrix, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    n = A.shape[0]
    if cfg.seed_strategy == "influencer":
        degree = np.asarray(A.sum(axis=1)).ravel()
        return np.argsort(degree)[-cfg.n_seeds:]
    return rng.choice(n, size=cfg.n_seeds, replace=False)


def run_sir(A: sp.csr_matrix, regions: np.ndarray, cfg: SimConfig) -> pd.DataFrame:
    """Run one simulation; returns one row per day."""
    rng = np.random.default_rng(cfg.seed)
    n = A.shape[0]
    n_regions = int(regions.max()) + 1
    region_sizes = np.bincount(regions, minlength=n_regions)

    state = np.full(n, S, dtype=np.int8)
    state[choose_seeds(A, cfg, rng)] = I

    rows = []
    prev_reach = 0
    for day in range(cfg.days + 1):
        infectious = state == I
        seen = state != S
        reach = int(seen.sum())
        region_reach = np.bincount(regions[seen], minlength=n_regions) / region_sizes
        rows.append({
            "day": day,
            "susceptible": int((state == S).sum()),
            "infectious": int(infectious.sum()),
            "recovered": int((state == R).sum()),
            "reach": reach,
            "new_views": reach - prev_reach,
            **{f"region_{r}_reach": float(region_reach[r]) for r in range(n_regions)},
        })
        prev_reach = reach
        if day == cfg.days:
            break

        # Social spread: count infectious neighbours of every node in one sparse mat-vec.
        k = A @ infectious.astype(np.float32)
        p_social = 1.0 - (1.0 - cfg.beta) ** k
        # Algorithmic spread: FYP exposure scales with how many people are sharing now.
        p_algo = cfg.algo_boost * cfg.completion_rate * infectious.mean()
        p_exposed = 1.0 - (1.0 - p_social) * (1.0 - p_algo)

        newly_infected = (state == S) & (rng.random(n) < p_exposed)
        newly_recovered = infectious & (rng.random(n) < cfg.interest_decay)
        state[newly_infected] = I
        state[newly_recovered] = R

    return pd.DataFrame(rows)


def run_many(A, regions, cfg: SimConfig, n_runs: int, base_seed: int = 0) -> list[pd.DataFrame]:
    """Monte Carlo: repeat a simulation with different random seeds."""
    out = []
    for i in range(n_runs):
        run_cfg = SimConfig(**{**cfg.__dict__, "seed": base_seed + i})
        out.append(run_sir(A, regions, run_cfg))
    return out
