import numpy as np
import pytest

from viral_sim.analysis import critical_transmissibility, transmissibility
from viral_sim.network import build_social_graph, to_arrays
from viral_sim.simulation import SimConfig, run_sir


@pytest.fixture(scope="module")
def small_graph():
    G = build_social_graph(n_users=600, n_regions=3, cross_region_edges=30, seed=1)
    return G, *to_arrays(G)


def test_graph_shape(small_graph):
    G, A, regions = small_graph
    assert G.number_of_nodes() == 600
    assert set(regions) == {0, 1, 2}
    assert (A != A.T).nnz == 0  # undirected


def test_population_is_conserved(small_graph):
    _, A, regions = small_graph
    df = run_sir(A, regions, SimConfig(completion_rate=0.9, days=30))
    assert (df[["susceptible", "infectious", "recovered"]].sum(axis=1) == 600).all()


def test_reach_never_decreases(small_graph):
    _, A, regions = small_graph
    df = run_sir(A, regions, SimConfig(completion_rate=0.9, days=30))
    assert (np.diff(df["reach"]) >= 0).all()


def test_zero_completion_never_spreads(small_graph):
    _, A, regions = small_graph
    df = run_sir(A, regions, SimConfig(completion_rate=0.0, days=20))
    assert df["reach"].iloc[-1] == 1


def test_deterministic_with_seed(small_graph):
    _, A, regions = small_graph
    a = run_sir(A, regions, SimConfig(seed=3))
    b = run_sir(A, regions, SimConfig(seed=3))
    assert a.equals(b)


def test_transmissibility_limits():
    assert transmissibility(0.0, 0.3) == pytest.approx(0.0)
    assert transmissibility(1.0, 0.3) == pytest.approx(1.0)
    assert transmissibility(0.2, 1.0) == pytest.approx(0.2)  # one day of sharing


def test_threshold_is_positive(small_graph):
    G, *_ = small_graph
    assert 0 < critical_transmissibility(G) < 1
