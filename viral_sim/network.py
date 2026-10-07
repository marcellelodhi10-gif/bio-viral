"""Build a synthetic social graph that looks like a real one.

Real social networks have three properties a plain random graph lacks:
  1. Heavy-tailed degree (a few "influencers" with huge followings)
  2. Clustering (your friends know each other)
  3. Communities (regions / languages / niches with few links between them)

We get (1) and (2) from the Holme-Kim power-law cluster model, and (3) by
building several regional sub-graphs and stitching them with sparse bridges.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp


def build_social_graph(
    n_users: int = 5000,
    n_regions: int = 5,
    m: int = 4,
    p_triangle: float = 0.3,
    cross_region_edges: int = 150,
    seed: int = 42,
) -> nx.Graph:
    """Return a graph whose nodes carry a ``region`` attribute.

    m                  -- edges each new user creates (controls average degree ~2m)
    p_triangle         -- chance a new edge closes a triangle (controls clustering)
    cross_region_edges -- number of bridge edges between regions
    """
    rng = np.random.default_rng(seed)
    sizes = [n_users // n_regions] * n_regions
    sizes[-1] += n_users - sum(sizes)

    G = nx.Graph()
    offset = 0
    for region, size in enumerate(sizes):
        sub = nx.powerlaw_cluster_graph(size, m, p_triangle, seed=int(rng.integers(1e9)))
        sub = nx.relabel_nodes(sub, {v: v + offset for v in sub})
        nx.set_node_attributes(sub, region, "region")
        G.update(sub)
        offset += size

    nodes_by_region = [
        [v for v, r in G.nodes(data="region") if r == region] for region in range(n_regions)
    ]
    added = 0
    while added < cross_region_edges:
        r1, r2 = rng.choice(n_regions, size=2, replace=False)
        u = int(rng.choice(nodes_by_region[r1]))
        v = int(rng.choice(nodes_by_region[r2]))
        if not G.has_edge(u, v):
            G.add_edge(u, v, bridge=True)
            added += 1
    return G


def to_arrays(G: nx.Graph) -> tuple[sp.csr_matrix, np.ndarray]:
    """Sparse adjacency matrix + region label per node (index-aligned)."""
    nodes = sorted(G.nodes)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.float32)
    regions = np.array([G.nodes[v]["region"] for v in nodes], dtype=np.int32)
    return sp.csr_matrix(A), regions


def graph_summary(G: nx.Graph) -> dict:
    deg = np.array([d for _, d in G.degree()])
    return {
        "users": G.number_of_nodes(),
        "connections": G.number_of_edges(),
        "mean_degree": round(float(deg.mean()), 2),
        "max_degree": int(deg.max()),
        "avg_clustering": round(nx.average_clustering(G), 3),
        "regions": len(set(nx.get_node_attributes(G, "region").values())),
    }
