"""
Volatility Complex — Vietoris-Rips complexes from sector stock volatility.

Single-file library for building, analyzing, and visualizing simplicial
complexes constructed from rolling volatility correlation/distance matrices
of S&P 500 sector stocks.

Sections
--------
1. Data Loading       — download prices via yfinance, S&P 500 sector/sample
2. Preprocessing      — log returns, rolling correlation & distance matrices
3. Simplicial Complex — SimplicialComplex dataclass, Rips & correlation construction
4. Rolling Complexes  — time-varying complex construction
5. Topological Tools  — f-vector, simplex counts
6. Visualization      — graph plots, summary plots, adjacency heatmaps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import yfinance as yf


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLP", "XLU", "XLY", "XLB", "XLRE", "XLC",
]

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_SECTOR_ALIASES: dict[str, str] = {
    "tech": "Information Technology",
    "technology": "Information Technology",
    "finance": "Financials",
    "financials": "Financials",
    "energy": "Energy",
    "oil": "Energy",
    "health": "Health Care",
    "healthcare": "Health Care",
    "industrials": "Industrials",
    "staples": "Consumer Staples",
    "consumer staples": "Consumer Staples",
    "utilities": "Utilities",
    "discretionary": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "materials": "Materials",
    "real estate": "Real Estate",
    "communication": "Communication Services",
    "communication services": "Communication Services",
}


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Download daily close prices via yfinance, cache to parquet."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    safe_name = "_".join(sorted(tickers)[:5]) + f"_{len(tickers)}_{start}_{end}"
    parquet_file = cache_path / f"{safe_name}.parquet"

    if parquet_file.exists():
        return pd.read_parquet(parquet_file)

    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="any")
    prices.to_parquet(parquet_file)
    return prices


def load_sector_etfs(start: str, end: str) -> pd.DataFrame:
    """Download the 11 SPDR sector ETFs."""
    return download_prices(SECTOR_ETFS, start, end)


def load_sp500_sample(
    n: int,
    start: str,
    end: str,
    seed: int = 42,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Download a random sample of n S&P 500 constituents."""
    import warnings

    tables = pd.read_html(
        _SP500_URL,
        storage_options={"User-Agent": "Mozilla/5.0"},
    )
    sp500 = tables[0]
    all_tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()

    rng = np.random.default_rng(seed)
    pool_size = min(len(all_tickers), max(n * 3, n + 50))
    pool = rng.choice(all_tickers, size=pool_size, replace=False).tolist()

    raw = yf.download(pool, start=start, end=end, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = pool

    max_rows = len(prices)
    coverage = prices.notna().sum() / max_rows
    valid = coverage[coverage >= min_coverage].index.tolist()

    rng.shuffle(valid)
    selected = valid[:n]

    if len(selected) < n:
        warnings.warn(
            f"Only {len(selected)} of {n} requested tickers have "
            f">={min_coverage:.0%} price coverage for {start} to {end}.",
            stacklevel=2,
        )

    return download_prices(selected, start, end)


def load_sp500_sector(
    sector: str,
    start: str,
    end: str,
    n: int = 50,
    seed: int = 42,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Download S&P 500 constituents from a specific GICS sector."""
    import warnings

    gics_sector = _SECTOR_ALIASES.get(sector.lower(), sector)

    tables = pd.read_html(
        _SP500_URL,
        storage_options={"User-Agent": "Mozilla/5.0"},
    )
    sp500 = tables[0]

    mask = sp500["GICS Sector"].str.lower() == gics_sector.lower()
    sector_df = sp500[mask]
    if sector_df.empty:
        valid_sectors = sorted(sp500["GICS Sector"].unique())
        raise ValueError(
            f"Sector {sector!r} (resolved to {gics_sector!r}) not found. "
            f"Valid GICS sectors: {valid_sectors}"
        )

    all_tickers = (
        sector_df["Symbol"].str.replace(".", "-", regex=False).tolist()
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(all_tickers)

    pool = all_tickers[: max(n * 2, len(all_tickers))]

    raw = yf.download(pool, start=start, end=end, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = pool

    max_rows = len(prices)
    coverage = prices.notna().sum() / max_rows
    valid = coverage[coverage >= min_coverage].index.tolist()

    rng.shuffle(valid)
    selected = valid[:n]

    if len(selected) < n:
        warnings.warn(
            f"Only {len(selected)} of {n} requested tickers in "
            f"{gics_sector!r} have >={min_coverage:.0%} price coverage "
            f"for {start} to {end}.",
            stacklevel=2,
        )

    return download_prices(selected, start, end)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log-returns: ln(P_t / P_{t-1})."""
    return np.log(1 + prices.pct_change()).dropna()


def rolling_correlation(
    returns: pd.DataFrame,
    window: int = 60,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Rolling Pearson correlation matrices."""
    result = {}
    dates = returns.index
    for i in range(window, len(dates)):
        t = dates[i]
        result[t] = returns.iloc[i - window : i].corr()
    return result


def rolling_distance(
    returns: pd.DataFrame,
    window: int = 60,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Rolling distance matrices: D_ij = sqrt(2*(1 - corr_ij))."""
    result = {}
    dates = returns.index
    for i in range(window, len(dates)):
        t = dates[i]
        corr_matrix = returns.iloc[i - window : i].corr()
        result[t] = np.sqrt(2 * (1 - corr_matrix))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMPLICIAL COMPLEX
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimplicialComplex:
    """Oriented simplicial complex with canonical orientation (indices sorted).

    Stores 0-, 1-, 2-cells in dedicated fields.
    Higher-dimensional cells (3-simplices, etc.) live in ``higher_cells``.
    """

    vertices: list[str]
    edges: list[tuple[int, int]]
    triangles: list[tuple[int, int, int]]
    vertex_labels: dict[int, str] | None = None
    higher_cells: dict[int, list[tuple[int, ...]]] = field(default_factory=dict)

    def __post_init__(self):
        if self.vertex_labels is None:
            self.vertex_labels = {i: v for i, v in enumerate(self.vertices)}

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def edge_index(self, i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        return self.edges.index(key)

    def has_edge(self, i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in self._edge_set

    @property
    def _edge_set(self) -> set[tuple[int, int]]:
        return set(self.edges)


def from_distance_matrix(
    dist: pd.DataFrame,
    radius: float,
    max_dim: int = 2,
) -> SimplicialComplex:
    """Build a Vietoris-Rips complex from a distance matrix.

    Edge (i, j) included if dist(i, j) < radius.  Higher-dimensional
    k-simplices are filled by clique detection (flag complex).

    Parameters
    ----------
    dist : pd.DataFrame
        Symmetric distance matrix (e.g. D_ij = sqrt(2*(1 - rho_ij))).
    radius : float
        Rips parameter -- maximum pairwise distance for simplex inclusion.
    max_dim : int
        Maximum simplex dimension to construct (default 2 = triangles).
    """
    vertices = list(dist.columns)
    n = len(vertices)

    edges: list[tuple[int, int]] = []
    edge_set: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if dist.iloc[i, j] < radius:
                edges.append((i, j))
                edge_set.add((i, j))

    triangles: list[tuple[int, int, int]] = []
    if max_dim >= 2:
        for i, j, k in combinations(range(n), 3):
            if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set:
                triangles.append((i, j, k))

    higher_cells: dict[int, list[tuple[int, ...]]] = {}
    if max_dim >= 3:
        prev_simplices = [set(t) for t in triangles]
        for dim in range(3, max_dim + 1):
            current: list[tuple[int, ...]] = []
            seen: set[tuple[int, ...]] = set()
            for simplex in prev_simplices:
                for v in range(n):
                    if v in simplex:
                        continue
                    if all((min(v, u), max(v, u)) in edge_set for u in simplex):
                        candidate = tuple(sorted(simplex | {v}))
                        if candidate not in seen:
                            seen.add(candidate)
                            current.append(candidate)
            if not current:
                break
            higher_cells[dim] = current
            prev_simplices = [set(c) for c in current]

    return SimplicialComplex(
        vertices=vertices, edges=edges, triangles=triangles,
        higher_cells=higher_cells,
    )


def from_correlation(
    corr: pd.DataFrame,
    threshold: float = 0.3,
    use_clique: bool = True,
) -> SimplicialComplex:
    """Build complex from correlation matrix.

    Edges where |rho_ij| > threshold.
    If use_clique=True, triangles = all cliques of 1-skeleton (flag complex).
    """
    vertices = list(corr.columns)
    n = len(vertices)
    edges: list[tuple[int, int]] = []
    edge_set: set[tuple[int, int]] = set()

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr.iloc[i, j]) > threshold:
                edges.append((i, j))
                edge_set.add((i, j))

    triangles: list[tuple[int, int, int]] = []
    if use_clique:
        for i, j, k in combinations(range(n), 3):
            if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set:
                triangles.append((i, j, k))

    return SimplicialComplex(vertices=vertices, edges=edges, triangles=triangles)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ROLLING COMPLEXES
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_complexes(
    returns: pd.DataFrame,
    window: int = 60,
    step: int = 1,
    method: str = "distance",
    threshold: float = 1.0,
    max_dim: int = 2,
) -> dict[pd.Timestamp, SimplicialComplex]:
    """Build time-varying complexes over rolling windows.

    Parameters
    ----------
    returns : pd.DataFrame
        Return or volatility series (columns = assets, index = dates).
    window : int
        Rolling window size in trading days.
    step : int
        Step size between successive windows.
    method : str
        "distance" (Vietoris-Rips) or "correlation" (threshold on |rho|).
    threshold : float
        Rips radius (distance) or correlation threshold.
    max_dim : int
        Maximum simplex dimension (distance method only).
    """
    if method == "distance":
        matrices = rolling_distance(returns, window)
        builder = lambda mat: from_distance_matrix(mat, radius=threshold, max_dim=max_dim)
    elif method == "correlation":
        matrices = rolling_correlation(returns, window)
        builder = lambda mat: from_correlation(mat, threshold=threshold)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'distance' or 'correlation'.")

    timestamps = sorted(matrices.keys())
    result = {}
    for idx, t in enumerate(timestamps):
        if idx % step == 0:
            result[t] = builder(matrices[t])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TOPOLOGICAL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def f_vector(K: SimplicialComplex) -> list[int]:
    """Return the f-vector (simplex counts by dimension).

    f_vector[0] = vertices, f_vector[1] = edges, f_vector[2] = triangles, ...
    """
    max_dim = max(K.higher_cells.keys(), default=-1)
    f_vec = [K.n_vertices, K.n_edges, K.n_triangles]
    for d in range(3, max_dim + 1):
        f_vec.append(len(K.higher_cells.get(d, [])))
    return f_vec


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_graph(
    K: SimplicialComplex,
    ax: plt.Axes | None = None,
    layout: str = "spring",
    pos: dict | None = None,
    node_color=None,
    title: str | None = None,
    **kwargs,
) -> tuple[plt.Axes, dict]:
    """Draw the 1-skeleton with networkx.

    Returns (ax, pos) so positions can be reused for warm-starting.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    G = nx.Graph()
    for i, v in enumerate(K.vertices):
        G.add_node(i, label=v)
    G.add_edges_from(K.edges)

    layout_fn = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
    }
    fn = layout_fn.get(layout, nx.spring_layout)

    if pos is not None and layout in ("kamada_kawai", "spring"):
        init = {n: pos[n] for n in G.nodes if n in pos}
        if init:
            pos = fn(G, pos=init, **kwargs)
        else:
            pos = fn(G, **kwargs)
    else:
        pos = fn(G, **kwargs)

    labels = {i: v for i, v in enumerate(K.vertices)}
    nx.draw_networkx(
        G, pos, ax=ax,
        labels=labels,
        node_color=node_color or "lightblue",
        node_size=300,
        font_size=8,
        edge_color="gray",
        alpha=0.8,
    )
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return ax, pos


def plot_complex_summary(
    complexes: dict, ax: plt.Axes | None = None,
) -> plt.Axes:
    """Time series of |edges| and |triangles|."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 5))

    timestamps = sorted(complexes.keys())
    n_edges = [complexes[t].n_edges for t in timestamps]
    n_triangles = [complexes[t].n_triangles for t in timestamps]

    ax.plot(timestamps, n_edges, label="|edges|", linewidth=1.5)
    ax.plot(timestamps, n_triangles, label="|triangles|", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.set_title("Complex Size Over Time")
    ax.legend()
    return ax


def plot_adjacency_heatmap(
    K: SimplicialComplex, ax: plt.Axes | None = None,
) -> plt.Axes:
    """Adjacency matrix heatmap of the 1-skeleton."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    n = K.n_vertices
    adj = np.zeros((n, n))
    for i, j in K.edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    ax.imshow(adj, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(K.vertices, rotation=90, fontsize=7)
    ax.set_yticklabels(K.vertices, fontsize=7)
    ax.set_title("Adjacency Matrix (1-skeleton)")
    return ax
