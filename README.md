# The Volatility Complex

Topological analysis of sector stock volatility using Vietoris-Rips complexes and persistent homology.

## Overview

We construct **Vietoris-Rips complexes** from the rolling volatility correlation structure of S&P 500 sector stocks. The pipeline:

1. Download sector stock prices (e.g., 50 Information Technology stocks)
2. Compute rolling annualized volatility from log-returns
3. Convert volatility correlations to a distance metric: $D_{ij} = \sqrt{2(1 - \rho_{ij})}$
4. Build Vietoris-Rips complex at each time step (edge when $D_{ij} < r$)
5. Track the complex's evolution (f-vector over time)
6. Compute persistent homology via filtration over the radius parameter

During market stress, volatilities become highly correlated and the complex becomes dense. In calm periods, the complex fragments into clusters. Persistent homology reveals the multi-scale topological structure at each point in time.

## Structure

```
volatility_complex.py           # Single importable module (all functions)
notebooks/volatility_complex.ipynb  # Main demonstration notebook
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter lab notebooks/volatility_complex.ipynb
```

## Key Dependencies

- **yfinance** — market data download
- **numpy / pandas** — data processing
- **networkx** — graph visualization
- **gudhi** — persistent homology computation
- **ipywidgets** — interactive exploration widgets
