import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def demo_df():
    """1000-row frame with categorical, numeric, text, and missing data."""
    rng = np.random.default_rng(42)
    n = 1000
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "name": [f"Person {i}" for i in range(n)],
            "region": rng.choice(
                ["North", "South", "East", "West"], n, p=[0.5, 0.3, 0.15, 0.05]
            ),
            "tier": rng.choice(["gold", "silver", "bronze"], n, p=[0.2, 0.3, 0.5]),
            "score": rng.normal(70, 15, n).round(1),
            "active": rng.choice([True, False], n),
            "notes": ["long text " * 20] * n,
        }
    )


@pytest.fixture
def missing_df(demo_df):
    df = demo_df.copy()
    df.loc[df.index[:100], "region"] = np.nan
    df.loc[df.index[:50], "score"] = np.nan
    return df
