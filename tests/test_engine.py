"""Tests for the DuckDB out-of-core engine (data_sampler.engine).

Skipped entirely if the optional 'large' extra (duckdb) is not installed.
"""

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from data_sampler.engine import (  # noqa: E402
    DuckDBEngine,
    _proportional_allocation,
    duckdb_available,
    large_materialization_warning,
    should_use_engine,
    sample as engine_sample,
)


@pytest.fixture
def big_df():
    """A 20k-row frame with skewed categoricals + numeric + text."""
    rng = np.random.default_rng(7)
    n = 20_000
    return pd.DataFrame(
        {
            "id": range(n),
            "region": rng.choice(["North", "South", "East", "West"], n, p=[0.5, 0.3, 0.15, 0.05]),
            "tier": rng.choice(["gold", "silver", "bronze"], n, p=[0.2, 0.3, 0.5]),
            "score": rng.normal(70, 15, n).round(1),
            "note": ["a fairly long free-text field exceeding fifty chars " * 2] * n,
        }
    )


@pytest.fixture
def engine():
    with DuckDBEngine(threads=4) as e:
        yield e


# ── allocation helper ─────────────────────────────────────────────────────────

def test_proportional_allocation_sums_and_bounds():
    sizes = np.array([500, 300, 150, 50])
    alloc = _proportional_allocation(sizes, 100)
    assert alloc.sum() == 100
    assert (alloc <= sizes).all()
    # proportional: the biggest stratum gets the most
    assert alloc[0] == alloc.max()


def test_proportional_allocation_never_over_allocates_small_strata():
    sizes = np.array([1_000_000, 1, 1, 1])
    alloc = _proportional_allocation(sizes, 10)
    assert alloc.sum() == 10
    assert (alloc <= sizes).all()


# ── introspection ─────────────────────────────────────────────────────────────

def test_row_count_and_columns(engine, big_df):
    assert engine.row_count(big_df) == 20_000
    assert engine.columns(big_df) == ["id", "region", "tier", "score", "note"]


# ── reservoir (random) sampling ───────────────────────────────────────────────

def test_reservoir_exact_count_and_subset(engine, big_df):
    result = engine.sample(big_df, 500, use_random=True, seed=1)
    assert result.method == "random"
    assert len(result.data) == 500
    # every sampled id exists in the source
    assert set(result.data["id"]).issubset(set(big_df["id"]))


def test_reservoir_seed_reproducible(engine, big_df):
    a = engine.sample(big_df, 300, use_random=True, seed=42)
    b = engine.sample(big_df, 300, use_random=True, seed=42)
    assert sorted(a.data["id"].tolist()) == sorted(b.data["id"].tolist())


# ── stratified sampling ───────────────────────────────────────────────────────

def test_stratified_exact_count_and_distribution(engine, big_df):
    result = engine.sample(big_df, 1000, strat_cols=["region"], seed=1)
    assert result.method == "stratified"
    assert len(result.data) == 1000
    # proportions preserved within a tolerance
    src_p = big_df["region"].value_counts(normalize=True)
    smp_p = result.data["region"].value_counts(normalize=True)
    for cat in src_p.index:
        assert abs(src_p[cat] - smp_p.get(cat, 0)) < 0.03


def test_stratified_allocation_sums_to_count(engine, big_df):
    result = engine.sample(big_df, 777, strat_cols=["region", "tier"], seed=3)
    assert len(result.data) == 777
    assert int(result.allocations.sum()) == 777


def test_stratified_seed_reproducible_single_threaded(big_df):
    # seeded stratified runs go single-threaded for determinism
    with DuckDBEngine(threads=4) as e1:
        a = e1.sample(big_df, 500, strat_cols=["region"], seed=9)
    with DuckDBEngine(threads=4) as e2:
        b = e2.sample(big_df, 500, strat_cols=["region"], seed=9)
    assert sorted(a.data["id"].tolist()) == sorted(b.data["id"].tolist())


def test_auto_stratification_picks_low_cardinality(engine, big_df):
    cols = engine.find_stratification_columns(big_df, 1000)
    # region/tier are low-cardinality categoricals; id/score/note are not
    assert "region" in cols or "tier" in cols
    assert "id" not in cols and "note" not in cols


def test_sample_all_when_count_exceeds_rows(engine):
    small = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = engine.sample(small, 10)
    assert result.method == "all"
    assert len(result.data) == 3


# ── file readers (CSV / Parquet) ──────────────────────────────────────────────

def test_reads_csv(tmp_path, big_df, engine):
    p = tmp_path / "data.csv"
    big_df.to_csv(p, index=False)
    result = engine.sample(str(p), 400, strat_cols=["region"], seed=1)
    assert len(result.data) == 400
    assert set(result.data.columns) == set(big_df.columns)


def test_reads_parquet(tmp_path, big_df, engine):
    p = tmp_path / "data.parquet"
    big_df.to_parquet(p, index=False)
    assert engine.row_count(str(p)) == 20_000
    result = engine.sample(str(p), 400, use_random=True, seed=1)
    assert len(result.data) == 400


def test_unsupported_extension_raises(engine, tmp_path):
    p = tmp_path / "data.xlsx"
    p.write_bytes(b"not really excel")
    with pytest.raises(ValueError, match="cannot read"):
        engine.row_count(str(p))


# ── module-level helpers ──────────────────────────────────────────────────────

def test_should_use_engine_for_parquet(tmp_path, big_df):
    p = tmp_path / "d.parquet"
    big_df.to_parquet(p, index=False)
    assert should_use_engine(str(p)) is True
    # a small pandas frame does not trigger the engine
    assert should_use_engine(big_df) is False


def test_large_materialization_warning():
    assert large_materialization_warning(10_000_000, 50) is not None
    assert "out-of-core" in large_materialization_warning(10_000_000, 50)
    assert large_materialization_warning(100, 5) is None


def test_module_level_sample_helper(big_df):
    result = engine_sample(big_df, 200, use_random=True, seed=1)
    assert len(result.data) == 200


def test_duckdb_available_true():
    assert duckdb_available() is True
