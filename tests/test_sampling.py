import numpy as np
import pandas as pd
import pytest

from data_sampler.sampling import (
    find_stratification_columns,
    sample,
    stratified_sample,
)


def test_sample_returns_exact_count(demo_df):
    result = sample(demo_df, 100, random_state=1)
    assert len(result.data) == 100
    assert result.method == "stratified"


def test_sample_preserves_proportions(demo_df):
    result = sample(demo_df, 200, random_state=1)
    orig = demo_df["region"].value_counts(normalize=True)
    samp = result.data["region"].value_counts(normalize=True)
    for region in orig.index:
        assert abs(orig[region] - samp.get(region, 0)) < 0.05


def test_request_more_than_rows_returns_all(demo_df):
    result = sample(demo_df, 10_000)
    assert len(result.data) == len(demo_df)
    assert result.method == "all"
    assert result.notes


def test_request_exact_row_count_returns_all(demo_df):
    result = sample(demo_df, len(demo_df))
    assert result.method == "all"


def test_pure_random_mode(demo_df):
    result = sample(demo_df, 50, use_random=True, random_state=7)
    assert len(result.data) == 50
    assert result.method == "random"
    assert result.strat_cols == []


def test_exclude_columns_are_not_stratified(demo_df):
    cols = find_stratification_columns(demo_df, 100)
    assert "region" in cols and "tier" in cols
    cols_excl = find_stratification_columns(demo_df, 100, exclude=["region"])
    assert "region" not in cols_excl
    result = sample(demo_df, 100, exclude_columns=["region", "tier", "active"])
    assert "region" not in result.strat_cols
    assert "tier" not in result.strat_cols
    assert "active" not in result.strat_cols


def test_exclude_all_falls_back_to_random(demo_df):
    result = sample(
        demo_df, 100, exclude_columns=list(demo_df.columns), random_state=3
    )
    assert result.method == "random"
    assert len(result.data) == 100


def test_seed_reproducibility(demo_df):
    a = sample(demo_df, 100, random_state=42)
    b = sample(demo_df, 100, random_state=42)
    assert sorted(a.data.index) == sorted(b.data.index)


def test_nan_rows_survive_stratification(missing_df):
    result = sample(missing_df, 200, random_state=1)
    if "region" in result.strat_cols:
        assert result.data["region"].isna().sum() > 0


def test_stratified_sample_direct(demo_df):
    out, sizes, allocs = stratified_sample(demo_df, 100, ["region"], random_state=0)
    assert len(out) == 100
    assert int(sizes.sum()) == len(demo_df)
    assert int(allocs.sum()) == 100


def test_sample_count_one(demo_df):
    result = sample(demo_df, 1, random_state=1)
    assert len(result.data) == 1


def test_two_row_frame():
    df = pd.DataFrame({"a": ["x", "y"], "n": [1, 2]})
    result = sample(df, 1, random_state=1)
    assert len(result.data) == 1


def test_no_mutation_of_source(demo_df):
    before = demo_df.copy()
    sample(demo_df, 100, random_state=1)
    pd.testing.assert_frame_equal(before, demo_df)


def test_sample_rows_come_from_source(demo_df):
    result = sample(demo_df, 100, random_state=1)
    assert result.data.index.isin(demo_df.index).all()
    assert not result.data.index.duplicated().any()
