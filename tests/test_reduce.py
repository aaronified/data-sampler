import numpy as np
import pandas as pd
import pytest

from data_sampler.reduce import ReductionResult, reduce_columns
from data_sampler.report import format_reduction_report


@pytest.fixture
def wide_df():
    """Frame with two correlated numeric clusters + an independent column."""
    rng = np.random.default_rng(7)
    n = 400
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "label": rng.choice(["x", "y", "z"], n),
            "height": a * 10 + 170 + rng.normal(0, 0.5, n),
            "weight": a * 8 + 70 + rng.normal(0, 0.5, n),
            "bmi": a * 3 + 24 + rng.normal(0, 0.3, n),
            "price": b * 100 + 500 + rng.normal(0, 2, n),
            "tax": b * 20 + 100 + rng.normal(0, 0.5, n),
            "noise": rng.normal(0, 1, n),
            "flag": rng.choice([True, False], n),
        }
    )


# ── selector validation ───────────────────────────────────────────────────────

def test_requires_exactly_one_selector(demo_df):
    with pytest.raises(ValueError):
        reduce_columns(demo_df)
    with pytest.raises(ValueError):
        reduce_columns(demo_df, n_components=2, variance_ratio=0.9)


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_rejects_bad_n_components(demo_df, bad):
    with pytest.raises(ValueError):
        reduce_columns(demo_df, n_components=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 2.0])
def test_rejects_bad_variance_ratio(demo_df, bad):
    with pytest.raises(ValueError):
        reduce_columns(demo_df, variance_ratio=bad)


def test_rejects_unknown_column(demo_df):
    with pytest.raises(ValueError):
        reduce_columns(demo_df, n_components=2, columns=["ghost"])


# ── n_components path ────────────────────────────────────────────────────────

def test_n_components_shape_and_preservation(wide_df):
    result = reduce_columns(wide_df, n_components=2, seed=1)
    assert result.n_components == 2
    assert result.component_names == ["PC1", "PC2"]
    # every numeric column consumed; label/flag preserved untouched
    assert result.source_columns == [
        "height", "weight", "bmi", "price", "tax", "noise"
    ]
    assert len(result.data) == len(wide_df)
    assert list(result.data.index) == list(wide_df.index)
    pd.testing.assert_series_equal(result.data["label"], wide_df["label"])
    pd.testing.assert_series_equal(result.data["flag"], wide_df["flag"])
    # PC block sits where the numeric block began
    assert list(result.data.columns) == ["label", "PC1", "PC2", "flag"]


def test_bool_column_is_not_reduced(demo_df):
    result = reduce_columns(demo_df, n_components=2)
    assert "active" not in result.source_columns
    pd.testing.assert_series_equal(result.data["active"], demo_df["active"])


def test_exclude_passes_column_through(wide_df):
    result = reduce_columns(wide_df, n_components=2, exclude=["noise"])
    assert "noise" not in result.source_columns
    pd.testing.assert_series_equal(result.data["noise"], wide_df["noise"])


def test_exclude_accepts_bare_string(wide_df):
    # exclude="noise" must mean the column, not the characters n/o/i/s/e
    result = reduce_columns(wide_df, n_components=2, exclude="noise")
    assert "noise" not in result.source_columns
    pd.testing.assert_series_equal(result.data["noise"], wide_df["noise"])


def test_columns_rejects_bool_and_datetime(wide_df):
    with pytest.raises(ValueError, match="boolean"):
        reduce_columns(wide_df, n_components=1, columns=["height", "flag"])
    df = wide_df.assign(when=pd.date_range("2024-01-01", periods=len(wide_df)))
    with pytest.raises(ValueError, match="datetime"):
        reduce_columns(df, n_components=1, columns=["height", "when"])


def test_columns_deduplicates(wide_df):
    result = reduce_columns(wide_df, n_components=1, columns=["height", "height", "weight"])
    assert result.source_columns == ["height", "weight"]


def test_columns_accepts_coercible_object_dtype(wide_df):
    # Decimal-typed columns (database drivers) classify as categorical but
    # coerce cleanly when named explicitly
    from decimal import Decimal

    df = wide_df.assign(dec=[Decimal(i) + Decimal("0.5") for i in range(len(wide_df))])
    result = reduce_columns(df, n_components=1, columns=["height", "dec"])
    assert result.source_columns == ["height", "dec"]
    assert result.n_components == 1


def test_n_components_clamped_to_column_count(wide_df):
    result = reduce_columns(wide_df, n_components=50)
    assert result.n_components == 6  # only 6 numeric columns
    assert any("only 6 are possible" in n for n in result.notes)


def test_no_mutation_of_source(wide_df):
    before = wide_df.copy()
    reduce_columns(wide_df, n_components=2)
    pd.testing.assert_frame_equal(wide_df, before)


def test_reproducible_across_runs(wide_df):
    a = reduce_columns(wide_df, n_components=3, seed=5)
    b = reduce_columns(wide_df, n_components=3, seed=5)
    pd.testing.assert_frame_equal(a.data, b.data)
    assert a.explained_variance_ratio == b.explained_variance_ratio


# ── variance_ratio path ──────────────────────────────────────────────────────

def test_variance_ratio_picks_fewest_components(wide_df):
    result = reduce_columns(wide_df, variance_ratio=0.9)
    cum = result.cumulative_variance_ratio
    assert cum[-1] >= 0.9
    if result.n_components > 1:
        assert cum[-2] < 0.9  # minimality: one fewer would not reach 0.9
    # two strong clusters + noise → 2 components carry most of the variance
    assert result.n_components <= 3
    assert any("retain ≥ 90%" in n for n in result.notes)


def test_variance_math_invariants(wide_df):
    result = reduce_columns(wide_df, n_components=6)
    evr = result.explained_variance_ratio
    assert all(v >= 0 for v in evr)
    assert all(evr[i] >= evr[i + 1] for i in range(len(evr) - 1))  # descending
    # all 6 components on standardized data → ratios sum to 1
    assert sum(evr) == pytest.approx(1.0)
    assert result.total_variance_retained == pytest.approx(1.0)


# ── missing / constant / degenerate inputs ────────────────────────────────────

def test_nan_imputed_without_row_loss(missing_df):
    result = reduce_columns(missing_df, n_components=1)  # id + score are numeric
    assert result.n_components == 1
    assert len(result.data) == len(missing_df)
    assert result.data["PC1"].notna().all()
    assert any("Mean-imputed" in n for n in result.notes)


def test_constant_column_passes_through(wide_df):
    df = wide_df.assign(constant=5.0)
    result = reduce_columns(df, n_components=2)
    assert "constant" in result.dropped_columns
    assert "constant" not in result.source_columns
    pd.testing.assert_series_equal(result.data["constant"], df["constant"])
    assert any("no signal" in n for n in result.notes)


def test_no_numeric_columns_is_noop():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    result = reduce_columns(df, n_components=2)
    assert result.n_components == 0
    pd.testing.assert_frame_equal(result.data, df)
    assert any("no numeric columns" in n for n in result.notes)


def test_single_numeric_column_is_noop(wide_df):
    result = reduce_columns(wide_df, n_components=1, columns=["noise"])
    assert result.n_components == 0
    pd.testing.assert_frame_equal(result.data, wide_df)
    assert any("only one usable numeric column" in n for n in result.notes)


def test_id_like_column_warns(demo_df):
    result = reduce_columns(demo_df, n_components=1)
    assert any("looks like an identifier" in n for n in result.notes)


def test_id_hint_matches_tokens_not_substrings(wide_df):
    # "tax_paid" contains "id" as a substring but is not an identifier;
    # the hint must be token-based (same fix the date hints already have)
    df = wide_df.rename(columns={"tax": "tax_paid"})
    result = reduce_columns(df, n_components=2)
    assert not any("tax_paid" in n and "identifier" in n for n in result.notes)
    # a real id-token column still warns
    df2 = wide_df.assign(user_id=np.arange(len(wide_df), dtype=float) + 0.5)
    result2 = reduce_columns(df2, n_components=2)
    assert any("'user_id' looks like an identifier" in n for n in result2.notes)


def test_prefix_collision_raises(wide_df):
    df = wide_df.assign(PC1=["a"] * len(wide_df))  # non-numeric, would collide
    with pytest.raises(ValueError):
        reduce_columns(df, n_components=2)
    # a different prefix resolves the collision
    result = reduce_columns(df, n_components=2, prefix="Z")
    assert result.component_names == ["Z1", "Z2"]
    assert "Z1" in result.data.columns and "PC1" in result.data.columns


def test_extreme_magnitudes_do_not_zero_variance_ratios():
    # sparse huge values pass the per-column guards but (S**2).sum() would
    # overflow to inf without pre-scaling, silently zeroing every ratio
    n = 10
    h = np.zeros(n); h[0], h[1] = 9.0e153, -9.0e153
    g = np.zeros(n); g[2], g[3] = 6.0e153, -6.0e153  # orthogonal to h
    df = pd.DataFrame({"h": h, "g": g})
    result = reduce_columns(df, n_components=2, standardize=False)
    evr = result.explained_variance_ratio
    assert evr[0] == pytest.approx(0.6923, abs=1e-3)  # 81 / (81 + 36)
    assert evr[1] == pytest.approx(0.3077, abs=1e-3)
    assert result.total_variance_retained == pytest.approx(1.0)


# ── standardization ───────────────────────────────────────────────────────────

def test_standardize_balances_mixed_scales():
    rng = np.random.default_rng(3)
    n = 500
    small = rng.normal(0, 1, n)
    big = rng.normal(0, 1, n) * 1e6
    df = pd.DataFrame({"small": small, "big": big})
    raw = reduce_columns(df, n_components=1, standardize=False)
    std = reduce_columns(df, n_components=1, standardize=True)
    # unstandardized: PC1 is essentially the big-unit column
    raw_l = np.abs(raw.components[0])
    assert raw_l[1] > 0.99 and raw_l[0] < 0.01
    # standardized: both columns contribute comparably
    std_l = np.abs(std.components[0])
    assert abs(std_l[0] - std_l[1]) < 0.2
    assert std.standardized and not raw.standardized


# ── rationale: correlation matrix + groups ────────────────────────────────────

def test_correlation_groups_cluster_correlated_columns(wide_df):
    result = reduce_columns(wide_df, n_components=3)
    assert sorted(map(sorted, result.column_groups)) == [
        ["bmi", "height", "weight"],
        ["price", "tax"],
    ]
    corr = result.correlation_matrix
    assert list(corr.index) == result.source_columns
    assert list(corr.columns) == result.source_columns
    assert corr.loc["height", "weight"] > 0.9
    assert abs(corr.loc["noise", "height"]) < 0.2


def test_group_threshold_is_respected(wide_df):
    # a threshold above every pairwise |r| leaves no groups
    result = reduce_columns(wide_df, n_components=2, group_threshold=0.9999)
    assert result.column_groups == []


# ── report rendering ─────────────────────────────────────────────────────────

def test_format_reduction_report_sections(wide_df):
    result = reduce_columns(wide_df, variance_ratio=0.9)
    text = format_reduction_report(result)
    assert "COLUMN REDUCTION (PCA)" in text
    assert "CORRELATED COLUMN GROUPS" in text
    assert "height, weight, bmi" in text or "height" in text
    assert "PC1" in text and "cumulative" in text
    assert "Top drivers" in text


def test_format_reduction_report_noop_returns_notes():
    df = pd.DataFrame({"a": ["x", "y"]})
    result = reduce_columns(df, n_components=2)
    text = format_reduction_report(result)
    assert "skipped" in text
    assert "COLUMN REDUCTION" not in text
