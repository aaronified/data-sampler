import numpy as np
import pandas as pd
import pytest

from data_sampler.stats import (
    ColumnStats,
    compute_column_stats,
    compute_stats,
    is_stratifiable,
    sparkline,
)


def test_compute_stats_covers_all_columns(demo_df):
    stats = compute_stats(demo_df)
    assert [s.name for s in stats] == list(demo_df.columns)
    assert all(isinstance(s, ColumnStats) for s in stats)


def test_numeric_column_stats(demo_df):
    s = compute_column_stats(demo_df["score"])
    assert s.kind == "numeric"
    assert s.min is not None and s.max is not None
    assert s.min <= s.median <= s.max
    assert len(s.histogram) == 10
    assert sum(s.histogram) == s.count
    # numeric columns render their histogram; top-values are skipped since
    # stringifying every number dominated runtime and nothing consumed them
    assert s.top_values == []


def test_mode_is_populated_for_every_kind():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 2.0, 3.0, 2.0],
            "cat": ["a", "b", "b", "b", "c"],
        }
    )
    num = compute_column_stats(df["num"])
    assert num.kind == "numeric"
    assert num.mode == 2.0  # most frequent numeric value, kept as a float
    cat = compute_column_stats(df["cat"])
    assert cat.mode == "b"  # most frequent category, as a string


def test_mode_is_none_for_empty_column():
    s = compute_column_stats(pd.Series([np.nan, np.nan], name="empty"))
    assert s.mode is None


def test_categorical_column_stats(demo_df):
    s = compute_column_stats(demo_df["region"])
    assert s.kind == "categorical"
    assert s.unique == 4
    assert s.top_values[0][0] == "North"  # most frequent first
    assert s.stratifiable


def test_long_text_classified_as_text_and_not_stratifiable(demo_df):
    s = compute_column_stats(demo_df["notes"])
    assert s.kind == "text"
    assert not s.stratifiable


def test_id_column_not_stratifiable(demo_df):
    s = compute_column_stats(demo_df["id"])
    assert not s.stratifiable


def test_missing_values_counted(missing_df):
    s = compute_column_stats(missing_df["region"])
    assert s.missing == 100
    assert abs(s.missing_pct - 10.0) < 1e-9


def test_all_null_column():
    s = compute_column_stats(pd.Series([None, None, None], name="empty"))
    assert s.count == 0
    assert s.missing == 3
    assert s.top_values == []
    assert s.summary() == ""


def test_empty_dataframe():
    stats = compute_stats(pd.DataFrame({"a": pd.Series(dtype=float)}))
    assert stats[0].count == 0
    assert stats[0].missing_pct == 0.0


def test_single_value_column_not_stratifiable():
    series = pd.Series(["same"] * 50, name="const")
    assert not is_stratifiable(series, 50)


def test_continuous_numeric_not_stratifiable():
    # low cardinality but fractional values → continuous, skipped by default
    series = pd.Series([9.99, 19.99, 29.99, 39.99] * 25, name="price")
    assert not is_stratifiable(series, 100)


def test_decimal_object_column_rules_like_the_engine():
    # parquet DECIMAL / SQL drivers arrive as object-of-Decimal: the fractional
    # rule is value-based, so these must rule identically to the DuckDB probe
    from decimal import Decimal

    frac = pd.Series(
        [Decimal("9.99"), Decimal("19.99"), Decimal("29.99"), Decimal("39.99")] * 25,
        name="price",
    )
    assert not is_stratifiable(frac, 100)
    whole = pd.Series([Decimal("1"), Decimal("2"), Decimal("3")] * 34, name="code")
    assert is_stratifiable(whole, 102)


def test_float_backed_categorical_rules_like_the_engine():
    # DuckDB registration flattens category-of-float to DOUBLE; the pandas
    # side must rule on the category values the same way
    frac = pd.Series([1.5, 2.5, 3.5, 4.5] * 25, name="band").astype("category")
    assert not is_stratifiable(frac, 100)
    whole = pd.Series([1.0, 2.0, 3.0, 4.0] * 25, name="code").astype("category")
    assert is_stratifiable(whole, 100)


def test_unused_fractional_category_ignored():
    # row-filtering keeps unused categories around; the rule must judge
    # observed values only — all the engine ever sees after registration
    s = pd.Series(
        pd.Categorical([1.0, 2.0] * 50, categories=[1.0, 2.0, 3.5]), name="band"
    )
    assert is_stratifiable(s, 100)


def test_arrow_decimal_column_rules_like_the_engine():
    # read_parquet(dtype_backend='pyarrow') yields ArrowDtype(decimal128) for
    # parquet DECIMAL — same verdict as the engine's DECIMAL probe required
    pa = pytest.importorskip("pyarrow")
    from decimal import Decimal

    frac = pd.Series(
        [Decimal("9.99"), Decimal("19.99"), Decimal("29.99")] * 34,
        dtype=pd.ArrowDtype(pa.decimal128(6, 2)),
        name="price",
    )
    assert not is_stratifiable(frac, 102)
    whole = pd.Series(
        [Decimal("10"), Decimal("20"), Decimal("30")] * 34,
        dtype=pd.ArrowDtype(pa.decimal128(6, 2)),
        name="code",
    )
    assert is_stratifiable(whole, 102)


def test_object_float_column_fractional_not_stratifiable():
    frac = pd.Series([9.99, 19.99, 29.99, 39.99] * 25, dtype=object, name="price")
    assert not is_stratifiable(frac, 100)
    mixed = pd.Series(["a", "b", 1.5, 2.5] * 25, dtype=object, name="junk")
    assert is_stratifiable(mixed, 100)  # not number-valued → old rules apply


def test_whole_number_float_still_stratifiable():
    # integer-coded columns arrive as float64 once NaNs appear — they stay
    # candidates as long as every value is a whole number
    series = pd.Series([1.0, 2.0, 3.0, np.nan] * 25, name="rating")
    assert is_stratifiable(series, 100)


def test_discrete_int_column_stratifiable():
    series = pd.Series([1, 2, 3, 4, 5] * 20, name="stars")
    assert is_stratifiable(series, 100)


def test_boolean_column_kind(demo_df):
    assert compute_column_stats(demo_df["active"]).kind == "boolean"


def test_sparkline_shapes():
    assert sparkline([]) == ""
    assert sparkline([0, 0]) == "▁▁"
    line = sparkline([1, 5, 10])
    assert len(line) == 3
    assert line[-1] == "█"
    assert sparkline([0, 10])[0] == "▁"  # zero stays at baseline


def test_injection_shaped_strings_are_safe():
    series = pd.Series(["'; DROP TABLE users; --", "<script>", "{}\n\t"], name="evil")
    s = compute_column_stats(series)
    assert s.unique == 3
    assert s.kind == "categorical"


def test_infinite_values_do_not_crash():
    series = pd.Series([1.0, np.inf, -np.inf, 5.0], name="inf")
    s = compute_column_stats(series)
    assert s.kind == "numeric"
