import numpy as np
import pandas as pd

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
