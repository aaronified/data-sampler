import numpy as np
import pandas as pd

from data_sampler.report import (
    column_histogram_data,
    format_column_histograms,
    format_distribution,
    format_stratification_report,
)
from data_sampler.sampling import sample


def test_stratified_report_contains_columns_and_bars(demo_df):
    result = sample(demo_df, 200, random_state=1)
    text = format_stratification_report(demo_df, result)
    assert "STRATIFICATION REPORT" in text
    for col in result.strat_cols:
        assert f"'{col}'" in text
    assert "█" in text
    assert "Totals" in text


def test_random_report_is_note_only(demo_df):
    result = sample(demo_df, 50, use_random=True, random_state=1)
    text = format_stratification_report(demo_df, result)
    assert "STRATIFICATION REPORT" not in text
    assert "random" in text.lower()


def test_report_flags_missing_category(missing_df):
    result = sample(missing_df, 200, random_state=1)
    text = format_stratification_report(missing_df, result)
    if "region" in result.strat_cols:
        assert "(missing)" in text


def test_report_truncates_long_labels(demo_df):
    df = demo_df.copy()
    df["region"] = df["region"].map(
        lambda r: f"{r}_very_long_category_label_exceeding_13"
    )
    result = sample(df, 200, random_state=1)
    text = format_stratification_report(df, result)
    if "region" in result.strat_cols:
        assert "..." in text


def test_format_distribution(demo_df):
    text = format_distribution(demo_df, "region")
    assert "North" in text
    assert "█" in text
    assert "%" in text


# ── column histograms (source vs sample) ─────────────────────────────────────

def test_column_histogram_data_numeric_shares_bins(demo_df):
    result = sample(demo_df, 200, random_state=1)
    data = column_histogram_data(demo_df, result.data)
    by_name = {d["name"]: d for d in data}
    score = by_name["score"]
    assert score["kind"] == "numeric"
    # 10 bins by default, labels aligned to counts, both series same length
    assert len(score["labels"]) == 10
    assert len(score["source_counts"]) == len(score["sample_counts"]) == 10
    # numeric counts total the (finite) non-null values in each frame
    assert sum(score["source_counts"]) == demo_df["score"].notna().sum()
    assert sum(score["sample_counts"]) == len(result.data)


def test_column_histogram_data_categorical_uses_source_top(demo_df):
    result = sample(demo_df, 200, random_state=1)
    data = column_histogram_data(demo_df, result.data)
    region = next(d for d in data if d["name"] == "region")
    assert region["kind"] == "categorical"
    # labels are the source's categories, present in the sample too
    assert set(region["labels"]) <= set(demo_df["region"].dropna().astype(str))
    # percentages are within 0..100
    assert all(0 <= p <= 100 for p in region["source_pct"])
    assert all(0 <= p <= 100 for p in region["sample_pct"])


def test_column_histogram_data_handles_all_nan_and_missing_cols():
    src = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": ["x", "y", "z"]})
    samp = pd.DataFrame({"a": [1.0], "b": ["x"]})
    data = column_histogram_data(src, samp)
    names = {d["name"] for d in data}
    assert names == {"a", "b"}
    # a source column absent from the sample is simply skipped
    data2 = column_histogram_data(src, samp[["a"]])
    assert {d["name"] for d in data2} == {"a"}


def test_format_column_histograms_text(demo_df):
    result = sample(demo_df, 200, random_state=1)
    text = format_column_histograms(demo_df, result.data)
    assert "COLUMN DISTRIBUTIONS" in text
    assert "score" in text and "region" in text
    assert "src" in text and "sam" in text
    assert "█" in text and "%" in text


def test_format_column_histograms_empty_when_no_columns():
    empty = pd.DataFrame()
    assert format_column_histograms(empty, empty) == ""


def test_column_histogram_inf_does_not_deflate_percentages():
    # ±inf is excluded from the bars, so it must be excluded from the
    # denominator too — percentages of the finite values sum to ~100
    src = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.inf, -np.inf]})
    samp = src.head(4)
    d = next(iter(column_histogram_data(src, samp)))
    assert abs(sum(d["source_pct"]) - 100.0) < 1e-6
    assert abs(sum(d["sample_pct"]) - 100.0) < 1e-6


def test_column_histogram_skips_near_unique_columns(demo_df):
    result = sample(demo_df, 200, random_state=1)
    names = {d["name"] for d in column_histogram_data(demo_df, result.data)}
    assert "name" not in names  # 1000 unique of 1000: no meaningful top-8
    assert "region" in names and "score" in names  # real distributions kept
