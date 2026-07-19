from data_sampler.report import format_distribution, format_stratification_report
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
