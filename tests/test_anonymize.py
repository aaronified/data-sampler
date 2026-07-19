import random

import numpy as np
import pandas as pd
import pytest

from data_sampler import _names
from data_sampler.anonymize import (
    NameAnonymizer,
    NumericJitterAnonymizer,
    RandomStringAnonymizer,
    SequentialIdAnonymizer,
    anonymize,
    make_anonymizer,
)


@pytest.fixture
def people_df():
    return pd.DataFrame(
        {
            "name": ["Ann Lee", "Bob Roy", "Ann Lee", "Cy Wu", None],
            "cust_id": ["C1", "C2", "C1", "C3", "C4"],
            "salary": [50_000, 60_000, 50_000, 75_000, 80_000],
            "email": ["a@x.com", "b@x.com", "a@x.com", "c@x.com", "d@x.com"],
        }
    )


# ── consistency: the core statistical guarantee ──────────────────────────────

@pytest.mark.parametrize(
    "kind,col",
    [("names", "name"), ("sequential_id", "cust_id"), ("numeric_jitter", "salary"), ("random_string", "email")],
)
def test_repeated_values_map_to_same_replacement(people_df, kind, col):
    out = anonymize(people_df, {col: kind}, seed=1)
    # rows 0 and 2 were identical originally → must stay identical
    assert out[col].iloc[0] == out[col].iloc[2]
    # distinct originals stay distinct
    assert out[col].iloc[0] != out[col].iloc[1]
    # value-count histogram (the distribution) is preserved exactly
    assert sorted(people_df[col].value_counts().values) == sorted(
        out[col].value_counts().values
    )


def test_nan_preserved(people_df):
    out = anonymize(people_df, {"name": "names"}, seed=1)
    assert out["name"].isna().iloc[4]
    assert out["name"].notna().iloc[:4].all()


def test_unnamed_columns_untouched(people_df):
    out = anonymize(people_df, {"name": "names"}, seed=1)
    pd.testing.assert_series_equal(out["salary"], people_df["salary"])
    pd.testing.assert_series_equal(out["email"], people_df["email"])


def test_source_frame_not_mutated(people_df):
    before = people_df.copy()
    anonymize(people_df, {"name": "names", "salary": "jitter"}, seed=1)
    pd.testing.assert_frame_equal(before, people_df)


def test_seed_reproducibility(people_df):
    a = anonymize(people_df, {"name": "names", "salary": "jitter"}, seed=99)
    b = anonymize(people_df, {"name": "names", "salary": "jitter"}, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_unknown_column_raises(people_df):
    with pytest.raises(KeyError, match="ghost"):
        anonymize(people_df, {"ghost": "names"})


# ── names ─────────────────────────────────────────────────────────────────────

def test_names_come_from_library():
    series = pd.Series([f"orig{i}" for i in range(50)])
    out = NameAnonymizer().transform(series, random.Random(0))
    for full in out:
        first, last = full.split(" ")
        assert first in _names.FIRST_NAMES
        assert last in _names.LAST_NAMES


def test_name_styles():
    rng = random.Random(0)
    series = pd.Series(["a", "b", "c"])
    fml = NameAnonymizer("first_middle_last").transform(series, rng)
    assert all(len(v.split(" ")) == 3 for v in fml)
    lf = NameAnonymizer("last_first").transform(series, random.Random(0))
    assert all(", " in v for v in lf)
    with pytest.raises(ValueError, match="style"):
        NameAnonymizer("nope")


def test_names_unique_even_beyond_style_capacity():
    # 300 uniques > capacity("first")=200 → auto-escalates, all still unique
    series = pd.Series([f"v{i}" for i in range(300)])
    out = NameAnonymizer("first").transform(series, random.Random(0))
    assert out.nunique() == 300


def test_names_huge_column_stays_unique():
    series = pd.Series([f"v{i}" for i in range(30_000)])
    out = NameAnonymizer().transform(series, random.Random(0))
    assert out.nunique() == 30_000


# ── sequential id ─────────────────────────────────────────────────────────────

def test_sequential_start_and_interval(people_df):
    out = anonymize(people_df, {"cust_id": ("sequential_id", {"start": 1000, "interval": 7})}, seed=1)
    # C1, C2, C3, C4 in order of first appearance → 1000, 1007, 1014, 1021
    assert out["cust_id"].tolist() == [1000, 1007, 1000, 1014, 1021]


def test_sequential_prefix_and_width():
    series = pd.Series(["a", "b", "a"])
    out = SequentialIdAnonymizer(start=5, prefix="ID-", width=4).transform(series)
    assert out.tolist() == ["ID-0005", "ID-0006", "ID-0005"]


def test_sequential_negative_interval():
    series = pd.Series(["a", "b", "c"])
    out = SequentialIdAnonymizer(start=10, interval=-2).transform(series)
    assert out.tolist() == [10, 8, 6]


def test_sequential_zero_interval_rejected():
    with pytest.raises(ValueError, match="interval"):
        SequentialIdAnonymizer(interval=0)


def test_sequential_with_nan_uses_nullable_int():
    series = pd.Series(["a", None, "b"])
    out = SequentialIdAnonymizer().transform(series)
    assert out.isna().iloc[1]
    assert out.iloc[0] == 1 and out.iloc[2] == 2


# ── numeric jitter ────────────────────────────────────────────────────────────

def test_jitter_within_20_pct():
    values = pd.Series([100.0, -50.0, 3.14, 1e9])
    out = NumericJitterAnonymizer().transform(values, random.Random(0))
    for orig, new in zip(values, out):
        assert abs(new - orig) <= abs(orig) * 0.2 + 1e-9
        assert new != orig or orig == 0


def test_jitter_custom_pct():
    values = pd.Series(np.linspace(10, 1000, 200))
    out = NumericJitterAnonymizer(pct=0.05).transform(values, random.Random(1))
    ratio = (out / values).abs()
    assert ((ratio >= 0.95) & (ratio <= 1.05)).all()


def test_jitter_integers_stay_integers():
    series = pd.Series([100, 200, 300])
    out = NumericJitterAnonymizer().transform(series, random.Random(0))
    assert all(float(v) == int(v) for v in out)


def test_jitter_zero_unchanged():
    out = NumericJitterAnonymizer().transform(pd.Series([0.0, 10.0]), random.Random(0))
    assert out.iloc[0] == 0.0


def test_jitter_round_to():
    out = NumericJitterAnonymizer(round_to=2).transform(
        pd.Series([3.14159, 2.71828]), random.Random(0)
    )
    assert all(round(v, 2) == v for v in out)


def test_jitter_rejects_non_numeric():
    with pytest.raises(TypeError, match="numeric"):
        NumericJitterAnonymizer().transform(pd.Series(["a", "b"]))


def test_jitter_rejects_bad_pct():
    with pytest.raises(ValueError):
        NumericJitterAnonymizer(pct=0)
    with pytest.raises(ValueError):
        NumericJitterAnonymizer(pct=1.5)


# ── random string / hex ──────────────────────────────────────────────────────

def test_random_string_length_charset_prefix():
    series = pd.Series([f"v{i}" for i in range(100)])
    out = RandomStringAnonymizer(length=12, prefix="tok_").transform(series, random.Random(0))
    assert out.nunique() == 100
    for v in out:
        assert v.startswith("tok_") and len(v) == 16
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789" for c in v[4:])


def test_hex_kind():
    series = pd.Series(["x", "y", "z"])
    out = make_anonymizer("hex", length=10).transform(series, random.Random(0))
    for v in out:
        assert len(v) == 10
        int(v, 16)  # must parse as hexadecimal


def test_random_string_rejects_bad_options():
    with pytest.raises(ValueError, match="charset"):
        RandomStringAnonymizer(charset="klingon")
    with pytest.raises(ValueError, match="length"):
        RandomStringAnonymizer(length=0)


def test_short_digit_strings_stay_unique_via_suffix_fallback():
    # 15 uniques but only 10 possible 1-char digit strings → suffix fallback
    series = pd.Series([f"v{i}" for i in range(15)])
    out = RandomStringAnonymizer(length=1, charset="digits").transform(
        series, random.Random(0)
    )
    assert out.nunique() == 15


# ── spec coercion ────────────────────────────────────────────────────────────

def test_spec_forms_equivalent(people_df):
    via_str = anonymize(people_df, {"email": "hex"}, seed=5)
    via_tuple = anonymize(people_df, {"email": ("hex", {})}, seed=5)
    via_dict = anonymize(people_df, {"email": {"kind": "hex"}}, seed=5)
    via_instance = anonymize(
        people_df, {"email": RandomStringAnonymizer(charset="hex")}, seed=5
    )
    pd.testing.assert_frame_equal(via_str, via_tuple)
    pd.testing.assert_frame_equal(via_str, via_dict)
    pd.testing.assert_frame_equal(via_str, via_instance)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown anonymizer kind"):
        make_anonymizer("rot13")


def test_bad_spec_type_raises(people_df):
    with pytest.raises(TypeError):
        anonymize(people_df, {"name": 42})


def test_multiple_columns_anonymized_together(people_df):
    out = anonymize(
        people_df,
        {
            "name": "names",
            "cust_id": ("seq", {"start": 100, "interval": 10}),
            "salary": "numbers",
            "email": {"kind": "hex", "length": 16},
        },
        seed=7,
    )
    assert out["name"].iloc[0] != "Ann Lee"
    assert out["cust_id"].iloc[0] == 100
    assert out["salary"].iloc[0] != 50_000
    assert len(out["email"].iloc[0]) == 16
