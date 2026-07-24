"""Grouped name library + gender/ethnicity-aware NameAnonymizer + custom lib."""

import random

import pandas as pd
import pytest

from data_sampler import _names
from data_sampler.anonymize import (
    NameAnonymizer,
    anonymize,
    suggest_ethnicity_mapping,
    suggest_gender_mapping,
)


# ── the library itself ──────────────────────────────────────────────────────────

def test_library_structure_and_coverage():
    assert isinstance(_names.FIRST_NAMES, dict)
    assert isinstance(_names.LAST_NAMES, dict)
    # broad coverage: many ethnicities, both genders per group
    assert len(_names.ETHNICITIES) >= 25
    for eth in ("chinese", "anglo", "russian", "ethiopian", "persian"):
        assert eth in _names.ETHNICITIES
    # every first-name group is non-empty and grouped male/female
    males = [k for k in _names.FIRST_NAMES if k.endswith("_male")]
    females = [k for k in _names.FIRST_NAMES if k.endswith("_female")]
    assert males and females and len(males) == len(females)
    assert all(_names.FIRST_NAMES[k] for k in _names.FIRST_NAMES)


def test_first_names_respect_gender_and_ethnicity():
    male = set(_names.first_names("male", "chinese"))
    female = set(_names.first_names("female", "chinese"))
    assert male and female and male.isdisjoint(female)
    # family prefix pulls every indian_* subgroup
    indian = _names.first_names("male", "indian")
    assert len(indian) > len(_names.first_names("male", "indian_tamil"))


def test_gendered_surnames():
    # Russian surnames are grammatically gendered, no unisex base
    assert "Ivanov" in _names.last_names("male", "russian")
    assert "Ivanova" in _names.last_names("female", "russian")
    assert "Ivanova" not in _names.last_names("male", "russian")
    # North-Indian Hindu: unisex base + distinctly-female additions
    fem = _names.last_names("female", "indian_north_hindu")
    assert "Devi" in fem and "Kumari" in fem
    assert "Devi" not in _names.last_names("male", "indian_north_hindu")
    assert "Sharma" in _names.last_names("male", "indian_north_hindu")


def test_unknown_filter_falls_back_to_full_pool():
    assert _names.first_names(ethnicity="atlantis") == _names.ALL_FIRST


# ── fixed gender / ethnicity on the anonymizer ──────────────────────────────────

def test_fixed_gender_and_ethnicity():
    s = pd.Series([f"orig{i}" for i in range(20)])
    out = NameAnonymizer(gender="female", ethnicity="chinese").transform(s, random.Random(0))
    pool = set(_names.first_names("female", "chinese"))
    assert all(full.split(" ")[0] in pool for full in out)


def test_third_gender_keeps_ethnicity_mixes_gender():
    s = pd.Series([f"o{i}" for i in range(40)])
    out = NameAnonymizer(gender="third", ethnicity="italian").transform(s, random.Random(1))
    italian = set(_names.first_names(None, "italian"))
    assert all(full.split(" ")[0] in italian for full in out)


def test_undisclosed_allows_any_ethnicity():
    s = pd.Series([f"o{i}" for i in range(30)])
    out = NameAnonymizer(gender="undisclosed", ethnicity="chinese").transform(s, random.Random(2))
    # ethnicity constraint is intentionally dropped → global pool
    assert all(full.split(" ")[0] in set(_names.ALL_FIRST) for full in out)


def test_bad_gender_rejected():
    with pytest.raises(ValueError, match="gender"):
        NameAnonymizer(gender="nope")


# ── gender/ethnicity read from another column ───────────────────────────────────

@pytest.fixture
def gendered_df():
    return pd.DataFrame(
        {
            "name": ["Ann", "Bob", "Cy", "Dee", "Ann"],
            "sex": ["F", "M", "male", "female", "F"],
            "origin": ["Chinese", "Chinese", "Italian", "Italian", "Chinese"],
        }
    )


def test_gender_column_maps_per_row(gendered_df):
    out = anonymize(gendered_df, {"name": NameAnonymizer(gender_column="sex")}, seed=3)
    male_pool = set(_names.first_names("male"))
    female_pool = set(_names.first_names("female"))
    # Bob (M) → a male first name; Ann (F) → a female first name
    assert out["name"].iloc[1].split(" ")[0] in male_pool
    assert out["name"].iloc[0].split(" ")[0] in female_pool
    # consistent mapping preserved (row 0 and 4 were both "Ann")
    assert out["name"].iloc[0] == out["name"].iloc[4]


def test_ethnicity_column_maps_per_row(gendered_df):
    out = anonymize(gendered_df, {"name": NameAnonymizer(ethnicity_column="origin")}, seed=4)
    chinese = set(_names.first_names(None, "chinese"))
    italian = set(_names.first_names(None, "italian"))
    assert out["name"].iloc[0].split(" ")[0] in chinese   # Ann → Chinese origin
    assert out["name"].iloc[2].split(" ")[0] in italian   # Cy  → Italian origin


def test_gender_map_override():
    df = pd.DataFrame({"name": ["a", "b"], "g": ["1", "2"]})
    # override the ISO-5218 default (1=male, 2=female) with a custom map
    anon = NameAnonymizer(gender_column="g", gender_map={"1": "female", "2": "male"})
    out = anonymize(df, {"name": anon}, seed=5)
    assert out["name"].iloc[0].split(" ")[0] in set(_names.first_names("female"))
    assert out["name"].iloc[1].split(" ")[0] in set(_names.first_names("male"))


def test_randomize_gender_rewrites_gender_column(gendered_df):
    anon = NameAnonymizer(gender_column="sex", randomize_gender=True)
    out = anonymize(gendered_df, {"name": anon}, seed=6)
    # the gender column is rewritten to canonical male/female labels...
    assert set(out["sex"].dropna().unique()) <= {"male", "female"}
    # ...and each name matches its (new) assigned gender
    for i in range(len(out)):
        g = out["sex"].iloc[i]
        if pd.isna(g):
            continue
        assert out["name"].iloc[i].split(" ")[0] in set(_names.first_names(g))


def test_linked_missing_column_raises(gendered_df):
    with pytest.raises(KeyError, match="ghost"):
        anonymize(gendered_df, {"name": NameAnonymizer(gender_column="ghost")})


# ── auto-detection helpers ──────────────────────────────────────────────────────

def test_suggest_gender_mapping():
    m = suggest_gender_mapping(["M", "F", "male", "female", "x", "", "wat"])
    assert m["M"] == "male" and m["F"] == "female"
    assert m["male"] == "male" and m["female"] == "female"
    assert m["x"] == "third"
    assert m[""] == "undisclosed"
    assert m["wat"] is None  # unrecognized → manual mapping needed


def test_suggest_ethnicity_mapping():
    m = suggest_ethnicity_mapping(["Chinese", "Italian", "indian", "Klingon"])
    assert m["Chinese"] == "chinese"
    assert m["Italian"] == "italian"
    assert m["indian"] == "indian"        # family prefix
    assert m["Klingon"] is None


# ── custom library: export / load / round-trip ──────────────────────────────────

def test_export_library_is_valid_source():
    src = _names.export_library()
    assert "FIRST_NAMES" in src and "LAST_NAMES" in src
    ns: dict = {}
    exec(compile(src, "<exported>", "exec"), ns)
    assert isinstance(ns["FIRST_NAMES"], dict) and ns["FIRST_NAMES"]


def test_load_library_swaps_pools_then_restores():
    saved = _names.export_library()  # capture the default library first
    try:
        _names.load_library(source=(
            "FIRST_NAMES = {'testland_male': ('Zzyzx',), "
            "'testland_female': ('Qwerty',)}\n"
            "LAST_NAMES = {'testland': ('Xylophone',)}\n"
        ))
        assert _names.first_names("male", "testland") == ("Zzyzx",)
        out = NameAnonymizer(ethnicity="testland").transform(
            pd.Series(["a", "b"]), random.Random(0)
        )
        assert all("Xylophone" in v for v in out)
    finally:
        _names.load_library(source=saved)  # restore the default library
    assert len(_names.ETHNICITIES) >= 25  # default library is back


def test_load_library_rejects_invalid():
    with pytest.raises(ValueError):
        _names.load_library(source="FIRST_NAMES = 'not a dict'\nLAST_NAMES = {}\n")
