"""Tests for the guided anonymization workflow (data_sampler.workflow)."""

import pandas as pd
import pytest

import data_sampler as ds
from data_sampler.anonymize import (
    DatetimeJitterAnonymizer,
    NumericJitterAnonymizer,
    anonymize,
)
from data_sampler.stats import compute_column_stats
from data_sampler.workflow import (
    TYPE_OPTIONS,
    AnonymizationPlan,
    suggest_type,
)


@pytest.fixture
def typed_df():
    """A frame with one column of each semantic shape suggest_type keys on."""
    n = 12
    return pd.DataFrame(
        {
            "signup_date": pd.to_datetime(
                [f"2020-{(i % 12) + 1:02d}-01" for i in range(n)]
            ),
            "email": [f"user{i}@example.com" for i in range(n)],
            "full_name": [f"Person Number {i}" for i in range(n)],
            "customer_id": range(1000, 1000 + n),  # numeric, all unique
            "user_code": [f"U{i:04d}" for i in range(n)],  # string id, all unique
            "salary": [50_000 + i * 1000 for i in range(n)],  # numeric, not id
            "region": (["North", "South", "East"] * n)[:n],  # categorical
            "active": [i % 2 == 0 for i in range(n)],  # boolean
            "notes": ["a long free-text note " * 5] * n,  # text (avg len > 50)
        }
    )


# ── suggest_type heuristics ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "col,expected",
    [
        ("signup_date", "datetime_jitter"),
        ("email", "hex"),
        ("full_name", "names"),
        ("customer_id", "sequential_id"),
        ("user_code", "sequential_id"),
        ("salary", "numeric_jitter"),
        ("region", "none"),
        ("active", "none"),
        ("notes", "random_string"),
    ],
)
def test_suggest_type_heuristics(typed_df, col, expected):
    stats = compute_column_stats(typed_df[col], len(typed_df))
    assert suggest_type(stats) == expected


@pytest.mark.parametrize(
    "name,values,expected",
    [
        # date-like NAMES on string columns (CSV loads dates as strings) → date jitter
        ("signup_date", [f"2021-01-0{i % 9 + 1}" for i in range(10)], "datetime_jitter"),
        ("created_at", [f"2021-02-0{i % 9 + 1}" for i in range(10)], "datetime_jitter"),
        ("date_of_birth", [f"1990-0{i % 9 + 1}-01" for i in range(10)], "datetime_jitter"),
        # names that merely CONTAIN a date word must NOT be treated as dates
        ("candidate_name", [f"Person {i}" for i in range(10)], "names"),  # name hint wins
        ("mandate_ref", [f"M{i}" for i in range(10)], "none"),  # no date token, no id hint
        ("update_reason", [f"reason {i}" for i in range(10)], "none"),
    ],
)
def test_suggest_type_date_name_hints_and_false_positives(name, values, expected):
    stats = compute_column_stats(pd.Series(values, name=name), len(values))
    assert suggest_type(stats) == expected


def test_suggest_type_date_named_numeric_stays_numeric():
    # a numeric column named like a date (e.g. a birth *year*) must not be
    # treated as a datetime — jittering the integer is right, date-parsing wrong
    stats = compute_column_stats(pd.Series(range(1980, 2000), name="birth_year"), 20)
    assert suggest_type(stats) == "numeric_jitter"


def test_suggest_type_numeric_without_id_hint_is_jitter(typed_df):
    # a numeric column that is all-unique but has no id-ish name stays jitter
    stats = compute_column_stats(typed_df["salary"], len(typed_df))
    assert stats.unique_pct == 100.0
    assert suggest_type(stats) == "numeric_jitter"


# ── plan construction / editing (pre-specify through a function) ──────────────

def test_for_columns_is_all_none(typed_df):
    plan = AnonymizationPlan.for_columns(typed_df.columns)
    assert set(plan.assignments) == set(typed_df.columns)
    assert all(k == "none" for k, _ in plan.assignments.values())
    assert plan.active_columns() == []
    assert plan.build_spec() == {}


def test_assign_validates_and_chains():
    plan = AnonymizationPlan()
    result = plan.assign("a", "numeric_jitter", pct=0.1).assign("b", "hex", length=12)
    assert result is plan  # chainable
    assert plan.type_of("a") == "numeric_jitter"
    spec = plan.build_spec()
    assert isinstance(spec["a"], NumericJitterAnonymizer)
    assert abs(spec["a"].pct - 0.1) < 1e-9


def test_assign_bad_kind_raises():
    with pytest.raises(ValueError, match="Unknown anonymizer kind"):
        AnonymizationPlan().assign("a", "rot13")


def test_assign_bad_option_raises():
    with pytest.raises(ValueError):
        AnonymizationPlan().assign("a", "numeric_jitter", pct=5)  # pct must be <1


def test_assign_none_clears():
    plan = AnonymizationPlan().assign("a", "hex")
    plan.assign("a", "none")
    assert plan.type_of("a") == "none"
    assert "a" not in plan.build_spec()


def test_suggest_builds_full_plan(typed_df):
    plan = AnonymizationPlan.suggest(typed_df)
    assert plan.type_of("signup_date") == "datetime_jitter"
    assert plan.type_of("region") == "none"
    spec = plan.build_spec()
    assert "region" not in spec  # none columns omitted
    assert isinstance(spec["signup_date"], DatetimeJitterAnonymizer)


def test_suggest_respects_columns_subset(typed_df):
    plan = AnonymizationPlan.suggest(typed_df, columns=["salary", "email"])
    assert set(plan.assignments) == {"salary", "email"}


# ── apply ────────────────────────────────────────────────────────────────────

def test_apply_matches_anonymize_directly(typed_df):
    plan = AnonymizationPlan().assign("salary", "numeric_jitter", pct=0.1)
    out_plan = plan.apply(typed_df, seed=7)
    out_direct = anonymize(typed_df, plan.build_spec(), seed=7)
    pd.testing.assert_frame_equal(out_plan, out_direct)


def test_apply_no_active_columns_is_copy(typed_df):
    plan = AnonymizationPlan.for_columns(typed_df.columns)
    out = plan.apply(typed_df, seed=1)
    pd.testing.assert_frame_equal(out, typed_df)


def test_summary_lists_active(typed_df):
    plan = AnonymizationPlan().assign("full_name", "names")
    assert "full_name" in plan.summary()
    assert AnonymizationPlan.for_columns(["x"]).summary() == "no columns anonymized"


# ── interactive (choose from options) ────────────────────────────────────────

class FakePrompt:
    """A stand-in for input(): returns queued answers, records the prompts."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, message: str) -> str:
        self.prompts.append(message)
        return self._answers.pop(0) if self._answers else ""


def test_choose_interactively_enter_accepts_suggestion(typed_df):
    df = typed_df[["salary", "signup_date"]]
    prompt = FakePrompt(["", ""])  # accept the default for both
    lines: list[str] = []
    plan = AnonymizationPlan.for_columns(df.columns).choose_interactively(
        df, prompt=prompt, echo=lines.append
    )
    assert plan.type_of("salary") == "numeric_jitter"  # suggested default
    assert plan.type_of("signup_date") == "datetime_jitter"
    # menu was actually shown
    assert any("datetime jitter" in line for line in lines)
    assert any("[suggested]" in line for line in lines)


def test_choose_interactively_numeric_choice_overrides(typed_df):
    df = typed_df[["salary"]]
    # option 1 is always "none"
    prompt = FakePrompt(["1"])
    plan = AnonymizationPlan.for_columns(df.columns).choose_interactively(
        df, prompt=prompt, echo=lambda _l: None
    )
    assert plan.type_of("salary") == "none"


def test_choose_interactively_out_of_range_falls_back_to_default(typed_df):
    df = typed_df[["salary"]]
    prompt = FakePrompt(["99"])  # invalid → keep suggested numeric_jitter
    plan = AnonymizationPlan.for_columns(df.columns).choose_interactively(
        df, prompt=prompt, echo=lambda _l: None
    )
    assert plan.type_of("salary") == "numeric_jitter"


def test_choose_interactively_non_numeric_falls_back(typed_df):
    df = typed_df[["salary"]]
    prompt = FakePrompt(["banana"])
    plan = AnonymizationPlan.for_columns(df.columns).choose_interactively(
        df, prompt=prompt, echo=lambda _l: None
    )
    assert plan.type_of("salary") == "numeric_jitter"


def test_choose_interactively_seeded_assignment_is_default(typed_df):
    df = typed_df[["salary"]]
    prompt = FakePrompt([""])  # accept default
    plan = AnonymizationPlan.for_columns(df.columns)
    plan.assign("salary", "hex")  # pre-seed (e.g. from CLI --anon)
    plan.choose_interactively(df, prompt=prompt, echo=lambda _l: None)
    assert plan.type_of("salary") == "hex"  # seeded value wins as default


def test_choose_interactively_without_df_uses_menu_only():
    prompt = FakePrompt(["2"])  # option 2 is "names"
    plan = AnonymizationPlan.for_columns(["x"]).choose_interactively(
        prompt=prompt, echo=lambda _l: None
    )
    assert plan.type_of("x") == "names"


# ── audit regressions (v3.2 pre-release audit) ───────────────────────────────

def test_assign_canonicalizes_alias_kinds():
    plan = AnonymizationPlan().assign("c", "seq", start=100)
    assert plan.type_of("c") == "sequential_id"  # not the raw alias
    for alias, canonical in (
        ("name", "names"), ("jitter", "numeric_jitter"),
        ("dates", "datetime_jitter"), ("string", "random_string"),
    ):
        assert AnonymizationPlan().assign("c", alias).type_of("c") == canonical


def test_wizard_enter_keeps_alias_seeded_assignment():
    # an --anon seeded with an alias kind must survive pressing Enter
    prompt = FakePrompt([""])
    plan = AnonymizationPlan.for_columns(["user_id"])
    plan.assign("user_id", "seq", start=100)
    plan.choose_interactively(columns=["user_id"], prompt=prompt, echo=lambda _l: None)
    assert plan.type_of("user_id") == "sequential_id"
    assert plan.assignments["user_id"][1] == {"start": 100}


def test_wizard_accept_preserves_options_change_resets():
    plan = AnonymizationPlan.for_columns(["salary"])
    plan.assign("salary", "numeric_jitter", pct=0.05)
    # Enter (accept current kind) keeps the seeded options
    plan.choose_interactively(
        columns=["salary"], prompt=FakePrompt([""]), echo=lambda _l: None
    )
    assert plan.assignments["salary"] == ("numeric_jitter", {"pct": 0.05})
    # choosing a DIFFERENT kind starts from that kind's defaults
    plan.choose_interactively(
        columns=["salary"], prompt=FakePrompt(["7"]), echo=lambda _l: None
    )  # option 7 = hex
    assert plan.assignments["salary"] == ("hex", {})


def test_suggest_type_never_jitters_other_kind():
    # TIME / INTERVAL / nested columns surface as kind "other" from the
    # engine; even a date-ish NAME must not suggest datetime_jitter for them
    from data_sampler.stats import ColumnStats

    cs = ColumnStats(
        name="start_time", dtype="TIME", kind="other",
        count=100, missing=0, missing_pct=0.0, unique=50, unique_pct=50.0,
    )
    assert suggest_type(cs) == "none"


# ── public API surface ───────────────────────────────────────────────────────

def test_public_api_exposes_workflow():
    assert ds.AnonymizationPlan is AnonymizationPlan
    assert ds.suggest_type is suggest_type
    assert ds.TYPE_OPTIONS == TYPE_OPTIONS
    assert TYPE_OPTIONS[0][0] == "none"
