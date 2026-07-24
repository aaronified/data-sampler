"""Tests for the DuckDB out-of-core engine (data_sampler.engine).

Skipped entirely if the optional 'large' extra (duckdb) is not installed.
"""

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from data_sampler.engine import (  # noqa: E402
    DuckDBEngine,
    _is_remote,
    _proportional_allocation,
    _remote_ext,
    duckdb_available,
    large_materialization_warning,
    should_use_engine,
    sample as engine_sample,
    stats as engine_stats,
)


def test_remote_detection_and_ext():
    assert _is_remote("https://host/data.parquet")
    assert _is_remote("s3://bucket/key.parquet")
    assert not _is_remote("/tmp/local.parquet")
    assert not _is_remote(pd.DataFrame())
    assert _remote_ext("https://host/dir/data.parquet?token=xyz") == ".parquet"
    assert _remote_ext("https://host/dir/data.csv") == ".csv"


def test_should_use_engine_for_remote_parquet():
    # remote Parquet auto-selects the engine (range requests); remote CSV doesn't
    assert should_use_engine("https://host/big.parquet") is True
    assert should_use_engine("https://host/big.csv") is False


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


# ── approximate stats ─────────────────────────────────────────────────────────

def test_stats_kinds_and_shape(engine, big_df):
    stats = {s.name: s for s in engine.stats(big_df)}
    assert set(stats) == set(big_df.columns)
    assert stats["id"].kind == "numeric"
    assert stats["region"].kind == "categorical"
    assert stats["score"].kind == "numeric"
    assert stats["note"].kind == "text"  # long free text
    # all marked approximate
    assert all(s.approximate for s in stats.values())


def test_stats_counts_and_numeric(engine, big_df):
    stats = {s.name: s for s in engine.stats(big_df)}
    assert stats["id"].count == 20_000 and stats["id"].missing == 0
    score = stats["score"]
    assert score.min is not None and score.max is not None
    assert score.min <= score.median <= score.max
    assert score.mean is not None
    # numeric histogram has the requested bins
    assert len(score.histogram) == 10
    # categorical top-values populated
    assert stats["region"].top_values
    assert sum(c for _, c in stats["region"].top_values) <= 20_000


def test_stats_approx_distinct_close_to_exact(engine, big_df):
    approx = {s.name: s for s in engine.stats(big_df)}
    exact = {s.name: s for s in engine.stats(big_df, approximate=False)}
    # small-cardinality categorical is exact even under HLL
    assert exact["region"].unique == big_df["region"].nunique()
    assert approx["region"].unique == big_df["region"].nunique()
    # high-cardinality id: HLL trades accuracy for speed — ballpark, not exact
    # (deterministic for given data; generous bound for HyperLogLog error)
    assert abs(approx["id"].unique - 20_000) / 20_000 < 0.15
    assert exact["id"].unique == 20_000  # exact mode is exact
    assert exact["region"].approximate is False


def test_stats_missing_counted(engine):
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, None], "y": ["a", "a", "b", None, "a"]})
    stats = {s.name: s for s in engine.stats(df)}
    assert stats["x"].missing == 2 and stats["x"].count == 3
    assert stats["y"].missing == 1


def test_stats_distributions_false_skips_per_column_passes(engine, big_df):
    stats = {s.name: s for s in engine.stats(big_df, distributions=False)}
    # scalar fields still present, but no histograms / top-values computed
    assert stats["id"].count == 20_000
    assert stats["region"].histogram == []
    assert stats["score"].histogram == []


def test_module_level_stats_helper(big_df):
    stats = engine_stats(big_df)
    assert any(s.name == "region" and s.kind == "categorical" for s in stats)


def test_stats_datetime_and_single_value(engine):
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-06-15", "2021-12-31"]),
            "const": ["only", "only", "only", "only"],
        }
    )
    stats = {s.name: s for s in engine.stats(df)}
    assert stats["when"].kind == "datetime"
    assert stats["when"].top_values  # datetime gets top-values too
    assert stats["const"].unique == 1
    # a constant column has no numeric spread; no crash, top-value present
    assert stats["const"].top_values[0][0] == "only"


def test_stats_empty_frame_no_crash(engine):
    df = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="object")})
    stats = {s.name: s for s in engine.stats(df)}
    assert stats["a"].count == 0 and stats["a"].missing == 0
    assert stats["a"].histogram == []


# ── audit regressions (v3.2 pre-release audit) ───────────────────────────────

def test_duckdb_kind_classification():
    from data_sampler.engine import _duckdb_kind

    assert _duckdb_kind("INTEGER[]") == "other"          # list, not numeric
    assert _duckdb_kind("DECIMAL(6,2)[]") == "other"
    assert _duckdb_kind("STRUCT(a INTEGER)") == "other"
    assert _duckdb_kind("MAP(VARCHAR, INTEGER)") == "other"
    assert _duckdb_kind("TIME") == "other"               # not datetime-jitterable
    assert _duckdb_kind("INTERVAL") == "other"           # not numeric ("int" prefix!)
    assert _duckdb_kind("TIMESTAMP WITH TIME ZONE") == "datetime"
    assert _duckdb_kind("DATE") == "datetime"
    assert _duckdb_kind("ENUM('a', 'b')") == "categorical"
    assert _duckdb_kind("DECIMAL(6,2)") == "numeric"
    assert _duckdb_kind("BOOLEAN") == "boolean"


def test_stats_survives_list_column_and_nan_float(tmp_path, engine):
    # LIST column must not poison the shared aggregate query, and a float
    # column containing real NaN VALUES (Parquet keeps NaN, unlike pandas
    # registration which nulls it) must not crash stddev
    df = pd.DataFrame(
        {
            "lst": [[1, 2], [3], [4, 5], []],
            "x": [1.0, float("nan"), 3.0, 5.0],
            "label": ["a", "b", "a", "c"],
        }
    )
    p = tmp_path / "nan_list.parquet"
    df.to_parquet(p, index=False)
    stats = {s.name: s for s in engine.stats(str(p))}
    assert stats["lst"].kind == "other"
    x = stats["x"]
    assert x.kind == "numeric"
    # aggregates computed over the finite values only
    assert x.min == 1.0 and x.max == 5.0
    assert x.std is not None
    assert stats["label"].top_values


def test_stats_unknown_column_raises(engine, big_df):
    with pytest.raises(KeyError, match="ghost"):
        engine.stats(big_df, columns=["ghost"])


def test_stats_unique_never_exceeds_count(engine, big_df):
    for s in engine.stats(big_df):
        assert s.unique <= max(s.count, 0)


def test_stratified_seeded_deterministic_under_remainder_ties():
    # 10 equal strata sampled to 105: every stratum's remainder ties at 0.5,
    # so the +1 slots are tie-broken — the stratum order must be pinned or
    # even seeded runs differ
    df = pd.DataFrame(
        {
            "stratum": np.repeat([f"s{i:02d}" for i in range(10)], 100),
            "id": range(1000),
        }
    )
    with DuckDBEngine(threads=4) as e1:
        a = e1.sample(df, 105, strat_cols=["stratum"], seed=5)
    with DuckDBEngine(threads=4) as e2:
        b = e2.sample(df, 105, strat_cols=["stratum"], seed=5)
    assert len(a.data) == len(b.data) == 105
    assert sorted(a.data["id"].tolist()) == sorted(b.data["id"].tolist())
    pd.testing.assert_series_equal(
        a.allocations.sort_index(), b.allocations.sort_index()
    )


def test_columns_oriented_json_rejected(tmp_path, engine, big_df):
    # pandas to_json() default (orient="columns") parses as ONE row of structs
    # in DuckDB — the engine must refuse instead of silently sampling garbage
    p = tmp_path / "cols.json"
    big_df.head(20).to_json(p)
    with pytest.raises(ValueError, match="columns-oriented"):
        engine.row_count(str(p))
    # records-oriented JSON stays fully supported
    p2 = tmp_path / "records.json"
    big_df.head(20).to_json(p2, orient="records")
    assert engine.row_count(str(p2)) == 20


def test_sample_all_rows_warns_on_large_source(engine, monkeypatch):
    import data_sampler.engine as eng

    monkeypatch.setattr(eng, "LARGE_ROW_THRESHOLD", 10)
    df = pd.DataFrame({"a": range(20)})
    result = engine.sample(df, 25)
    assert result.method == "all"
    assert any("WARNING" in n and "Materializing" in n for n in result.notes)


def test_row_count_cached_per_file(tmp_path, big_df):
    p = tmp_path / "counted.csv"
    big_df.to_csv(p, index=False)
    with DuckDBEngine(threads=2) as e:
        assert e.row_count(str(p)) == 20_000
        src = e._source_sql(str(p))
        assert e._count_cache[src] == 20_000  # cached
        # sample() reuses the cache instead of re-scanning
        result = e.sample(str(p), 50, use_random=True, seed=1)
        assert len(result.data) == 50


# ── two-phase narrow sampling (rank narrow, fetch winners) ───────────────────

def test_narrow_parquet_stratified_schema_count_distribution(tmp_path, engine, big_df):
    p = tmp_path / "narrow.parquet"
    big_df.to_parquet(p, index=False)
    result = engine.sample(str(p), 500, strat_cols=["region"], seed=2)
    assert len(result.data) == 500
    # full schema, original order, no file_row_number/_ds_rowid leak
    assert list(result.data.columns) == list(big_df.columns)
    src_p = big_df["region"].value_counts(normalize=True)
    smp_p = result.data["region"].value_counts(normalize=True)
    for cat in src_p.index:
        assert abs(src_p[cat] - smp_p.get(cat, 0)) < 0.05


def test_narrow_parquet_seeded_reproducible_across_engines(tmp_path, big_df):
    p = tmp_path / "repro.parquet"
    big_df.to_parquet(p, index=False)
    with DuckDBEngine(threads=4) as e1:
        s1 = e1.sample(str(p), 300, strat_cols=["region", "tier"], seed=9)
        r1 = e1.sample(str(p), 300, use_random=True, seed=9)
    with DuckDBEngine(threads=4) as e2:
        s2 = e2.sample(str(p), 300, strat_cols=["region", "tier"], seed=9)
        r2 = e2.sample(str(p), 300, use_random=True, seed=9)
    # stratified: identical rows AND order (fetch is ORDER BY row id)
    assert s1.data["id"].tolist() == s2.data["id"].tolist()
    assert sorted(r1.data["id"].tolist()) == sorted(r2.data["id"].tolist())


def test_narrow_nan_stratum_survives_both_sources(tmp_path, engine, big_df):
    df = big_df.copy()
    df.loc[df.index[:2000], "region"] = None  # 10% missing stratum
    p = tmp_path / "nanstrat.parquet"
    df.to_parquet(p, index=False)
    for source in (str(p), df):
        result = engine.sample(source, 1000, strat_cols=["region"], seed=1)
        assert len(result.data) == 1000
        # the missing-value stratum is sampled proportionally (~100 rows)
        nan_rows = int(result.data["region"].isna().sum())
        assert 60 <= nan_rows <= 140


def test_narrow_guard_reserved_column_names_fall_back(tmp_path, engine, big_df):
    # a real column named file_row_number must not break Parquet sampling
    decoy = big_df.copy()
    decoy["file_row_number"] = 1
    p = tmp_path / "decoy.parquet"
    decoy.to_parquet(p, index=False)
    result = engine.sample(str(p), 100, strat_cols=["region"], seed=1)
    assert len(result.data) == 100
    assert "file_row_number" in result.data.columns
    assert (result.data["file_row_number"] == 1).all()
    # same for a DataFrame column named _ds_rid
    decoy2 = big_df.copy()
    decoy2["_ds_rid"] = 7
    result2 = engine.sample(decoy2, 100, strat_cols=["region"], seed=1)
    assert len(result2.data) == 100
    assert (result2.data["_ds_rid"] == 7).all()


def test_narrow_hostile_strat_column_name(engine):
    hostile = 'my col"; DROP TABLE x;--'
    df = pd.DataFrame({hostile: (["a", "b", "c"] * 400)[:1000], "v": range(1000)})
    result = engine.sample(df, 100, strat_cols=[hostile], seed=1)
    assert len(result.data) == 100
    assert set(result.data.columns) == {hostile, "v"}


def test_narrow_dataframe_preserves_dtypes(engine):
    # the pandas take() fetch returns slices of the original frame, so exotic
    # dtypes (nullable string, categorical) survive exactly
    df = pd.DataFrame(
        {
            "s": pd.Series((["x", "y", None] * 400)[:1000], dtype="string"),
            "c": pd.Categorical((["g", "h"] * 500)[:1000]),
            "grp": (["a", "b"] * 500)[:1000],
        }
    )
    result = engine.sample(df, 100, strat_cols=["grp"], seed=3)
    assert isinstance(result.data["s"].dtype, pd.StringDtype)
    assert isinstance(result.data["c"].dtype, pd.CategoricalDtype)


def test_narrow_wide_parquet_correctness(tmp_path, engine):
    # 120 payload columns: all must come back, in order, exact count
    rng = np.random.default_rng(4)
    n = 5000
    wide = pd.DataFrame({f"c{i:03d}": rng.normal(size=n) for i in range(120)})
    wide.insert(0, "grp", rng.choice(["a", "b", "c", "d"], n))
    p = tmp_path / "wide.parquet"
    wide.to_parquet(p, index=False)
    result = engine.sample(str(p), 200, strat_cols=["grp"], seed=5)
    assert len(result.data) == 200
    assert list(result.data.columns) == list(wide.columns)
    reservoir = engine.sample(str(p), 200, use_random=True, seed=5)
    assert len(reservoir.data) == 200
    assert list(reservoir.data.columns) == list(wide.columns)


def test_narrow_glob_multi_file_parquet_falls_back_and_stays_exact(tmp_path, engine, big_df):
    """Regression (caught in verification): file_row_number is per-FILE, so the
    narrow shape on a multi-file glob would fan winners out across files and
    silently inflate/duplicate the sample. Globs must take the full-width
    fallback and return exact, duplicate-free counts."""
    for i in range(3):
        big_df.iloc[i * 5000 : (i + 1) * 5000].to_parquet(
            tmp_path / f"part{i}.parquet", index=False
        )
    glob = str(tmp_path / "*.parquet")
    assert engine._narrow_scan(glob, ["region"]) is None  # gate: not a single file
    strat = engine.sample(glob, 1200, strat_cols=["region"], seed=1)
    assert len(strat.data) == 1200
    assert not strat.data["id"].duplicated().any()
    res = engine.sample(glob, 1200, use_random=True, seed=1)
    assert len(res.data) == 1200
    assert not res.data["id"].duplicated().any()


def test_narrow_csv_keeps_single_pass_and_works(tmp_path, engine, big_df):
    # CSV has no stable row id: the full-width single-pass shape still applies
    p = tmp_path / "plain.csv"
    big_df.to_csv(p, index=False)
    assert engine._narrow_scan(str(p), ["region"]) is None
    result = engine.sample(str(p), 300, strat_cols=["region"], seed=1)
    assert len(result.data) == 300
    assert set(result.data.columns) == set(big_df.columns)


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
