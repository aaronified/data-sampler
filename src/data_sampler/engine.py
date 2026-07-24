"""DuckDB-backed out-of-core engine for large inputs.

Optional — install the extra: ``pip install "data-sampler[large]"``.

The pandas path (:mod:`data_sampler.sampling`) loads the whole file into memory
and is the default for small/medium data. This engine instead pushes loading,
stratification, and sampling **into DuckDB**: vectorized, multi-threaded, and
able to spill to disk, so inputs far larger than RAM can be sampled without ever
materializing them in pandas. Only the resulting sample (``count`` rows) comes
back as a DataFrame.

Highlights:

- **Parallelism** — ``PRAGMA threads`` uses all cores by default.
- **Out-of-core** — a memory limit plus a temp directory let DuckDB spill.
- **Native readers** — CSV/TSV/JSON and, especially, Parquet are read directly
  with projection pushdown (only the needed columns are scanned).
- **Streaming sampling** — reservoir sampling for the random case (exact count,
  single pass) and window-based proportional sampling for the stratified case.

This module imports :mod:`duckdb` lazily, so importing it never fails just
because the optional dependency is absent — only *using* the engine does, with a
clear message.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from ._logging import get_logger
from .sampling import SampleResult
from .stats import ColumnStats, _fmt_num

log = get_logger(__name__)

# Above this row count the engine is a good idea and materializing the whole
# thing in pandas risks exhausting memory (used for auto-selection + warnings).
LARGE_ROW_THRESHOLD = 5_000_000

# extensions DuckDB can read natively (no pandas round-trip)
_NATIVE_READERS = (".csv", ".tsv", ".json", ".parquet")


def _is_remote(source) -> bool:
    """Whether ``source`` is a remote URI DuckDB reads via httpfs (http/https/s3)."""
    return isinstance(source, str) and source.lower().startswith(
        ("http://", "https://", "s3://", "gs://", "gcs://", "az://")
    )


def _remote_ext(uri: str) -> str:
    """Extension of a remote URI's path, ignoring the query string."""
    return Path(urlparse(uri).path).suffix.lower()


class DuckDBUnavailable(RuntimeError):
    """Raised when the DuckDB engine is used without the optional dependency."""


def _require_duckdb():
    try:
        import duckdb  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise DuckDBUnavailable(
            "The DuckDB engine needs the optional 'large' extra. Install it with:"
            "  pip install \"data-sampler[large]\""
        ) from exc
    return duckdb


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier (column name), escaping embedded quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def _quote_str(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _to_float(v) -> float | None:
    """Coerce a DuckDB scalar to a finite float, or None (NULL/NaN/inf)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _duckdb_kind(dtype: str) -> str:
    """Map a DuckDB column type name to a :class:`ColumnStats` kind."""
    d = dtype.lower()
    # nested/wrapped types first: INTEGER[], DECIMAL(6,2)[], STRUCT(...), MAP(...)
    # would otherwise match the scalar numeric prefixes and crash the numeric
    # aggregates ("Unimplemented type for cast (INTEGER[] -> DOUBLE)")
    if d.endswith("]") or d.startswith(("struct", "map", "list", "union", "array", "json")):
        return "other"
    if d.startswith("bool"):
        return "boolean"
    if d.startswith(("timestamp", "date")):
        return "datetime"
    # TIME (no date part) and INTERVAL (timedelta) cannot be datetime-jittered
    # or averaged; "interval" must also never match the "int" numeric prefix
    if d.startswith(("time", "interval")):
        return "other"
    if d.startswith((
        "tinyint", "smallint", "integer", "int", "bigint", "hugeint",
        "utinyint", "usmallint", "uinteger", "ubigint", "uhugeint",
        "decimal", "double", "float", "real", "numeric",
    )):
        return "numeric"
    if d.startswith(("varchar", "char", "text", "string", "uuid", "bit", "blob", "enum")):
        return "categorical"  # refined to "text" by average length
    return "other"


def _duckdb_can_be_fractional(dtype: str) -> bool:
    """Whether a scalar DuckDB numeric type can hold fractional values.

    Integer types cannot, so the continuous-column probe (a value scan) is
    skipped for them.
    """
    return dtype.lower().startswith(("decimal", "double", "float", "real", "numeric"))


def _proportional_allocation(sizes: np.ndarray, count: int) -> np.ndarray:
    """Largest-remainder proportional allocation summing exactly to ``count``.

    Mirrors :func:`data_sampler.sampling.stratified_sample` so the DuckDB path
    allocates strata identically to the pandas path.
    """
    sizes = np.asarray(sizes, dtype=float)
    total = sizes.sum()
    if total <= 0:
        return np.zeros(len(sizes), dtype=np.int64)
    exact = sizes / total * count
    alloc = np.floor(exact).astype(np.int64)
    shortfall = int(count - alloc.sum())
    if shortfall > 0:
        # hand the remaining slots to the largest fractional remainders
        order = np.argsort(-(exact - alloc), kind="stable")
        alloc[order[:shortfall]] += 1
    # never allocate a stratum more rows than it has
    return np.minimum(alloc, sizes.astype(np.int64))


class DuckDBEngine:
    """A configured DuckDB connection with sampling/stats helpers.

    ``threads`` defaults to all CPU cores; ``memory_limit`` (e.g. ``"8GB"``)
    caps RAM so DuckDB spills to ``temp_directory`` instead of OOM-ing.
    Use as a context manager to close the connection.
    """

    def __init__(
        self,
        threads: int | None = None,
        memory_limit: str | None = None,
        temp_directory: str | None = None,
    ):
        duckdb = _require_duckdb()
        self.con = duckdb.connect(":memory:")
        self.threads = int(threads) if threads else (os.cpu_count() or 4)
        self.con.execute(f"PRAGMA threads={self.threads}")
        if memory_limit:
            self.con.execute(f"PRAGMA memory_limit={_quote_str(memory_limit)}")
        self.con.execute(
            f"PRAGMA temp_directory={_quote_str(temp_directory or tempfile.gettempdir())}"
        )
        self._reg = 0
        self._httpfs_ready = False  # lazily INSTALL/LOAD httpfs on first remote read
        # per-source row-count cache: one engine session assumes its file
        # sources do not change underneath it, so count(*) runs once per file
        # instead of once per operation (row_count/sample/strat-selection/stats)
        self._count_cache: dict[str, int] = {}
        log.info("DuckDBEngine ready (threads=%d, memory_limit=%s)", self.threads, memory_limit)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "DuckDBEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── source resolution ─────────────────────────────────────────────────────

    def _ensure_httpfs(self) -> None:
        """Load the httpfs extension so ``read_parquet``/``read_csv`` can take
        http(s)/s3 URIs (Parquet uses HTTP range requests — only the needed
        row-groups are fetched, so a multi-GB remote file is never downloaded
        whole). Best-effort: needs network on first install."""
        if self._httpfs_ready:
            return
        try:
            self.con.execute("INSTALL httpfs")
            self.con.execute("LOAD httpfs")
            self._httpfs_ready = True
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "reading a remote (http/s3) source needs DuckDB's httpfs "
                f"extension, which could not be installed/loaded: {exc}"
            ) from exc

    def _source_sql(self, source) -> str:
        """Return a SQL table expression for ``source``.

        ``source`` may be a local path or an http(s)/s3 **URL** (CSV/TSV/JSON/
        Parquet, read natively — remote via httpfs), or a pandas DataFrame
        (registered zero-copy via Arrow).
        """
        if isinstance(source, pd.DataFrame):
            self._reg += 1
            name = f"_src_df_{self._reg}"
            self.con.register(name, source)
            return name
        if _is_remote(source):
            self._ensure_httpfs()
            ext = _remote_ext(str(source))
            path = _quote_str(str(source))  # keep the URI verbatim (no Path())
        else:
            p = Path(source)
            ext = p.suffix.lower()
            path = _quote_str(str(p))
        if ext == ".parquet":
            return f"read_parquet({path})"
        if ext == ".csv":
            return f"read_csv({path}, header=true, auto_detect=true)"
        if ext == ".tsv":
            return f"read_csv({path}, header=true, auto_detect=true, delim='\t')"
        if ext == ".json":
            expr = f"read_json_auto({path})"
            # DuckDB parses a columns-oriented JSON document — the DEFAULT
            # output of pandas DataFrame.to_json() — as ONE row of index-keyed
            # STRUCTs, which would silently corrupt the sample. Detect that
            # shape (1 row, all STRUCT/MAP columns) and refuse with guidance.
            info = self.con.execute(f"DESCRIBE SELECT * FROM {expr}").fetchall()
            if info and all(
                str(r[1]).upper().startswith(("STRUCT", "MAP")) for r in info
            ):
                n = self.con.execute(
                    f"SELECT count(*) FROM (SELECT * FROM {expr} LIMIT 2) t"
                ).fetchone()[0]
                if int(n) == 1:
                    raise ValueError(
                        f"{p.name} looks columns-oriented (the pandas to_json "
                        "default); the DuckDB engine reads records-oriented or "
                        "newline-delimited JSON — re-save with "
                        "orient='records' (or lines=True), or use --engine pandas"
                    )
            return expr
        raise ValueError(
            f"DuckDB engine cannot read '{ext}' natively; supported: "
            f"{', '.join(_NATIVE_READERS)} (or pass a pandas DataFrame)"
        )

    # ── introspection ─────────────────────────────────────────────────────────

    def _count(self, source, src_sql: str) -> int:
        """Row count of ``source``, cached per file (free for DataFrames)."""
        if isinstance(source, pd.DataFrame):
            return len(source)
        if src_sql not in self._count_cache:
            self._count_cache[src_sql] = int(
                self.con.execute(f"SELECT count(*) FROM {src_sql}").fetchone()[0]
            )
        return self._count_cache[src_sql]

    def row_count(self, source) -> int:
        return self._count(source, self._source_sql(source))

    def columns(self, source) -> list[str]:
        src = self._source_sql(source)
        rel = self.con.execute(f"SELECT * FROM {src} LIMIT 0")
        return [d[0] for d in rel.description]

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            # setseed wants a value in [-1, 1]; map the int deterministically
            self.con.execute(f"SELECT setseed({(int(seed) % 1000) / 1000.0})")

    # ── stratification-column selection ───────────────────────────────────────

    def find_stratification_columns(
        self, source, sample_count: int, exclude: Iterable[str] = ()
    ) -> list[str]:
        """Pick low-cardinality columns to stratify on, using approximate
        distinct counts (HyperLogLog) so it stays cheap on huge inputs."""
        exclude = set(exclude)
        src = self._source_sql(source)
        n_rows = self._count(source, src)
        info = self.con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
        # classify via _duckdb_kind so type rules match stats() (and never
        # match INTERVAL against the "int" numeric prefix)
        cols = [
            (r[0], _duckdb_kind(str(r[1])), str(r[1]))
            for r in info
            if r[0] not in exclude
        ]
        if not cols:
            return []

        # one pass: approx distinct per column (HLL) + avg text length for
        # string-ish columns (VARCHAR and ENUM both take this rule) + a
        # fractional-values probe for float-typed columns (continuous columns
        # are skipped as strata, mirroring stats.is_stratifiable)
        parts = []
        for name, kind, dtype in cols:
            q = _quote_ident(name)
            parts.append(f"approx_count_distinct({q}) AS {_quote_ident(name + '::nd')}")
            if kind == "categorical":
                parts.append(
                    f"avg(length(CAST({q} AS VARCHAR))) AS {_quote_ident(name + '::len')}"
                )
            elif kind == "numeric" and _duckdb_can_be_fractional(dtype):
                parts.append(
                    f"bool_or({q}::DOUBLE <> trunc({q}::DOUBLE)) "
                    f"FILTER (WHERE isfinite({q}::DOUBLE)) "
                    f"AS {_quote_ident(name + '::frac')}"
                )
        row = self.con.execute(f"SELECT {', '.join(parts)} FROM {src}").fetchdf().iloc[0]

        candidates: list[tuple[str, int]] = []
        for name, kind, _dtype in cols:
            n_unique = int(row[name + "::nd"] or 0)
            if n_unique < 2 or n_unique > min(100, n_rows * 0.5):
                continue
            if kind == "numeric" and n_unique > min(20, n_rows * 0.3):
                continue
            frac = row.get(name + "::frac")
            if frac is not None and pd.notna(frac) and bool(frac):
                continue
            if (name + "::len") in row and (row[name + "::len"] or 0) > 50:
                continue
            candidates.append((name, n_unique))

        candidates.sort(key=lambda x: x[1])  # fewest categories first
        selected: list[str] = []
        combo = 1
        for name, n_unique in candidates:
            if combo * n_unique > sample_count:
                break
            combo *= n_unique
            selected.append(name)
        log.debug("engine stratification columns: %s", selected)
        return selected

    # ── approximate stats ─────────────────────────────────────────────────────

    def stats(
        self,
        source,
        columns: Iterable[str] | None = None,
        approximate: bool = True,
        histogram_bins: int = 10,
        top: int = 8,
        distributions: bool = True,
    ) -> list[ColumnStats]:
        """Per-column :class:`ColumnStats`, computed in DuckDB.

        With ``approximate`` (default) distinct counts use HyperLogLog
        (``approx_count_distinct``) and the median uses ``approx_quantile`` — both
        streaming, so column stats stay cheap over billions of rows. Set
        ``approximate=False`` for exact counts/quantiles on small inputs.

        ``distributions`` controls whether per-column histograms / top-values are
        computed (one extra streaming pass per column). Turn it off for a single
        cheap scalar pass across very wide (thousands-of-columns) inputs.
        """
        con = self.con
        src = self._source_sql(source)
        total = self._count(source, src)
        info = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
        col_types = {r[0]: str(r[1]) for r in info}
        if columns is None:
            cols = list(col_types)
        else:
            cols = [str(c) for c in columns]
            unknown = [c for c in cols if c not in col_types]
            if unknown:
                raise KeyError(f"Column(s) not found in source: {', '.join(unknown)}")
        if not cols:
            return []

        # Pass A: all scalar aggregates for every column in ONE query
        parts: list[str] = []
        for i, c in enumerate(cols):
            q = _quote_ident(c)
            a = f"c{i}"
            kind = _duckdb_kind(col_types[c])
            parts.append(f"count({q}) AS {a}_cnt")
            parts.append(
                (f"approx_count_distinct({q})" if approximate else f"count(DISTINCT {q})")
                + f" AS {a}_nd"
            )
            if kind == "numeric":
                # NaN is a VALUE to DuckDB, not NULL: unfiltered, it poisons
                # min/max; and ±inf makes stddev_samp raise "out of range" —
                # filter every sensitive aggregate down to finite values
                fin = f"FILTER (WHERE isfinite({q}::DOUBLE))"
                parts += [
                    f"min({q}::DOUBLE) {fin} AS {a}_min",
                    f"max({q}::DOUBLE) {fin} AS {a}_max",
                    f"avg({q}::DOUBLE) {fin} AS {a}_avg",
                    f"stddev_samp({q}::DOUBLE) {fin} AS {a}_std",
                    (f"approx_quantile({q}::DOUBLE, 0.5)" if approximate
                     else f"quantile_cont({q}::DOUBLE, 0.5)") + f" {fin} AS {a}_med",
                ]
                if _duckdb_can_be_fractional(col_types[c]):
                    parts.append(
                        f"bool_or({q}::DOUBLE <> trunc({q}::DOUBLE)) {fin} AS {a}_frac"
                    )
            elif kind == "categorical":
                parts.append(f"avg(length(CAST({q} AS VARCHAR))) AS {a}_len")
        row = con.execute(f"SELECT {', '.join(parts)} FROM {src}").fetchdf().iloc[0]

        out: list[ColumnStats] = []
        for i, c in enumerate(cols):
            a = f"c{i}"
            kind = _duckdb_kind(col_types[c])
            count = int(row[f"{a}_cnt"] or 0)
            missing = total - count
            # clamp: HLL can overestimate distinct counts past the row count,
            # which would push unique_pct over 100% and skew suggest_type
            unique = min(int(row[f"{a}_nd"] or 0), count)
            if kind == "categorical" and (row.get(f"{a}_len") or 0) > 50:
                kind = "text"

            cs = ColumnStats(
                name=str(c), dtype=col_types[c], kind=kind, count=count,
                missing=missing, missing_pct=missing / total * 100 if total else 0.0,
                unique=unique, unique_pct=unique / count * 100 if count else 0.0,
                approximate=approximate,
            )
            # stratifiable flag, mirroring stats.is_stratifiable's rules
            strat = 2 <= unique <= min(100, total * 0.5)
            if kind == "numeric" and unique > min(20, total * 0.3):
                strat = False
            frac = row.get(f"{a}_frac")
            if pd.notna(frac) and bool(frac):
                strat = False  # continuous (fractional-valued) numeric column
            if kind == "text":
                strat = False
            cs.stratifiable = strat

            if kind == "numeric" and count > 0:
                cs.min = _to_float(row.get(f"{a}_min"))
                cs.max = _to_float(row.get(f"{a}_max"))
                cs.mean = _to_float(row.get(f"{a}_avg"))
                cs.std = _to_float(row.get(f"{a}_std")) or 0.0
                cs.median = _to_float(row.get(f"{a}_med"))

            if distributions and count > 0:
                self._fill_distribution(cs, src, c, kind, histogram_bins, top)
            out.append(cs)
        return out

    def _fill_distribution(self, cs, src, col, kind, bins, top) -> None:
        con = self.con
        q = _quote_ident(col)
        if kind == "numeric" and cs.min is not None and cs.max is not None and cs.min < cs.max:
            lo, hi = float(cs.min), float(cs.max)
            width = hi - lo
            # equal-width bins via floor (width_bucket isn't available in
            # DuckDB); clamp to [0, bins-1] so the max value lands in the last
            # bin, and drop NaN/±inf explicitly (NaN is a value, not NULL, in
            # DuckDB, and inf would overflow the INTEGER bin cast)
            rows = con.execute(
                f"SELECT least({bins - 1}, greatest(0, CAST(floor("
                f"({q}::DOUBLE - {lo!r}) / {width!r} * {bins}) AS INTEGER))) AS b, "
                f"count(*) AS n FROM {src} "
                f"WHERE {q} IS NOT NULL AND isfinite({q}::DOUBLE) GROUP BY b"
            ).fetchall()
            counts = [0] * bins
            for b, n in rows:
                counts[int(b)] += int(n)
            cs.histogram = counts
            edges = [lo + (hi - lo) * k / bins for k in range(bins + 1)]
            cs.histogram_labels = [
                f"{_fmt_num(edges[k])} – {_fmt_num(edges[k + 1])}" for k in range(bins)
            ]
        elif kind in ("categorical", "boolean", "text", "datetime", "other"):
            rows = con.execute(
                f"SELECT CAST({q} AS VARCHAR) AS v, count(*) AS n FROM {src} "
                f"WHERE {q} IS NOT NULL GROUP BY v ORDER BY n DESC, v LIMIT {top}"
            ).fetchall()
            cs.top_values = [(str(v), int(n)) for v, n in rows]
            cs.histogram = [int(n) for _, n in rows]
            cs.histogram_labels = [str(v) for v, _ in rows]

    # ── sampling ──────────────────────────────────────────────────────────────

    # column names the two-phase narrow shape needs for itself; a source that
    # already uses one of these falls back to the single-pass full-width shape
    _NARROW_RESERVED = frozenset({"file_row_number", "_ds_rid", "_ds_rowid", "_rn"})

    def _narrow_scan(self, source, strat_cols: list[str]):
        """Set up a two-phase "narrow" sample: rank/sample over only
        ``strat_cols`` plus a stable row id, then fetch the winning full rows.

        Sorting or reservoir-buffering the full row payload is the dominant
        cost on wide tables; this shape makes phase 1 width-independent.
        Returns ``(scan_sql, rid_expr, fetch)`` where ``fetch(row_ids)``
        materializes the winners with every column — or ``None`` when the
        source has no cheap stable row id (CSV/TSV/JSON must re-parse the
        whole text per scan, so single-pass full-width is already the optimal
        shape there) or a column name collides with the shape's internals.
        """
        if isinstance(source, pd.DataFrame):
            if self._NARROW_RESERVED & set(map(str, source.columns)):
                return None
            # only the stratification columns + a positional id enter DuckDB;
            # the wide payload never leaves pandas
            if strat_cols:
                narrow = source.loc[:, list(strat_cols)].reset_index(drop=True).copy()
            else:
                narrow = pd.DataFrame()
            narrow["_ds_rid"] = np.arange(len(source), dtype=np.int64)
            self._reg += 1
            name = f"_narrow_{self._reg}"
            self.con.register(name, narrow)

            def fetch(row_ids: np.ndarray) -> pd.DataFrame:
                # positional take preserves the original dtypes exactly
                return source.take(np.asarray(row_ids)).reset_index(drop=True)

            return name, _quote_ident("_ds_rid"), fetch

        if _is_remote(source):
            # a URL has no cheap local stable row id; the full-width single-pass
            # shape reads it fine via httpfs (range requests still apply)
            return None
        p = Path(source)
        if p.suffix.lower() != ".parquet":
            return None
        # file_row_number is numbered PER FILE: it is a valid global row id
        # only for a single physical file. Globs / multi-file datasets
        # (data/*.parquet) must use the full-width fallback, which never
        # relies on it — a per-file id would fan each winner out to every
        # file sharing that row number and silently corrupt the sample.
        # (A composite filename+row id could lift this later.)
        if not p.is_file():
            return None
        if self._NARROW_RESERVED & set(map(str, self.columns(source))):
            return None
        scan = f"read_parquet({_quote_str(str(p))}, file_row_number=true)"

        def fetch(row_ids: np.ndarray) -> pd.DataFrame:
            self._reg += 1
            rid_tbl = f"_rids_{self._reg}"
            self.con.register(
                rid_tbl, pd.DataFrame({"rid": np.asarray(row_ids, dtype=np.int64)})
            )
            # ORDER BY the row id: deterministic output order for seeded runs
            return self.con.execute(
                f"SELECT s.* EXCLUDE (file_row_number) FROM {scan} s "
                f"JOIN {rid_tbl} r ON s.file_row_number = r.rid "
                f"ORDER BY s.file_row_number"
            ).df()

        return scan, "file_row_number", fetch

    def sample(
        self,
        source,
        count: int,
        use_random: bool = False,
        exclude_columns: Iterable[str] = (),
        strat_cols: list[str] | None = None,
        seed: int | None = None,
    ) -> SampleResult:
        """Draw ``count`` rows out-of-core, returning a :class:`SampleResult`.

        Stratifies automatically (unless ``use_random``) on suitable columns;
        pass ``strat_cols`` to force a set. Only the sample is materialized.
        """
        src = self._source_sql(source)
        total = self._count(source, src)
        log.info("engine sampling %d of %d rows (random=%s)", count, total, use_random)

        if count >= total:
            notes = [f"Requested {count} rows but source has {total}. Returning all rows."]
            if total >= LARGE_ROW_THRESHOLD:
                # the whole point of this engine is to never materialize the
                # source — warn loudly before honoring an all-rows request
                warn = (
                    f"Materializing all {total:,} rows into memory defeats "
                    "out-of-core sampling and may exhaust RAM — request fewer rows."
                )
                log.warning(warn)
                notes.append(f"WARNING: {warn}")
            data = self.con.execute(f"SELECT * FROM {src}").df()
            return SampleResult(data=data, method="all", requested=count, notes=notes)

        self._set_seed(seed)

        if use_random:
            return self._reservoir(source, src, count, seed)

        if strat_cols is None:
            strat_cols = self.find_stratification_columns(source, count, exclude_columns)
        if not strat_cols:
            result = self._reservoir(source, src, count, seed)
            result.notes = ["No suitable stratification columns found; using reservoir sampling."]
            return result

        return self._stratified(source, src, count, strat_cols, seed)

    def _reservoir(self, source, src: str, count: int, seed: int | None) -> SampleResult:
        rep = f" REPEATABLE ({int(seed)})" if seed is not None else ""
        narrow = self._narrow_scan(source, [])
        if narrow is not None:
            # phase 1: reservoir over just the row id (width-independent);
            # phase 2: fetch the winners' full rows
            scan, rid_expr, fetch = narrow
            row_ids = self.con.execute(
                f"SELECT {rid_expr} AS rid FROM {scan} "
                f"USING SAMPLE reservoir({int(count)} ROWS){rep}"
            ).df()["rid"].to_numpy()
            data = fetch(row_ids)
        else:
            data = self.con.execute(
                f"SELECT * FROM {src} USING SAMPLE reservoir({int(count)} ROWS){rep}"
            ).df()
        return SampleResult(
            data=data, method="random", requested=count,
            notes=["Mode: DuckDB reservoir sampling (out-of-core)"],
        )

    def _stratified(
        self, source, src: str, count: int, strat_cols: list[str], seed: int | None
    ) -> SampleResult:
        cols_sql = ", ".join(_quote_ident(c) for c in strat_cols)
        # pass 1: stratum sizes (small — one row per stratum). ORDER BY pins
        # the stratum order: DuckDB's GROUP BY row order is nondeterministic,
        # and _proportional_allocation breaks remainder ties by position, so
        # an unpinned order would make even seeded runs nondeterministic.
        group_df = self.con.execute(
            f"SELECT {cols_sql}, count(*) AS _n FROM {src} "
            f"GROUP BY {cols_sql} ORDER BY {cols_sql} NULLS LAST"
        ).df()
        allocations = _proportional_allocation(group_df["_n"].to_numpy(), count)
        group_df["_alloc"] = allocations

        # pass 2: proportional per-stratum sampling. row_number over a random
        # order per partition, keep the first _alloc of each stratum. DuckDB
        # parallelizes the partitions and spills if needed.
        alloc_tbl = f"_alloc_{self._reg}"
        self._reg += 1
        self.con.register(alloc_tbl, group_df)
        join_on = " AND ".join(
            f"r.{_quote_ident(c)} IS NOT DISTINCT FROM a.{_quote_ident(c)}"
            for c in strat_cols
        )
        narrow = self._narrow_scan(source, strat_cols)
        if narrow is not None:
            # two-phase: rank only the strat columns + row id (the window sort
            # never touches the wide payload), then fetch the winners' rows
            scan, rid_expr, fetch = narrow
            query = f"""
                WITH ranked AS (
                    SELECT {rid_expr} AS _ds_rowid, {cols_sql}, row_number() OVER (
                        PARTITION BY {cols_sql} ORDER BY random()
                    ) AS _rn
                    FROM {scan}
                )
                SELECT r._ds_rowid
                FROM ranked r
                JOIN {alloc_tbl} a ON {join_on}
                WHERE r._rn <= a._alloc
            """
        else:
            fetch = None
            query = f"""
                WITH ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY {cols_sql} ORDER BY random()
                    ) AS _rn
                    FROM {src}
                )
                SELECT r.* EXCLUDE (_rn)
                FROM ranked r
                JOIN {alloc_tbl} a ON {join_on}
                WHERE r._rn <= a._alloc
            """
        # DuckDB's random() ordering is only reproducible single-threaded, so a
        # seeded run goes single-threaded for determinism; unseeded runs keep all
        # cores (the distribution is preserved either way).
        reproducible = seed is not None
        if reproducible:
            self.con.execute("PRAGMA threads=1")
            self._set_seed(seed)
        try:
            result_df = self.con.execute(query).df()
            if fetch is not None:
                data = fetch(result_df["_ds_rowid"].to_numpy())
            else:
                data = result_df
        finally:
            if reproducible:
                self.con.execute(f"PRAGMA threads={self.threads}")

        group_sizes = group_df.set_index(strat_cols)["_n"]
        alloc_series = group_df.set_index(strat_cols)["_alloc"]
        return SampleResult(
            data=data, method="stratified", requested=count,
            strat_cols=list(strat_cols), group_sizes=group_sizes, allocations=alloc_series,
            notes=[f"Stratifying on columns (DuckDB, out-of-core): {list(strat_cols)}"],
        )


# ── module-level conveniences ──────────────────────────────────────────────────


def duckdb_available() -> bool:
    """Whether the optional DuckDB dependency is importable."""
    try:
        import duckdb  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def should_use_engine(source, row_threshold: int = LARGE_ROW_THRESHOLD) -> bool:
    """Heuristic for auto-selecting the DuckDB engine: Parquet inputs (always a
    win via pushdown) or files large enough that pandas would strain."""
    if not duckdb_available():
        return False
    if isinstance(source, pd.DataFrame):
        return len(source) >= row_threshold
    if _is_remote(source):
        # remote Parquet is always a win (httpfs range requests avoid a full
        # download); other remote formats fall to the pandas streaming path
        return _remote_ext(source) == ".parquet"
    ext = Path(source).suffix.lower()
    if ext == ".parquet":
        return True
    if ext not in _NATIVE_READERS:
        return False  # e.g. Excel: DuckDB cannot read it natively
    try:
        # rough gate on file size (bytes) as a proxy for row count
        return os.path.getsize(source) >= row_threshold * 20
    except OSError:
        return False


def large_materialization_warning(
    n_rows: int, n_cols: int, row_threshold: int = LARGE_ROW_THRESHOLD
) -> str | None:
    """A warning to show before loading a big source fully into pandas.

    Parquet compresses hard on disk, so a modest file can explode in memory —
    surface the speedbump and point at the out-of-core engine.
    """
    if n_rows >= row_threshold:
        return (
            f"{n_rows:,} rows x {n_cols} columns is large: loading it fully into "
            "pandas may exhaust memory (Parquet especially expands well beyond its "
            "on-disk size). The DuckDB engine samples it out-of-core without "
            "materializing the whole dataset."
        )
    return None


def sample(
    source,
    count: int,
    use_random: bool = False,
    exclude_columns: Iterable[str] = (),
    strat_cols: list[str] | None = None,
    seed: int | None = None,
    threads: int | None = None,
    memory_limit: str | None = None,
) -> SampleResult:
    """Convenience: open a :class:`DuckDBEngine`, sample ``source``, close it."""
    with DuckDBEngine(threads=threads, memory_limit=memory_limit) as engine:
        return engine.sample(
            source, count, use_random=use_random,
            exclude_columns=exclude_columns, strat_cols=strat_cols, seed=seed,
        )


def stats(
    source,
    columns: Iterable[str] | None = None,
    approximate: bool = True,
    threads: int | None = None,
    memory_limit: str | None = None,
) -> list[ColumnStats]:
    """Convenience: open a :class:`DuckDBEngine`, compute stats, close it."""
    with DuckDBEngine(threads=threads, memory_limit=memory_limit) as engine:
        return engine.stats(source, columns=columns, approximate=approximate)
