"""Per-column statistics, modelled after the Data Wrangler column summary view."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SPARK_CHARS = "▁▂▃▄▅▆▇█"


@dataclass
class ColumnStats:
    """Summary statistics for a single DataFrame column."""

    name: str
    dtype: str
    kind: str  # "numeric" | "boolean" | "datetime" | "categorical" | "text" | "other"
    count: int  # non-null values
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float
    # numeric-only (None otherwise)
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    # most frequent value: a float for numeric columns, a string otherwise
    # (None when the column is empty). Populated for every kind.
    mode: object | None = None
    # most frequent values as (value, count), up to 8
    top_values: list[tuple[str, int]] = field(default_factory=list)
    # distribution: 10-bin histogram counts for numeric, top-8 category counts otherwise
    histogram: list[int] = field(default_factory=list)
    # labels matching histogram entries (bin ranges or category names)
    histogram_labels: list[str] = field(default_factory=list)
    # whether the auto-stratifier would consider this column
    stratifiable: bool = False
    # True when unique/quantile figures are approximate (HLL / approx_quantile),
    # as produced by the DuckDB engine over very large inputs
    approximate: bool = False

    def summary(self) -> str:
        """One-line summary suitable for a table cell."""
        if self.kind == "numeric" and self.min is not None:
            return f"{_fmt_num(self.min)} … {_fmt_num(self.max)}  μ={_fmt_num(self.mean)}"
        if self.top_values:
            value, count = self.top_values[0]
            pct = count / self.count * 100 if self.count else 0
            return f"top: {value[:20]} ({pct:.0f}%)"
        return ""


def _fmt_num(x: float | None) -> str:
    if x is None:
        return "—"
    if not np.isfinite(x):
        return str(x)
    if x == int(x) and abs(x) < 1e15:
        return f"{int(x):,}"
    return f"{x:,.2f}"


def sparkline(counts: list[int]) -> str:
    """Render a list of counts as a unicode sparkline (e.g. ``▂▅█▃▁``)."""
    if not counts:
        return ""
    peak = max(counts)
    if peak == 0:
        return SPARK_CHARS[0] * len(counts)
    out = []
    for c in counts:
        if c == 0:
            out.append(SPARK_CHARS[0])
        else:
            idx = max(1, int(c / peak * (len(SPARK_CHARS) - 1)))
            out.append(SPARK_CHARS[idx])
    return "".join(out)


def _fractionable_values(series: pd.Series) -> np.ndarray | None:
    """Unique non-null OBSERVED values as float64 for columns whose
    representation can hold fractional numbers — float dtypes (numpy,
    nullable, or Arrow-backed), decimal extension dtypes (Arrow decimal128
    from ``read_parquet(dtype_backend='pyarrow')``), float-backed
    categoricals, and object columns of floats/Decimals (how parquet DECIMAL
    and SQL drivers arrive by default). None for everything else (ints,
    bools, strings, dates): those can never be continuous. Mirrors which
    DuckDB types get the engine-side probe
    (:func:`data_sampler.engine._duckdb_can_be_fractional`), and rules on
    observed values — not declared categories — because that is all the
    engine ever sees, so both paths reach the same verdict. Callers gate on
    cardinality first — the returned array is only ever unique values, so it
    stays small.
    """
    non_null = series.dropna()
    if isinstance(series.dtype, pd.CategoricalDtype):
        if not pd.api.types.is_float_dtype(series.cat.categories.dtype):
            return None
        return np.asarray(non_null.unique(), dtype=float)
    if (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_integer_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
        and not pd.api.types.is_complex_dtype(series)
    ):
        try:
            return np.asarray(non_null.unique(), dtype=float)
        except (TypeError, ValueError, OverflowError):
            return None
    if series.dtype == object:
        uniq = non_null.unique()
        if len(uniq) == 0 or not all(isinstance(v, numbers.Number) for v in uniq):
            return None
        try:
            return np.array([float(v) for v in uniq], dtype=float)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def is_stratifiable(series: pd.Series, n_rows: int, n_unique: int | None = None) -> bool:
    """Whether a column passes the auto-stratification candidate checks.

    Mirrors the selection rules used by
    :func:`data_sampler.sampling.find_stratification_columns`. Pass a
    precomputed ``n_unique`` to avoid a second ``nunique`` hash pass.
    """
    n_unique = series.nunique() if n_unique is None else n_unique
    if n_unique > min(100, n_rows * 0.5):
        return False
    if n_unique < 2:
        return False
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        avg_len = series.dropna().astype(str).str.len().mean()
        if avg_len > 50:
            return False
    if pd.api.types.is_numeric_dtype(series):
        if n_unique > min(20, n_rows * 0.3):
            return False
    # continuous columns (fractional values — prices, rates, measurements)
    # make poor strata even at low cardinality: skip them by default. The
    # check is value-based, not dtype-based, so parquet DECIMAL (object of
    # decimal.Decimal) and float-backed categorical columns rule the same way
    # the DuckDB engine's probe does. Whole-number floats stay candidates —
    # integer-coded columns (ratings, counts) usually arrive as float64 once
    # NaNs appear.
    vals = _fractionable_values(series)
    if vals is not None:
        finite = vals[np.isfinite(vals)]
        if finite.size and not np.all(finite == np.round(finite)):
            return False
    return True


def _classify(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if (
        series.dtype == object
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        avg_len = series.dropna().astype(str).str.len().mean()
        if pd.notna(avg_len) and avg_len > 50:
            return "text"
        return "categorical"
    return "other"


def compute_column_stats(series: pd.Series, n_rows: int | None = None) -> ColumnStats:
    """Compute :class:`ColumnStats` for a single Series."""
    n_rows = n_rows if n_rows is not None else len(series)
    non_null = series.dropna()
    count = len(non_null)
    missing = len(series) - count
    unique = series.nunique()
    kind = _classify(series)

    stats = ColumnStats(
        name=str(series.name),
        dtype=str(series.dtype),
        kind=kind,
        count=count,
        missing=missing,
        missing_pct=missing / len(series) * 100 if len(series) else 0.0,
        unique=unique,
        unique_pct=unique / count * 100 if count else 0.0,
        stratifiable=is_stratifiable(series, n_rows, n_unique=unique),
    )

    if kind == "numeric" and count > 0:
        stats.min = float(non_null.min())
        stats.max = float(non_null.max())
        stats.mean = float(non_null.mean())
        stats.std = float(non_null.std()) if count > 1 else 0.0
        stats.median = float(non_null.median())
        m = non_null.mode()
        stats.mode = float(m.iloc[0]) if len(m) else None
        finite = non_null.astype(float)
        finite = finite[np.isfinite(finite)]
        if len(finite) > 0:
            counts, edges = np.histogram(finite, bins=10)
            stats.histogram = [int(c) for c in counts]
            stats.histogram_labels = [
                f"{_fmt_num(float(edges[i]))} – {_fmt_num(float(edges[i + 1]))}"
                for i in range(len(counts))
            ]

    # top values only for non-numeric columns: numeric columns render their
    # histogram instead, and stringifying millions of numbers just to throw
    # the result away dominated this function's runtime (~60% on 1M rows)
    if count > 0 and kind != "numeric":
        vc = non_null.astype(str).value_counts().head(8)
        stats.top_values = [(str(v), int(c)) for v, c in vc.items()]
        stats.histogram = [int(c) for c in vc.values]
        stats.histogram_labels = [str(v) for v in vc.index]
        # the most frequent category is the first entry of value_counts
        stats.mode = stats.top_values[0][0] if stats.top_values else None

    return stats


def compute_stats(df: pd.DataFrame) -> list[ColumnStats]:
    """Compute per-column statistics for every column of ``df``."""
    n_rows = len(df)
    return [compute_column_stats(df[col], n_rows) for col in df.columns]
