"""PCA-based column reduction for sampled data.

Pure logic — nothing here prints. Callers (CLI, TUI) render the
:class:`ReductionResult` via :func:`data_sampler.report.format_reduction_report`.

The reduction replaces the numeric block of a DataFrame with its first *k*
principal components while preserving every other column. *k* is chosen either
directly (``n_components``) or as the fewest components whose cumulative
explained-variance ratio reaches a threshold (``variance_ratio``). The SVD is
exact (``np.linalg.svd``) with a deterministic sign convention, so the same
input always produces the same output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from ._logging import get_logger
from .stats import _classify

log = get_logger(__name__)

# name hints that a numeric column is an identifier (mirrors the anonymization
# workflow's suggestion heuristic). Matched per name *token* (equality or
# prefix) rather than as a bare substring, so "customer_id"/"userid" match but
# "tax_paid"/"valid_pct" (which merely contain "id") do not.
_ID_HINTS = ("id", "uuid", "guid", "account", "ssn", "customer", "cust", "user")


def _has_id_token(name: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    return any(t == h or t.startswith(h) for t in tokens for h in _ID_HINTS)


@dataclass
class ReductionResult:
    """Outcome of a PCA column reduction, with everything a report needs."""

    data: pd.DataFrame                      # preserved cols + PC1..PCk; same index & row count
    n_components: int                       # k actually produced (post-clamp; 0 = no-op)
    source_columns: list[str] = field(default_factory=list)   # numeric cols consumed
    component_names: list[str] = field(default_factory=list)  # ["PC1", ..., "PCk"]
    explained_variance_ratio: list[float] = field(default_factory=list)
    cumulative_variance_ratio: list[float] = field(default_factory=list)
    total_variance_retained: float = 0.0
    standardized: bool = True
    components: np.ndarray | None = None    # (k, p) loadings; rows are PCs, cols source_columns
    correlation_matrix: pd.DataFrame | None = None  # Pearson corr among source_columns
    column_groups: list[list[str]] = field(default_factory=list)  # correlated clusters
    dropped_columns: list[str] = field(default_factory=list)  # constant/all-missing (passthrough)
    notes: list[str] = field(default_factory=list)


def _numeric_columns(df: pd.DataFrame) -> list:
    """Columns classified ``"numeric"`` (bool/datetime are not candidates)."""
    return [c for c in df.columns if _classify(df[c]) == "numeric"]


def _svd_flip(u: np.ndarray, vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix the sign ambiguity of an SVD: make each component's largest-|loading|
    positive, so outputs are identical across runs, BLAS builds, and platforms."""
    max_abs = np.argmax(np.abs(vt), axis=1)
    signs = np.sign(vt[np.arange(vt.shape[0]), max_abs])
    signs[signs == 0] = 1.0
    return u * signs, vt * signs[:, None]


def _correlation_groups(
    corr: np.ndarray, names: list[str], threshold: float
) -> list[list[str]]:
    """Connected components of the |corr| >= threshold graph (size >= 2 only).

    Each group is a cluster of columns that move together — the redundancy
    PCA exploits to represent them in fewer dimensions.
    """
    parent = list(range(len(names)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        groups.setdefault(find(i), []).append(name)
    return [g for g in groups.values() if len(g) > 1]


def reduce_columns(
    df: pd.DataFrame,
    *,
    n_components: int | None = None,
    variance_ratio: float | None = None,
    columns: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
    standardize: bool = True,
    group_threshold: float = 0.7,
    prefix: str = "PC",
    seed: int | None = None,
) -> ReductionResult:
    """Reduce the numeric columns of ``df`` to *k* principal components.

    Exactly one of ``n_components`` (the number of output components) or
    ``variance_ratio`` (a float in (0, 1): keep the fewest components whose
    cumulative explained-variance ratio reaches it) must be given.

    ``columns`` limits which columns are reduced (default: every column whose
    kind is numeric — booleans and datetimes are never candidates, and naming
    one explicitly raises ``ValueError``; other explicitly named columns are
    coerced with ``pd.to_numeric``, so e.g. ``Decimal``-typed database columns
    can be reduced by naming them). ``exclude`` names columns to pass through
    unchanged (useful for identifiers). Columns not selected are always
    preserved. Missing values are mean-imputed per column (the row count never
    changes); constant, all-missing, and non-coercible columns carry no signal
    and pass through unchanged.

    With ``standardize`` (the default) each column is z-scored first, so the
    PCA runs on the correlation matrix and a large-unit column cannot dominate
    the components. ``seed`` is accepted for API parity with ``sample`` /
    ``anonymize``; the exact SVD is deterministic and does not consume it.

    Returns a :class:`ReductionResult` whose ``data`` has the consumed numeric
    block replaced (in place, at the position of its first column) by
    ``PC1..PCk``, plus the correlation matrix and the ``group_threshold``-based
    clusters of correlated columns that explain *why* the block compresses.
    """
    if (n_components is None) == (variance_ratio is None):
        raise ValueError(
            "provide exactly one of n_components or variance_ratio"
        )
    if n_components is not None and (
        not isinstance(n_components, (int, np.integer)) or n_components < 1
    ):
        raise ValueError(f"n_components must be a positive integer, got {n_components!r}")
    if variance_ratio is not None and not (0.0 < variance_ratio < 1.0):
        raise ValueError(
            f"variance_ratio must be strictly between 0 and 1, got {variance_ratio!r}"
        )
    if not (0.0 < group_threshold <= 1.0):
        raise ValueError(
            f"group_threshold must be in (0, 1], got {group_threshold!r}"
        )

    notes: list[str] = []
    # a lone string is a natural mistake for a one-column argument; iterating
    # it character-by-character would silently exclude/select the wrong thing
    if isinstance(exclude, str):
        exclude = [exclude]
    if isinstance(columns, str):
        columns = [columns]
    exclude = {str(c) for c in exclude}

    if columns is None:
        selected = [c for c in _numeric_columns(df) if str(c) not in exclude]
    else:
        selected = []
        for c in columns:
            if c not in df.columns:
                raise ValueError(f"column {c!r} not found in DataFrame")
            kind = _classify(df[c])
            if kind in ("boolean", "datetime"):
                raise ValueError(
                    f"column {c!r} is {kind}; only numeric (or numerically "
                    "coercible) columns can be reduced"
                )
            if str(c) not in exclude and c not in selected:
                selected.append(c)

    def no_op(reason: str) -> ReductionResult:
        notes.append(f"Column reduction skipped: {reason}")
        log.info("column reduction skipped: %s", reason)
        return ReductionResult(
            data=df.copy(), n_components=0, standardized=standardize, notes=notes
        )

    if len(selected) == 0:
        return no_op("no numeric columns to reduce")

    # ── build the numeric matrix (impute, never drop rows) ────────────────────
    n_rows = len(df)
    kept: list = []  # original column labels, in df order
    dropped: list = []
    columns_data: list[np.ndarray] = []
    imputed_cells = 0
    imputed_cols: list[str] = []

    for col in selected:
        # explicitly named columns may be object dtype (e.g. Decimal values
        # from a database driver); coerce, and treat anything non-finite
        # (NaN, ±inf, unparseable text) as missing so it is imputed — or the
        # whole column dropped — rather than poisoning the SVD
        x = pd.to_numeric(df[col], errors="coerce").astype("float64").to_numpy(copy=True)
        finite = np.isfinite(x)
        vals = x[finite]
        if vals.size == 0:
            dropped.append(col)
            continue
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        if std == 0.0 or not np.isfinite(std):
            dropped.append(col)
            continue
        n_missing = int(n_rows - vals.size)
        if n_missing:
            x[~finite] = float(vals.mean())
            imputed_cells += n_missing
            imputed_cols.append(col)
        kept.append(col)
        columns_data.append(x)

    if dropped:
        notes.append(
            "Passed through unchanged (constant, all-missing, or non-numeric "
            "— no signal to reduce): " + ", ".join(str(c) for c in dropped)
        )
    if imputed_cols:
        notes.append(
            f"Mean-imputed {imputed_cells} missing value(s) in: "
            + ", ".join(str(c) for c in imputed_cols)
        )

    if len(kept) == 0:
        return no_op("every numeric column is constant or all-missing")
    if len(kept) == 1:
        return no_op(
            f"only one usable numeric column ({kept[0]}); nothing to combine"
        )

    # ── ID-like warning (mirrors the workflow suggestion heuristic) ───────────
    for col, x in zip(kept, columns_data):
        n_unique = df[col].nunique()
        n_nonnull = int(df[col].notna().sum()) or 1
        unique_pct = n_unique / n_nonnull * 100
        name_hint = _has_id_token(str(col))
        all_integral = bool(np.all(x == np.round(x)))
        if (name_hint and unique_pct >= 90) or (all_integral and unique_pct >= 99):
            notes.append(
                f"'{col}' looks like an identifier ({n_unique:,} unique values); "
                "it will form its own artificial component — consider excluding it"
            )

    X = np.column_stack(columns_data)
    means = X.mean(axis=0)
    X = X - means
    if standardize:
        stds = X.std(axis=0, ddof=1)
        X = X / stds  # zero-variance columns were dropped above

    # correlation among the reduced columns: the rationale for why they
    # compress (on standardized data, PCA diagonalizes exactly this matrix)
    kept_names = [str(c) for c in kept]
    corr = np.corrcoef(X, rowvar=False)
    corr_df = pd.DataFrame(corr, index=kept_names, columns=kept_names)
    groups = _correlation_groups(corr, kept_names, group_threshold)

    # ── exact PCA via SVD ─────────────────────────────────────────────────────
    # pre-scale by the largest |value| so extreme magnitudes (possible with
    # standardize=False) cannot overflow S**2 to inf and silently zero the
    # variance ratios — the ratios are scale-invariant, and the scores are
    # multiplied back afterwards
    scale = float(np.max(np.abs(X)))
    if not np.isfinite(scale) or scale <= 0.0:
        return no_op("numeric columns have no usable variance after preprocessing")
    U, S, Vt = np.linalg.svd(X / scale, full_matrices=False)
    U, Vt = _svd_flip(U, Vt)
    total_ss = float((S**2).sum())
    if not np.isfinite(total_ss) or total_ss <= 0.0:
        return no_op("numeric columns have no usable variance after preprocessing")
    evr = (S**2) / total_ss
    cum = np.cumsum(evr)

    max_rank = len(S)  # == min(n_rows, p)
    if n_components is not None:
        k = int(n_components)
        if k > max_rank:
            notes.append(
                f"Requested {k} components but only {max_rank} are possible "
                f"({len(kept)} usable numeric columns, {n_rows} rows); producing {max_rank}"
            )
            k = max_rank
    else:
        k = int(np.searchsorted(cum, variance_ratio) + 1)
        k = min(k, max_rank)
        notes.append(
            f"{k} component(s) needed to retain ≥ {variance_ratio:.0%} of the variance"
        )

    component_names = [f"{prefix}{i + 1}" for i in range(k)]
    passthrough_names = {str(c) for c in df.columns if c not in kept}
    collision = [c for c in component_names if c in passthrough_names]
    if collision:
        raise ValueError(
            f"component name(s) {collision} collide with existing column(s); "
            "pass a different prefix"
        )

    scores = (U[:, :k] * S[:k] * scale).astype("float64")
    pc_frame = pd.DataFrame(scores, columns=component_names, index=df.index)

    # rebuild: preserved columns keep their order; the PC block sits where the
    # first consumed column was
    out_cols: list = []
    inserted = False
    for c in df.columns:
        if c in kept:
            if not inserted:
                out_cols.extend(component_names)
                inserted = True
            continue
        out_cols.append(c)
    out = pd.concat([df.drop(columns=kept), pc_frame], axis=1).loc[:, out_cols]

    log.info(
        "reduced %d numeric column(s) to %d component(s), %.1f%% variance retained",
        len(kept), k, float(cum[k - 1]) * 100,
    )
    return ReductionResult(
        data=out,
        n_components=k,
        source_columns=kept_names,
        component_names=component_names,
        explained_variance_ratio=[float(v) for v in evr[:k]],
        cumulative_variance_ratio=[float(v) for v in cum[:k]],
        total_variance_retained=float(cum[k - 1]),
        standardized=standardize,
        components=Vt[:k],
        correlation_matrix=corr_df,
        column_groups=groups,
        dropped_columns=[str(c) for c in dropped],
        notes=notes,
    )
