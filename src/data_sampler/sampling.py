"""Stratified and random sampling engine.

Pure logic — nothing here prints. Callers (CLI, TUI) render the
:class:`SampleResult` via :mod:`data_sampler.report`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from ._logging import get_logger
from .stats import is_stratifiable

log = get_logger(__name__)


@dataclass
class SampleResult:
    """Outcome of a sampling run, with everything needed to build a report."""

    data: pd.DataFrame
    method: str  # "stratified" | "random" | "all"
    requested: int
    strat_cols: list[str] = field(default_factory=list)
    group_sizes: pd.Series | None = None
    allocations: pd.Series | None = None
    notes: list[str] = field(default_factory=list)


def find_stratification_columns(
    df: pd.DataFrame,
    sample_count: int,
    exclude: Iterable[str] = (),
) -> list[str]:
    """Pick columns suitable for stratification.

    Candidates are categorical / low-cardinality columns with 2–100 unique
    values; long text, ID-like numeric, and continuous numeric columns
    (fractional values) are skipped. Columns listed in ``exclude`` are never
    considered (the user's "skip" selections). The final set is pruned so the
    intersection-group count fits ``sample_count``.
    """
    exclude = set(exclude)
    candidates = []
    n_rows = len(df)

    for col in df.columns:
        if col in exclude:
            log.debug("stratification: column %r excluded by user", col)
            continue
        series = df[col]
        n_unique = series.nunique()
        if not is_stratifiable(series, n_rows, n_unique=n_unique):
            continue
        candidates.append((col, n_unique))

    # sort by fewest categories first (easiest to represent)
    candidates.sort(key=lambda x: x[1])

    # prune: only keep columns whose combined group count fits the sample size
    selected = []
    combo_count = 1
    for col, n_unique in candidates:
        new_combo = combo_count * n_unique
        if new_combo > sample_count:
            break
        combo_count = new_combo
        selected.append(col)

    log.debug("stratification columns selected: %s", selected)
    return selected


def stratified_sample(
    df: pd.DataFrame,
    count: int,
    strat_cols: list[str],
    random_state: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Sample ``count`` rows preserving the joint distribution of ``strat_cols``.

    Returns ``(sampled_df, group_sizes, allocations)``.
    """
    rng = np.random.default_rng(random_state) if not isinstance(
        random_state, np.random.Generator
    ) else random_state

    grouped = df.groupby(strat_cols, observed=True, dropna=False)
    group_sizes = grouped.size()
    total = group_sizes.sum()

    # compute proportional allocation
    allocations = (group_sizes / total * count).apply(math.floor)

    # distribute remaining slots to the largest remainders
    remainders = (group_sizes / total * count) - allocations
    shortfall = count - int(allocations.sum())
    if shortfall > 0:
        top_positions = remainders.values.argsort()[-shortfall:]
        for pos in top_positions:
            allocations.iloc[pos] += 1

    # sample from each group
    samples = []
    for i, (_, group_df) in enumerate(grouped):
        n = int(allocations.iloc[i])
        if n == 0:
            continue
        n = min(n, len(group_df))
        samples.append(group_df.sample(n=n, random_state=rng))

    result = pd.concat(samples).sample(frac=1, random_state=rng)  # shuffle

    # adjust if rounding left us short/over
    if len(result) < count:
        remaining = df.drop(result.index)
        extra = remaining.sample(
            n=min(count - len(result), len(remaining)), random_state=rng
        )
        result = pd.concat([result, extra])
    elif len(result) > count:
        result = result.head(count)

    return result, group_sizes, allocations


def sample(
    df: pd.DataFrame,
    count: int,
    use_random: bool = False,
    exclude_columns: Iterable[str] = (),
    random_state: int | np.random.Generator | None = None,
) -> SampleResult:
    """Create a representative sample of ``count`` rows.

    Stratifies automatically on suitable columns unless ``use_random`` is
    set. Columns in ``exclude_columns`` are never used for stratification
    (they still appear in the output). Returns a :class:`SampleResult`.
    """
    log.info("sampling %d of %d rows (random=%s)", count, len(df), use_random)

    if count >= len(df):
        return SampleResult(
            data=df,
            method="all",
            requested=count,
            notes=[
                f"Requested {count} samples but file only has {len(df)} rows. "
                "Returning all rows."
            ],
        )

    rng = np.random.default_rng(random_state) if not isinstance(
        random_state, np.random.Generator
    ) else random_state

    if use_random:
        return SampleResult(
            data=df.sample(n=count, random_state=rng),
            method="random",
            requested=count,
            notes=["Mode: pure random sampling"],
        )

    strat_cols = find_stratification_columns(df, count, exclude=exclude_columns)

    if not strat_cols:
        return SampleResult(
            data=df.sample(n=count, random_state=rng),
            method="random",
            requested=count,
            notes=[
                "No suitable columns for stratification found. "
                "Using pure random sampling."
            ],
        )

    result, group_sizes, allocations = stratified_sample(
        df, count, strat_cols, random_state=rng
    )
    return SampleResult(
        data=result,
        method="stratified",
        requested=count,
        strat_cols=strat_cols,
        group_sizes=group_sizes,
        allocations=allocations,
        notes=[f"Stratifying on columns: {strat_cols}"],
    )
