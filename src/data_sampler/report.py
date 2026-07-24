"""Terminal-friendly stratification reports (returned as strings, not printed)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .reduce import ReductionResult
from .sampling import SampleResult
from .stats import _classify, _fmt_num

BAR_WIDTH = 30  # max width of distribution bars
HIST_BAR_WIDTH = 20  # max width of the per-column comparison histogram bars


def format_distribution(df: pd.DataFrame, col: str, label: str = "Source") -> str:
    """Build a bar chart of a column's distribution."""
    out: list[str] = []
    counts = df[col].value_counts().sort_index()
    total = counts.sum()
    max_count = counts.max()
    max_label_len = max(len(str(v)) for v in counts.index)

    out.append(f"\n  {label} distribution for '{col}' ({counts.shape[0]} categories):")
    out.append(f"  {'─' * (max_label_len + BAR_WIDTH + 20)}")

    for value, count in counts.items():
        pct = count / total * 100
        bar_len = int(count / max_count * BAR_WIDTH) if max_count > 0 else 0
        bar = "█" * bar_len + "░" * (BAR_WIDTH - bar_len)
        out.append(f"  {str(value):>{max_label_len}}  {bar}  {count:>6} ({pct:5.1f}%)")

    out.append(f"  {'─' * (max_label_len + BAR_WIDTH + 20)}")
    return "\n".join(out)


def column_histogram_data(
    df_source: pd.DataFrame,
    df_sample: pd.DataFrame,
    bins: int = 10,
    top: int = 8,
) -> list[dict]:
    """Per-column source-vs-sample distributions, aligned for comparison.

    Returns one dict per column present in both frames::

        {"name", "kind", "labels": [str],
         "source_counts": [int], "sample_counts": [int],
         "source_pct": [float], "sample_pct": [float]}

    Numeric columns share bin edges (computed from the source) so the two
    histograms line up bin-for-bin; other columns use the source's ``top``
    most-frequent categories, looked up in the sample. Percentages are of each
    frame's non-null count, so for categorical columns the shown bars may sum
    to under 100 % when there are more than ``top`` categories.

    Pass the *pre-anonymization* sample (``SampleResult.data``) so the numbers
    describe how well the sample preserved the source distribution.
    """
    out: list[dict] = []
    for col in df_source.columns:
        if col not in df_sample.columns:
            continue
        src = df_source[col]
        samp = df_sample[col]
        kind = _classify(src)

        if kind == "numeric":
            s_src = pd.to_numeric(src, errors="coerce")
            s_src = s_src[np.isfinite(s_src)]
            s_samp = pd.to_numeric(samp, errors="coerce")
            s_samp = s_samp[np.isfinite(s_samp)]
            if len(s_src) == 0:
                continue
            edges = np.histogram_bin_edges(s_src, bins=bins)
            source_counts = np.histogram(s_src, bins=edges)[0]
            sample_counts = (
                np.histogram(s_samp, bins=edges)[0]
                if len(s_samp)
                else np.zeros(len(source_counts), dtype=int)
            )
            labels = [
                f"{_fmt_num(float(edges[i]))} – {_fmt_num(float(edges[i + 1]))}"
                for i in range(len(source_counts))
            ]
            # percentages over the FINITE counts the bins were built from —
            # ±inf values are excluded from the bars, so including them in the
            # denominator would silently deflate every percentage
            src_total = len(s_src) or 1
            samp_total = len(s_samp) or 1
        else:
            src_nonnull = src.dropna()
            if src_nonnull.empty:
                continue
            # a near-unique column (ids, emails, free text) has no meaningful
            # top-8 distribution — skip it rather than hash-count every value
            # to draw eight bars of count 1
            n_unique = src_nonnull.nunique()
            if n_unique > max(100, 0.5 * len(src_nonnull)):
                continue
            src_vc = src_nonnull.astype(str).value_counts()
            top_labels = list(src_vc.head(top).index)
            samp_vc = samp.dropna().astype(str).value_counts()
            source_counts = np.array([int(src_vc.get(l, 0)) for l in top_labels])
            sample_counts = np.array([int(samp_vc.get(l, 0)) for l in top_labels])
            labels = [str(l) for l in top_labels]
            src_total = int(src.notna().sum()) or 1
            samp_total = int(samp.notna().sum()) or 1
        out.append(
            {
                "name": str(col),
                "kind": kind,
                "labels": labels,
                "source_counts": [int(c) for c in source_counts],
                "sample_counts": [int(c) for c in sample_counts],
                "source_pct": [c / src_total * 100 for c in source_counts],
                "sample_pct": [c / samp_total * 100 for c in sample_counts],
            }
        )
    return out


def format_column_histograms(
    df_source: pd.DataFrame,
    df_sample: pd.DataFrame,
    bins: int = 10,
    top: int = 8,
) -> str:
    """Text render of :func:`column_histogram_data`: per-column source-vs-sample
    bars, so you can eyeball how each column's distribution held up."""
    data = column_histogram_data(df_source, df_sample, bins=bins, top=top)
    if not data:
        return ""

    out: list[str] = []
    out.append("╔═════════════════════════════════════════════════════════════════════╗")
    out.append("║                    COLUMN DISTRIBUTIONS (source → sample)           ║")
    out.append("╚═════════════════════════════════════════════════════════════════════╝")

    for d in data:
        labels = d["labels"]
        if not labels:
            continue
        label_w = min(20, max(len(l) for l in labels))
        peak = max([*d["source_pct"], *d["sample_pct"], 1e-9])
        out.append(f"\n  {d['name']}  ({d['kind']})")
        for label, s_pct, m_pct in zip(labels, d["source_pct"], d["sample_pct"]):
            lbl = label if len(label) <= label_w else label[: label_w - 1] + "…"
            s_len = int(s_pct / peak * HIST_BAR_WIDTH)
            m_len = int(m_pct / peak * HIST_BAR_WIDTH)
            s_bar = "█" * s_len + "░" * (HIST_BAR_WIDTH - s_len)
            m_bar = "█" * m_len + "░" * (HIST_BAR_WIDTH - m_len)
            out.append(
                f"  {lbl:>{label_w}}  src {s_bar} {s_pct:5.1f}%   "
                f"sam {m_bar} {m_pct:5.1f}%"
            )
    return "\n".join(out)


def format_stratification_report(
    df_original: pd.DataFrame, result: SampleResult
) -> str:
    """Build the side-by-side distribution comparison for a stratified sample.

    For random / all-rows results, returns the run notes only.
    """
    if result.method != "stratified":
        return "\n".join(result.notes)

    df_sample = result.data
    strat_cols = result.strat_cols
    group_sizes = result.group_sizes
    allocations = result.allocations
    sample_count = result.requested

    out: list[str] = []
    out.append("╔═════════════════════════════════════════════════════════════════════╗")
    out.append("║                        STRATIFICATION REPORT                        ║")
    out.append("╚═════════════════════════════════════════════════════════════════════╝")
    out.append(f"\n  Columns used: {', '.join(strat_cols)}")

    if len(strat_cols) > 1:
        out.append(
            "\n  Method: rows are grouped by the intersection of all stratification\n"
            "  columns simultaneously (e.g. A=x AND B=y AND C=z). Proportional\n"
            "  allocation is then computed per intersection group. A category value\n"
            "  that is small in every intersection group it appears in will receive\n"
            "  0 allocation — even if it looks sizable in the per-column view below."
        )
    else:
        out.append(
            f"\n  Method: rows are grouped by '{strat_cols[0]}' and sampled proportionally."
        )

    for col in strat_cols:
        orig_counts = df_original[col].value_counts(dropna=False).sort_index(na_position="last")
        samp_counts = df_sample[col].value_counts(dropna=False).sort_index(na_position="last")
        orig_total = orig_counts.sum()
        samp_total = samp_counts.sum()

        # sort with NaN last; use pandas isna to safely detect NaN keys
        raw_values = orig_counts.index.tolist()
        all_values = sorted((v for v in raw_values if not pd.isna(v)), key=str) + \
                     [v for v in raw_values if pd.isna(v)]

        def make_label(v):
            s = "(missing)" if pd.isna(v) else str(v)
            return s if len(s) <= 13 else s[:10] + "..."

        all_labels = [make_label(v) for v in all_values]
        max_label_len = 13
        half_bar = BAR_WIDTH // 2

        def get_count(counts, v):
            if pd.isna(v):
                nan_mask = counts.index.isna()
                return int(counts[nan_mask].sum()) if nan_mask.any() else 0
            return int(counts.get(v, 0))

        zero_rep = [v for v in all_values if get_count(samp_counts, v) == 0]

        out.append(f"\n  Column: '{col}' ({len(all_values)} categories)")
        header = f"  {'Value':>{max_label_len}}  {'Original':^{half_bar + 10}}  {'Sample':^{half_bar + 10}}"
        out.append(header)
        out.append(f"  {'─' * len(header)}")

        orig_max = orig_counts.max() if len(orig_counts) > 0 else 1
        samp_max = samp_counts.max() if len(samp_counts) > 0 else 1

        for value, label in zip(all_values, all_labels):
            o_count = get_count(orig_counts, value)
            s_count = get_count(samp_counts, value)
            o_pct = o_count / orig_total * 100 if orig_total > 0 else 0
            s_pct = s_count / samp_total * 100 if samp_total > 0 else 0

            o_bar_len = int(o_count / orig_max * half_bar) if orig_max > 0 else 0
            s_bar_len = int(s_count / samp_max * half_bar) if samp_max > 0 else 0

            o_bar = "█" * o_bar_len + "░" * (half_bar - o_bar_len)
            s_bar = "█" * s_bar_len + "░" * (half_bar - s_bar_len)

            flag = " ← not represented" if s_count == 0 else ""
            out.append(
                f"  {label:>{max_label_len}}  "
                f"{o_bar} {o_pct:5.1f}%  "
                f"{s_bar} {s_pct:5.1f}%{flag}"
            )

        out.append(f"  {'─' * len(header)}")
        out.append(
            f"  {'Totals':>{max_label_len}}  {' ' * half_bar} {orig_total:>5}   "
            f"{' ' * half_bar} {samp_total:>5}"
        )

        if zero_rep and group_sizes is not None and allocations is not None:
            total_rows = int(group_sizes.sum())
            # min group size needed to receive at least 1 sample
            min_size_for_one = math.ceil(total_rows / sample_count)

            out.append(
                f"\n  Warning: {len(zero_rep)} of {len(all_values)} categories in '{col}' "
                f"received 0 samples.\n"
                f"  (A group needs ≥ {min_size_for_one} rows to receive ≥ 1 sample at this scale.)"
            )

            if len(strat_cols) > 1:
                # For each zero-represented value, look up its intersection sub-groups
                col_idx = strat_cols.index(col)
                for value in zero_rep:
                    if isinstance(group_sizes.index, pd.MultiIndex):
                        mask = group_sizes.index.get_level_values(col_idx) == value
                        sub_sizes = group_sizes[mask]
                        sub_allocs = allocations[mask]
                    else:
                        sub_sizes = group_sizes[[value]] if value in group_sizes.index else pd.Series(dtype=int)
                        sub_allocs = allocations[[value]] if value in allocations.index else pd.Series(dtype=int)

                    n_subgroups = len(sub_sizes)
                    largest = int(sub_sizes.max()) if n_subgroups > 0 else 0
                    total_alloc = int(sub_allocs.sum())

                    out.append(
                        f"    {str(value):>{max_label_len}}: {n_subgroups} intersection group(s), "
                        f"largest has {largest} rows (needs {min_size_for_one}), "
                        f"total allocated = {total_alloc}"
                    )

    return "\n".join(out)


def format_reduction_report(result: ReductionResult, top_drivers: int = 3) -> str:
    """Build the PCA column-reduction report: variance kept per component,
    the correlated column groups that explain *why* the block compresses, and
    each component's top driving columns.

    For a no-op reduction (nothing was reduced), returns the notes only.
    """
    if result.n_components == 0:
        return "\n".join(result.notes)

    out: list[str] = []
    out.append("╔═════════════════════════════════════════════════════════════════════╗")
    out.append("║                       COLUMN REDUCTION (PCA)                        ║")
    out.append("╚═════════════════════════════════════════════════════════════════════╝")
    out.append(
        f"\n  {len(result.source_columns)} numeric columns → "
        f"{result.n_components} component(s), "
        f"{result.total_variance_retained * 100:.1f}% of the variance retained"
        + ("  (columns z-scored first)" if result.standardized else "")
    )
    out.append(f"  Columns reduced: {', '.join(result.source_columns)}")

    out.append(f"\n  {'Component':>12}  {'variance':>9}  {'cumulative':>10}")
    out.append(f"  {'─' * 37}")
    for name, evr, cum in zip(
        result.component_names,
        result.explained_variance_ratio,
        result.cumulative_variance_ratio,
    ):
        bar = "█" * int(evr * BAR_WIDTH) + "░" * (BAR_WIDTH - int(evr * BAR_WIDTH))
        out.append(f"  {name:>12}  {evr * 100:8.1f}%  {cum * 100:9.1f}%  {bar}")

    # rationale: which source columns move together (high |correlation|),
    # i.e. the redundancy PCA collapsed into shared components
    corr = result.correlation_matrix
    if result.column_groups and corr is not None:
        out.append("\n  CORRELATED COLUMN GROUPS — these move together, so PCA")
        out.append("  represents each group with shared components:")
        for i, group in enumerate(result.column_groups, 1):
            pair_rs = [
                abs(float(corr.loc[a, b]))
                for gi, a in enumerate(group)
                for b in group[gi + 1 :]
            ]
            mean_r = sum(pair_rs) / len(pair_rs) if pair_rs else 0.0
            out.append(f"    group {i}: {', '.join(group)}  (mean |r| = {mean_r:.2f})")
        grouped = {c for g in result.column_groups for c in g}
        singles = [c for c in result.source_columns if c not in grouped]
        if singles:
            out.append(
                f"    not strongly paired ({len(singles)}): {', '.join(singles)}"
            )
    elif corr is not None:
        out.append(
            "\n  No strongly correlated column groups — the columns are largely"
            "\n  independent, so each component blends many of them."
        )

    if result.components is not None and len(result.components):
        out.append("\n  Top drivers per component (|loading|):")
        for name, row in zip(result.component_names, result.components):
            order = np.argsort(-np.abs(row))[:top_drivers]
            drivers = ", ".join(
                f"{result.source_columns[j]} ({row[j]:+.2f})" for j in order
            )
            out.append(f"    {name}  ←  {drivers}")

    if result.notes:
        out.append("")
        for note in result.notes:
            out.append(f"  Note: {note}")

    return "\n".join(out)
