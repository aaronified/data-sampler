"""Terminal-friendly stratification reports (returned as strings, not printed)."""

from __future__ import annotations

import math

import pandas as pd

from .sampling import SampleResult

BAR_WIDTH = 30  # max width of distribution bars


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
