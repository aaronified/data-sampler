import sys
import os
import argparse
import math
import pandas as pd
from pathlib import Path

# ensure UTF-8 output on Windows (stdout is None in windowed frozen apps)
if sys.stdout is not None and sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BAR_WIDTH = 30  # max width of distribution bars in terminal


def load_file(filepath, sheet=None):
    ext = Path(filepath).suffix.lower()
    readers = {
        ".csv": lambda: pd.read_csv(filepath),
        ".tsv": lambda: pd.read_csv(filepath, sep="\t"),
        ".json": lambda: pd.read_json(filepath),
        ".xlsx": lambda: pd.read_excel(filepath, sheet_name=sheet or 0),
        ".xls": lambda: pd.read_excel(filepath, sheet_name=sheet or 0),
        ".parquet": lambda: pd.read_parquet(filepath),
    }
    if ext not in readers:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(readers.keys())}"
        )
    return readers[ext]()


def find_stratification_columns(df, sample_count):
    candidates = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        n_unique = series.nunique()

        # skip columns with too many unique values
        if n_unique > min(100, n_rows * 0.5):
            continue

        # skip columns with only 1 unique value (no variance)
        if n_unique < 2:
            continue

        # skip long text columns
        if series.dtype == object:
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 50:
                continue

        # skip high-cardinality numeric columns (likely IDs or continuous)
        if pd.api.types.is_numeric_dtype(series):
            if n_unique > min(20, n_rows * 0.3):
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

    return selected


def print_distribution(df, col, label="Source"):
    """Print a terminal-friendly bar chart of a column's distribution."""
    counts = df[col].value_counts().sort_index()
    total = counts.sum()
    max_count = counts.max()
    max_label_len = max(len(str(v)) for v in counts.index)

    print(f"\n  {label} distribution for '{col}' ({counts.shape[0]} categories):")
    print(f"  {'─' * (max_label_len + BAR_WIDTH + 20)}")

    for value, count in counts.items():
        pct = count / total * 100
        bar_len = int(count / max_count * BAR_WIDTH) if max_count > 0 else 0
        bar = "█" * bar_len + "░" * (BAR_WIDTH - bar_len)
        print(f"  {str(value):>{max_label_len}}  {bar}  {count:>6} ({pct:5.1f}%)")

    print(f"  {'─' * (max_label_len + BAR_WIDTH + 20)}")


def print_stratification_report(df_original, df_sample, strat_cols, group_sizes=None, allocations=None, sample_count=None):
    """Print side-by-side distribution comparison for each stratification column."""
    print("\n╔═════════════════════════════════════════════════════════════════════╗")
    print(  "║                        STRATIFICATION REPORT                        ║")
    print(  "╚═════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Columns used: {', '.join(strat_cols)}")

    if len(strat_cols) > 1:
        print(
            f"\n  Method: rows are grouped by the intersection of all stratification\n"
            f"  columns simultaneously (e.g. A=x AND B=y AND C=z). Proportional\n"
            f"  allocation is then computed per intersection group. A category value\n"
            f"  that is small in every intersection group it appears in will receive\n"
            f"  0 allocation — even if it looks sizable in the per-column view below."
        )
    else:
        print(
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

        print(f"\n  Column: '{col}' ({len(all_values)} categories)")
        header = f"  {'Value':>{max_label_len}}  {'Original':^{half_bar + 10}}  {'Sample':^{half_bar + 10}}"
        print(header)
        print(f"  {'─' * len(header)}")

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
            print(
                f"  {label:>{max_label_len}}  "
                f"{o_bar} {o_pct:5.1f}%  "
                f"{s_bar} {s_pct:5.1f}%{flag}"
            )

        print(f"  {'─' * len(header)}")
        print(f"  {'Totals':>{max_label_len}}  {' ' * half_bar} {orig_total:>5}   {' ' * half_bar} {samp_total:>5}")

        if zero_rep and group_sizes is not None and allocations is not None and sample_count is not None:
            total_rows = int(group_sizes.sum())
            # min group size needed to receive at least 1 sample
            min_size_for_one = math.ceil(total_rows / sample_count)

            print(
                f"\n  Warning: {len(zero_rep)} of {len(all_values)} categories in '{col}' "
                f"received 0 samples.\n"
                f"  (A group needs ≥ {min_size_for_one} rows to receive ≥ 1 sample at this scale.)"
            )

            if len(strat_cols) > 1:
                # For each zero-represented value, look up its intersection sub-groups
                # group_sizes index is a MultiIndex; normalise to always use tuples
                col_idx = strat_cols.index(col)
                for value in zero_rep:
                    if isinstance(group_sizes.index, pd.MultiIndex):
                        mask = group_sizes.index.get_level_values(col_idx) == value
                        sub_sizes = group_sizes[mask]
                        sub_allocs = allocations[mask]
                    else:
                        # single-col (shouldn't happen here, but guard anyway)
                        sub_sizes = group_sizes[[value]] if value in group_sizes.index else pd.Series(dtype=int)
                        sub_allocs = allocations[[value]] if value in allocations.index else pd.Series(dtype=int)

                    n_subgroups = len(sub_sizes)
                    largest = int(sub_sizes.max()) if n_subgroups > 0 else 0
                    total_alloc = int(sub_allocs.sum())

                    print(
                        f"    {str(value):>{max_label_len}}: {n_subgroups} intersection group(s), "
                        f"largest has {largest} rows (needs {min_size_for_one}), "
                        f"total allocated = {total_alloc}"
                    )


def stratified_sample(df, count, strat_cols):
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
        samples.append(group_df.sample(n=n))

    result = pd.concat(samples).sample(frac=1)  # shuffle final result

    # adjust if rounding left us short/over
    if len(result) < count:
        remaining = df.drop(result.index)
        extra = remaining.sample(n=min(count - len(result), len(remaining)))
        result = pd.concat([result, extra])
    elif len(result) > count:
        result = result.head(count)

    return result, group_sizes, allocations


def sample(df, count, use_random=False):
    if count >= len(df):
        print(f"Warning: requested {count} samples but file only has {len(df)} rows. Returning all rows.")
        return df

    if use_random:
        print("Mode: pure random sampling")
        return df.sample(n=count)

    strat_cols = find_stratification_columns(df, count)

    if not strat_cols:
        print("No suitable columns for stratification found. Using pure random sampling.")
        return df.sample(n=count)

    print(f"Stratifying on columns: {strat_cols}")
    result, group_sizes, allocations = stratified_sample(df, count, strat_cols)
    print_stratification_report(df, result, strat_cols, group_sizes, allocations, count)
    return result


def save_output(df, source_path, count, output_folder=None):
    p = Path(source_path)
    out_name = f"{p.stem}_sample_{count}{p.suffix}"
    out_dir = Path(output_folder) if output_folder else p.parent
    out_path = out_dir / out_name
    ext = p.suffix.lower()

    out_dir.mkdir(parents=True, exist_ok=True)
    if ext == ".csv":
        df.to_csv(out_path, index=False)
    elif ext == ".tsv":
        df.to_csv(out_path, sep="\t", index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(out_path, index=False)
    elif ext == ".json":
        df.to_json(out_path, orient="records", indent=2)
    elif ext == ".parquet":
        df.to_parquet(out_path, index=False)

    print(f"\nOutput saved to: {out_path}")
    return out_path


def interactive_mode():
    source = input("Source file path: ").strip().strip('"').strip("'")
    if not os.path.isfile(source):
        print(f"Error: file not found: {source}")
        sys.exit(1)

    count = int(input("Number of samples: ").strip())

    sheet = None
    ext = Path(source).suffix.lower()
    if ext in (".xls", ".xlsx"):
        sheet = input("Sheet name (leave blank for first sheet): ").strip() or None

    use_random = input("Use pure random sampling? (y/n, default: n): ").strip().lower() == "y"

    return source, count, sheet, use_random


def main():
    outdir = None
    if len(sys.argv) <= 1:
        source, count, sheet, use_random = interactive_mode()
    else:
        parser = argparse.ArgumentParser(description="Create representative samples from data files.")
        parser.add_argument("source", help="Path to the source data file")
        parser.add_argument("count", type=int, help="Number of rows to sample")
        parser.add_argument("--sheet", default=None, help="Sheet name for Excel files (default: first sheet)")
        parser.add_argument("--random", action="store_true", dest="use_random", help="Use pure random sampling instead of stratified")
        parser.add_argument("--outdir", default=None, help="Output folder (default: same folder as source file)")
        args = parser.parse_args()
        source, count, sheet, use_random, outdir = args.source, args.count, args.sheet, args.use_random, args.outdir

    df = load_file(source, sheet=sheet)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {source}")

    result = sample(df, count, use_random=use_random)
    print(f"\nSampled {len(result)} rows.")

    save_output(result, source, count, output_folder=outdir)


if __name__ == "__main__":
    main()
