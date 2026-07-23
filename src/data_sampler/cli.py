"""Command-line interface.

``data-sampler`` with no arguments launches the TUI. With a source file and
count it runs headless:

    data-sampler data.csv 500 --sheet S --outdir out --random --seed 7 \\
        --skip region,tier --anon "name=names" \\
        --anon "cust_id=sequential_id:start=1000,interval=7" \\
        --anon "salary=numeric_jitter:pct=0.1" --anon "email=hex:length=12"

For inputs too large for memory, ``--engine duckdb`` (or ``--engine auto``,
which picks DuckDB for Parquet/large files) samples out-of-core in parallel:

    data-sampler huge.parquet 10000 --engine duckdb --threads 8 \\
        --memory-limit 8GB --suggest
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from ._logging import get_logger
from .anonymize import anonymize, make_anonymizer
from .io import load_file, save_output
from .report import format_column_histograms, format_stratification_report
from .sampling import sample

log = get_logger(__name__)


def _coerce_value(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def parse_anon_option(text: str) -> tuple[str, str, dict]:
    """Parse ``COL=KIND[:key=value,key=value]`` into (column, kind, options)."""
    col, sep, rhs = text.partition("=")
    if not sep or not col.strip() or not rhs.strip():
        raise ValueError(
            f"--anon expects COL=KIND[:key=value,...], got {text!r}"
        )
    kind, _, optstr = rhs.partition(":")
    options: dict = {}
    if optstr:
        for pair in optstr.split(","):
            key, psep, value = pair.partition("=")
            if not psep or not key.strip():
                raise ValueError(
                    f"--anon option {pair!r} must be key=value (in {text!r})"
                )
            options[key.strip()] = _coerce_value(value.strip())
    return col.strip(), kind.strip(), options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-sampler",
        description=(
            "Create representative (stratified) samples from data files, "
            "optionally anonymizing columns. Run with no arguments to open "
            "the terminal UI."
        ),
    )
    parser.add_argument(
        "source", nargs="?",
        help="path to the source data file (omit to launch the TUI)",
    )
    parser.add_argument(
        "count", nargs="?", type=int, help="number of rows to sample",
    )
    parser.add_argument("--sheet", help="sheet name for Excel files (default: first)")
    parser.add_argument(
        "--random", action="store_true", dest="use_random",
        help="pure random sampling instead of stratified",
    )
    parser.add_argument("--outdir", help="output folder (default: source folder)")
    parser.add_argument("--seed", type=int, help="seed for reproducible runs")
    parser.add_argument(
        "--skip", action="append", default=[], metavar="COL[,COL...]",
        help="column(s) to exclude from stratification (repeatable)",
    )
    parser.add_argument(
        "--anon", action="append", default=[], metavar="COL=KIND[:k=v,...]",
        help=(
            "anonymize a column (repeatable). KIND: names, sequential_id, "
            "numeric_jitter, datetime_jitter, random_string, hex. "
            "Example: --anon \"id=sequential_id:start=1000,interval=7\""
        ),
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help=(
            "guided anonymization workflow: choose a type for each column from "
            "a menu (seeded by any --anon flags and per-column suggestions)"
        ),
    )
    parser.add_argument(
        "--suggest", action="store_true",
        help=(
            "auto-assign a suggested anonymizer type to each column from its "
            "stats (columns set via --anon are left as given)"
        ),
    )
    parser.add_argument(
        "--engine", choices=("auto", "pandas", "duckdb"), default="auto",
        help=(
            "sampling engine: 'pandas' (in-memory, default for small/medium data), "
            "'duckdb' (out-of-core, parallel; needs the 'large' extra), or 'auto' "
            "(duckdb for Parquet/large inputs, pandas otherwise)"
        ),
    )
    parser.add_argument(
        "--threads", type=int, metavar="N",
        help="DuckDB engine: number of threads (default: all cores)",
    )
    parser.add_argument(
        "--memory-limit", metavar="SIZE", dest="memory_limit",
        help="DuckDB engine: memory limit before spilling to disk (e.g. '8GB')",
    )
    parser.add_argument(
        "--tui", action="store_true",
        help="open the TUI (optionally preloading SOURCE)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _build_engine_spec(args, parser, engine, cols) -> dict:
    """Build the anonymizer spec for the engine path (validates against ``cols``,
    supports explicit --anon and --suggest via the engine's approximate stats)."""
    anon_specs: list[tuple[str, str, dict]] = []
    for raw in args.anon:
        try:
            col, kind, options = parse_anon_option(raw)
            if col not in cols:
                raise ValueError(f"column {col!r} not found in file")
            anon_specs.append((col, kind, options))
        except (ValueError, TypeError) as exc:
            parser.error(f"--anon {raw!r}: {exc}")

    if args.suggest:
        from .workflow import AnonymizationPlan, suggest_type

        plan = AnonymizationPlan.for_columns(cols)
        try:
            for col, kind, options in anon_specs:
                plan.assign(col, kind, **options)
        except (ValueError, TypeError) as exc:
            parser.error(f"--anon: {exc}")
        # approximate stats over the full (possibly huge) source, cheaply
        for cs in engine.stats(args.source, distributions=False):
            if plan.type_of(cs.name) == "none":
                plan.assignments[cs.name] = (suggest_type(cs), {})
        return plan.build_spec()

    spec = {}
    for col, kind, options in anon_specs:
        try:
            spec[col] = make_anonymizer(kind, **options)
        except (ValueError, TypeError) as exc:
            parser.error(f"--anon {col}={kind}: {exc}")
    return spec


def _run_with_engine(args, parser, skip) -> int:
    """Headless sampling via the DuckDB out-of-core engine."""
    from .engine import DuckDBEngine

    with DuckDBEngine(threads=args.threads, memory_limit=args.memory_limit) as engine:
        cols = engine.columns(args.source)
        total = engine.row_count(args.source)
        print(
            f"Loaded {total:,} rows, {len(cols)} columns from {args.source} "
            f"(DuckDB engine, {engine.threads} threads)"
        )
        unknown_skip = [c for c in skip if c not in cols]
        if unknown_skip:
            parser.error(
                f"--skip: column(s) not found in file: {', '.join(unknown_skip)}"
            )
        spec = _build_engine_spec(args, parser, engine, cols)
        result = engine.sample(
            args.source, args.count,
            use_random=args.use_random, exclude_columns=skip, seed=args.seed,
        )

    for note in result.notes:
        print(note)
    if skip:
        print(f"Columns excluded from stratification: {', '.join(skip)}")
    if result.method == "stratified" and result.group_sizes is not None:
        print(
            f"  ({len(result.group_sizes)} strata; "
            f"allocations sum to {int(result.allocations.sum())})"
        )

    data = result.data
    if spec:
        try:
            data = anonymize(data, spec, seed=args.seed)
        except (ValueError, TypeError) as exc:
            parser.error(f"anonymization failed: {exc}")
        print(f"Anonymized columns: {', '.join(spec)}")

    tag = f"sample_{args.count}" + ("_anon" if spec else "")
    out_path = save_output(data, args.source, tag, output_folder=args.outdir)
    print(f"\nSampled {len(data):,} rows.")
    print(f"Output saved to: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # ensure UTF-8 output on Windows (stdout is None in windowed frozen apps)
    if sys.stdout is not None and sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.tui or args.source is None:
        from .tui import run_tui

        run_tui(path=args.source, sheet=args.sheet)
        return 0

    if args.count is None:
        parser.error("count is required when a source file is given (or use --tui)")

    skip = [c.strip() for chunk in args.skip for c in chunk.split(",") if c.strip()]

    # ── engine selection ──────────────────────────────────────────────────────
    from .engine import (
        DuckDBUnavailable,
        duckdb_available,
        large_materialization_warning,
        should_use_engine,
    )

    if args.engine == "duckdb":
        use_engine = True
    elif args.engine == "auto":
        use_engine = should_use_engine(args.source)
    else:
        use_engine = False

    if use_engine:
        if args.interactive:
            parser.error(
                "--interactive is not supported with the DuckDB engine; "
                "use --engine pandas, or --suggest"
            )
        try:
            return _run_with_engine(args, parser, skip)
        except (DuckDBUnavailable, ValueError) as exc:
            # ValueError: source DuckDB can't read (unsupported format,
            # columns-oriented JSON, ...)
            if args.engine == "duckdb":
                parser.error(str(exc))
            print(f"Note: DuckDB engine declined ({exc}); using pandas.")
            use_engine = False  # auto: fall back to pandas
        except Exception as exc:
            # real DuckDB read/parse failures (InvalidInputException etc.) are
            # duckdb.Error subclasses, NOT ValueError — under auto they must
            # also fall back to pandas instead of escaping as a traceback
            if not type(exc).__module__.startswith("duckdb"):
                raise
            if args.engine == "duckdb":
                parser.error(f"DuckDB engine failed: {exc}")
            print(f"Note: DuckDB engine failed ({type(exc).__name__}); using pandas.")
            use_engine = False

    df = load_file(args.source, sheet=args.sheet)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {args.source}")
    warn = large_materialization_warning(len(df), len(df.columns))
    if warn and duckdb_available():
        print(f"Note: {warn}")
    unknown_skip = [c for c in skip if c not in df.columns]
    if unknown_skip:
        parser.error(f"--skip: column(s) not found in file: {', '.join(unknown_skip)}")

    # parse explicit --anon flags into (column, kind, options) triples
    anon_specs: list[tuple[str, str, dict]] = []
    for raw in args.anon:
        try:
            col, kind, options = parse_anon_option(raw)
            if col not in df.columns:
                raise ValueError(f"column {col!r} not found in file")
            anon_specs.append((col, kind, options))
        except (ValueError, TypeError) as exc:
            parser.error(f"--anon {raw!r}: {exc}")

    if args.interactive or args.suggest:
        from .workflow import AnonymizationPlan

        plan = AnonymizationPlan.for_columns(df.columns)
        try:
            for col, kind, options in anon_specs:
                plan.assign(col, kind, **options)
        except (ValueError, TypeError) as exc:
            parser.error(f"--anon: {exc}")
        if args.suggest:
            suggested = AnonymizationPlan.suggest(df)
            for col in df.columns:
                if plan.type_of(col) == "none":
                    plan.assignments[col] = suggested.assignments[col]
        if args.interactive:
            plan.choose_interactively(df)
        spec = plan.build_spec()
    else:
        spec = {}
        for col, kind, options in anon_specs:
            try:
                spec[col] = make_anonymizer(kind, **options)
            except (ValueError, TypeError) as exc:
                parser.error(f"--anon {col}={kind}: {exc}")

    result = sample(
        df, args.count,
        use_random=args.use_random,
        exclude_columns=skip,
        random_state=args.seed,
    )
    for note in result.notes:
        print(note)
    if skip:
        print(f"Columns excluded from stratification: {', '.join(skip)}")
    if result.method == "stratified":
        print(format_stratification_report(df, result))

    # per-column source-vs-sample histograms (only when a real sample was taken)
    if len(result.data) < len(df):
        hist = format_column_histograms(df, result.data)
        if hist:
            print(hist)

    data = result.data
    if spec:
        try:
            data = anonymize(data, spec, seed=args.seed)
        except (ValueError, TypeError) as exc:
            parser.error(f"anonymization failed: {exc}")
        print(f"Anonymized columns: {', '.join(spec)}")

    tag = f"sample_{args.count}" + ("_anon" if spec else "")
    out_path = save_output(data, args.source, tag, output_folder=args.outdir)
    print(f"\nSampled {len(data)} rows.")
    print(f"Output saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
