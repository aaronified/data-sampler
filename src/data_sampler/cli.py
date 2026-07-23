"""Command-line interface.

``data-sampler`` with no arguments launches the TUI. With a source file and
count it runs headless:

    data-sampler data.csv 500 --sheet S --outdir out --random --seed 7 \\
        --skip region,tier --anon "name=names" \\
        --anon "cust_id=sequential_id:start=1000,interval=7" \\
        --anon "salary=numeric_jitter:pct=0.1" --anon "email=hex:length=12"
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from ._logging import get_logger
from .anonymize import anonymize, make_anonymizer
from .io import load_file, save_output
from .report import format_stratification_report
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
        "--tui", action="store_true",
        help="open the TUI (optionally preloading SOURCE)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


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

    df = load_file(args.source, sheet=args.sheet)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {args.source}")

    spec = {}
    for raw in args.anon:
        try:
            col, kind, options = parse_anon_option(raw)
            if col not in df.columns:
                raise ValueError(f"column {col!r} not found in file")
            spec[col] = make_anonymizer(kind, **options)
        except (ValueError, TypeError) as exc:
            parser.error(f"--anon {raw!r}: {exc}")

    skip = [c.strip() for chunk in args.skip for c in chunk.split(",") if c.strip()]

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
