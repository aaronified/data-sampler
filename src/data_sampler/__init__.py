"""data-sampler: representative sampling and anonymization for tabular data.

Public API::

    import data_sampler as ds

    df = ds.load_file("data.csv")
    stats = ds.compute_stats(df)                 # Data Wrangler-style column stats
    result = ds.sample(df, 500, exclude_columns=["notes"])
    print(ds.format_stratification_report(df, result))
    anon = ds.anonymize(result.data, {"name": "names", "id": ("sequential_id", {"start": 1000})})
    ds.save_output(anon, "data.csv", tag="sample_500_anon")

    ds.run_tui()                                 # launch the terminal UI
"""

from __future__ import annotations

__version__ = "3.0.1"

from .io import SUPPORTED_EXTENSIONS, list_sheets, load_file, save_output
from .report import format_distribution, format_stratification_report
from .sampling import (
    SampleResult,
    find_stratification_columns,
    sample,
    stratified_sample,
)
from .stats import ColumnStats, compute_column_stats, compute_stats, sparkline

__all__ = [
    "__version__",
    "SUPPORTED_EXTENSIONS",
    "load_file",
    "list_sheets",
    "save_output",
    "ColumnStats",
    "compute_stats",
    "compute_column_stats",
    "sparkline",
    "SampleResult",
    "sample",
    "stratified_sample",
    "find_stratification_columns",
    "format_stratification_report",
    "format_distribution",
    "anonymize",
    "make_anonymizer",
    "AnonymizationPlan",
    "suggest_type",
    "TYPE_OPTIONS",
    "run_tui",
]


def __getattr__(name: str):
    # Lazy imports: keep pandas-only workflows fast and let the core API work
    # even while optional pieces (TUI) are not yet importable.
    if name in ("anonymize", "make_anonymizer"):
        # importlib, not `from . import anonymize`: the fromlist form calls
        # hasattr(package, "anonymize"), which re-enters this __getattr__.
        import importlib

        _anon = importlib.import_module(".anonymize", __name__)
        return getattr(_anon, name)
    if name in ("AnonymizationPlan", "suggest_type", "TYPE_OPTIONS"):
        import importlib

        _wf = importlib.import_module(".workflow", __name__)
        return getattr(_wf, name)
    if name == "run_tui":
        from .tui import run_tui

        return run_tui
    raise AttributeError(f"module 'data_sampler' has no attribute {name!r}")
