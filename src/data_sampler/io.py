"""File loading and saving for all supported tabular formats."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".json", ".xlsx", ".xls", ".parquet")


def is_url(filepath: str | Path) -> bool:
    """Whether ``filepath`` is an ``http(s)`` (or ``s3``) URL, not a local path."""
    return str(filepath).lower().startswith(("http://", "https://", "s3://"))


def load_file(filepath: str | Path, sheet: str | None = None) -> pd.DataFrame:
    """Load a data file into a DataFrame.

    Supports CSV, TSV, JSON, Excel (.xlsx/.xls) and Parquet, from a local path
    **or an http(s) URL** (e.g. a GitHub raw link) — pandas streams the URL, so
    it downloads the whole file into memory; for very large remote data use the
    out-of-core DuckDB engine instead. For Excel files, ``sheet`` selects the
    sheet by name (default: first sheet). The file type is taken from the
    extension (of the URL path, ignoring any query string).
    """
    # derive the extension from the URL path (not the query) for remote sources
    ext_source = urlparse(str(filepath)).path if is_url(filepath) else str(filepath)
    ext = Path(ext_source).suffix.lower()
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


def list_sheets(filepath: str | Path) -> list[str]:
    """Return the sheet names of an Excel file (empty list for other formats)."""
    ext = Path(filepath).suffix.lower()
    if ext not in (".xlsx", ".xls"):
        return []
    with pd.ExcelFile(filepath) as xls:
        return [str(s) for s in xls.sheet_names]


def save_output(
    df: pd.DataFrame,
    source_path: str | Path,
    tag: str,
    output_folder: str | Path | None = None,
) -> Path:
    """Save ``df`` next to ``source_path`` (or into ``output_folder``).

    The output keeps the source file's format and is named
    ``{stem}_{tag}{ext}`` (e.g. ``data_sample_500.csv``). When ``source_path``
    is a URL the name comes from the URL's path and, without an explicit
    ``output_folder``, the file is written to the current directory (a URL has
    no meaningful local parent).
    """
    if is_url(source_path):
        p = Path(urlparse(str(source_path)).path)
        default_dir = Path.cwd()
    else:
        p = Path(source_path)
        default_dir = p.parent
    out_name = f"{p.stem}_{tag}{p.suffix}"
    out_dir = Path(output_folder) if output_folder else default_dir
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
    else:
        raise ValueError(f"Unsupported output file type: '{ext}'")

    return out_path
