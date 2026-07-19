"""File loading and saving for all supported tabular formats."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".json", ".xlsx", ".xls", ".parquet")


def load_file(filepath: str | Path, sheet: str | None = None) -> pd.DataFrame:
    """Load a data file into a DataFrame.

    Supports CSV, TSV, JSON, Excel (.xlsx/.xls) and Parquet. For Excel files,
    ``sheet`` selects the sheet by name (default: first sheet).
    """
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
    ``{stem}_{tag}{ext}`` (e.g. ``data_sample_500.csv``).
    """
    p = Path(source_path)
    out_name = f"{p.stem}_{tag}{p.suffix}"
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
    else:
        raise ValueError(f"Unsupported output file type: '{ext}'")

    return out_path
