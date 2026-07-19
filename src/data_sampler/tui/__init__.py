"""Terminal UI for data-sampler (Textual)."""

from __future__ import annotations


def run_tui(path: str | None = None, sheet: str | None = None) -> None:
    """Launch the interactive terminal UI.

    ``path`` (optional) pre-loads a data file, skipping the file picker.
    """
    from .app import DataSamplerApp

    DataSamplerApp(path=path, sheet=sheet).run()


__all__ = ["run_tui"]
