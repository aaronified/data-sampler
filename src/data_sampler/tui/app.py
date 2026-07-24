"""The data-sampler terminal UI.

A panel-based, color-coded dashboard (in the spirit of btop / lazydocker):

- File screen — pick a data file (typed path or directory browser).
- Columns screen — Data Wrangler-style column stats table, a detail panel
  with distribution bars, per-column anonymizer configuration, and
  stratification skip toggles.
- Report screen — stratification report, anonymization summary, output path.
"""

from __future__ import annotations

import copy
import os
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    DirectoryTree,
    Footer,
    Input,
    Label,
    Select,
    Static,
    Switch,
)

from .. import __version__, _names
from .._logging import get_logger, redirect_to_file
from ..anonymize import make_anonymizer, suggest_ethnicity_mapping, suggest_gender_mapping
from ..io import is_url as io_is_url, load_file, save_output
from ..reduce import reduce_columns
from ..report import format_reduction_report, format_stratification_report
from ..sampling import sample
from ..stats import ColumnStats, _fmt_num, compute_stats, sparkline

log = get_logger(__name__)

# ── palette ───────────────────────────────────────────────────────────────────

CYAN = "#00d7ff"
MAGENTA = "#ff5fd7"
GREEN = "#5fff87"
YELLOW = "#ffd75f"
ORANGE = "#ff875f"
RED = "#ff5f5f"
DIM = "#5f6b85"
FG = "#c8d3f5"
# input chrome: a muted resting outline that brightens to CYAN on focus, over a
# panel-distinct fill — the btop-style "quiet until focused" treatment
BORDER = "#2b3550"
INPUT_BG = "#121826"
INPUT_BG_FOCUS = "#16202e"

KIND_COLORS = {
    "numeric": CYAN,
    "categorical": MAGENTA,
    "boolean": YELLOW,
    "datetime": ORANGE,
    "text": GREEN,
    "other": DIM,
}

ANON_CHOICES = [
    ("none", "none"),
    ("names (from name library)", "names"),
    ("sequential id (start + interval)", "sequential_id"),
    ("numeric jitter (± percent)", "numeric_jitter"),
    ("datetime jitter (± window)", "datetime_jitter"),
    ("random string", "random_string"),
    ("hex string", "hex"),
]

ANON_SHORT = {
    "none": "—",
    "names": "names",
    "sequential_id": "seq id",
    "numeric_jitter": "jitter",
    "datetime_jitter": "date jit",
    "random_string": "string",
    "hex": "hex",
}

# PCA column-reduction modes for the run bar: off, an exact component count,
# or a target share of variance to retain (labels stay short so the run bar
# still fits an 80-120 column terminal)
REDUCE_CHOICES = [
    ("no reduction", "off"),
    ("N components", "components"),
    ("variance ≥ R", "variance"),
]

# names anonymizer: gender + ethnicity for the replacement names
GENDER_CHOICES = [
    ("mixed (any)", "mixed"),
    ("male", "male"),
    ("female", "female"),
    ("third gender", "third"),
    ("not disclosed", "undisclosed"),
    ("from a column…", "column"),
]


def _pretty_ethnicity(key: str) -> str:
    return key.replace("_", " ").title()


ETHNICITY_CHOICES = (
    [("all / mixed", "all")]
    + [(_pretty_ethnicity(e), e) for e in _names.ETHNICITIES]
    + [("from a column…", "column")]
)

# gender targets offered when manually mapping a gender column's values
GENDER_MAP_TARGETS = [
    ("male", "male"), ("female", "female"), ("third gender", "third"),
    ("not disclosed", "undisclosed"),
]
ETHNICITY_MAP_TARGETS = [(_pretty_ethnicity(e), e) for e in _names.ETHNICITIES]


@dataclass
class ColumnConfig:
    """Per-column TUI state: anonymizer + stratification / reduction skips."""

    kind: str = "none"
    options: dict[str, str] = field(default_factory=dict)
    skip_strat: bool = False
    skip_reduce: bool = False
    # names anonymizer only: "mixed" | male | female | third | undisclosed | "column"
    gender: str = "mixed"
    # names anonymizer only: "all" | <ethnicity> | "column"
    ethnicity: str = "all"
    gender_column: str = ""     # used when gender == "column"
    ethnicity_column: str = ""  # used when ethnicity == "column"
    randomize_gender: bool = False
    gender_map: dict[str, str] = field(default_factory=dict)     # raw value -> gender
    ethnicity_map: dict[str, str] = field(default_factory=dict)  # raw value -> ethnicity


def _names_kwargs(cfg: ColumnConfig) -> dict:
    """Build the ``NameAnonymizer`` kwargs a names config maps to — shared by
    :func:`build_anonymizer` (instance) and :func:`anon_api_kwargs` (snippet),
    so the reproduce-in-Python code always matches what actually ran."""
    kw: dict = {"style": cfg.options.get("style") or "first_last"}
    if cfg.gender == "column" and cfg.gender_column:
        kw["gender_column"] = cfg.gender_column
        if cfg.gender_map:
            kw["gender_map"] = dict(cfg.gender_map)
        if cfg.randomize_gender:
            kw["randomize_gender"] = True
    elif cfg.gender not in ("mixed", "column"):
        kw["gender"] = cfg.gender
    if cfg.ethnicity == "column" and cfg.ethnicity_column:
        kw["ethnicity_column"] = cfg.ethnicity_column
        if cfg.ethnicity_map:
            kw["ethnicity_map"] = dict(cfg.ethnicity_map)
    elif cfg.ethnicity not in ("all", "column"):
        kw["ethnicity"] = cfg.ethnicity
    return kw


def build_anonymizer(cfg: ColumnConfig):
    """Turn a ColumnConfig's raw option strings into an anonymizer instance."""
    o = cfg.options
    if cfg.kind == "names":
        return make_anonymizer("names", **_names_kwargs(cfg))
    if cfg.kind == "sequential_id":
        return make_anonymizer(
            "sequential_id",
            start=int(o.get("start") or 1),
            interval=int(o.get("interval") or 1),
            prefix=o.get("prefix") or "",
            width=int(o.get("width") or 0),
        )
    if cfg.kind == "numeric_jitter":
        kwargs = {"pct": float(o.get("pct") or 20) / 100.0}
        if (o.get("round_to") or "").strip():
            kwargs["round_to"] = int(o["round_to"])
        return make_anonymizer("numeric_jitter", **kwargs)
    if cfg.kind == "datetime_jitter":
        return make_anonymizer(
            "datetime_jitter",
            max_delta=(o.get("max_delta") or "7D").strip(),
            unit=(o.get("unit") or "s").strip(),
        )
    if cfg.kind == "random_string":
        return make_anonymizer(
            "random_string",
            length=int(o.get("length") or 8),
            prefix=o.get("prefix") or "",
        )
    if cfg.kind == "hex":
        return make_anonymizer("hex", length=int(o.get("length") or 8))
    raise ValueError(f"No anonymizer configured ({cfg.kind!r})")


def anon_api_kwargs(cfg: ColumnConfig) -> tuple[str, dict]:
    """The ``(kind, kwargs)`` a column's config maps to in the public API —
    mirrors :func:`build_anonymizer` so the report's copy-paste snippet
    reproduces the run exactly (e.g. ``pct`` 20 → ``0.2``)."""
    o = cfg.options
    if cfg.kind == "names":
        return "names", _names_kwargs(cfg)
    if cfg.kind == "sequential_id":
        kw: dict = {"start": int(o.get("start") or 1), "interval": int(o.get("interval") or 1)}
        if (o.get("prefix") or ""):
            kw["prefix"] = o["prefix"]
        if int(o.get("width") or 0):
            kw["width"] = int(o["width"])
        return "sequential_id", kw
    if cfg.kind == "numeric_jitter":
        kw = {"pct": float(o.get("pct") or 20) / 100.0}
        if (o.get("round_to") or "").strip():
            kw["round_to"] = int(o["round_to"])
        return "numeric_jitter", kw
    if cfg.kind == "datetime_jitter":
        return "datetime_jitter", {
            "max_delta": (o.get("max_delta") or "7D").strip(),
            "unit": (o.get("unit") or "s").strip(),
        }
    if cfg.kind == "random_string":
        kw = {"length": int(o.get("length") or 8)}
        if (o.get("prefix") or ""):
            kw["prefix"] = o["prefix"]
        return "random_string", kw
    if cfg.kind == "hex":
        return "hex", {"length": int(o.get("length") or 8)}
    return cfg.kind, {}


def anon_label(cfg: ColumnConfig) -> str:
    """Short table-cell label for a column's anonymizer config."""
    if cfg.kind == "none":
        return "—"
    o = cfg.options
    if cfg.kind == "sequential_id":
        return f"seq {o.get('start') or 1}+{o.get('interval') or 1}"
    if cfg.kind == "numeric_jitter":
        return f"jitter ±{o.get('pct') or 20}%"
    if cfg.kind == "datetime_jitter":
        return f"date ±{o.get('max_delta') or '7D'}"
    if cfg.kind in ("random_string", "hex"):
        return f"{ANON_SHORT[cfg.kind]}[{o.get('length') or 8}]"
    return ANON_SHORT[cfg.kind]


def _fmt_stat(x: object | None) -> str:
    """Table-cell rendering for a single summary statistic.

    Numbers go through the shared numeric formatter; a categorical mode (a
    string) is shown verbatim but truncated so it can't blow out the column.
    """
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return _fmt_num(float(x))
    s = str(x)
    return s if len(s) <= 14 else s[:13] + "…"


# ── screens ───────────────────────────────────────────────────────────────────


class FileScreen(Screen):
    """Pick the source data file."""

    BINDINGS = [
        Binding("ctrl+l", "load", "load file"),
        Binding("ctrl+r", "refresh_browser", "refresh browser"),
    ]

    def __init__(self, path: str | None = None, sheet: str | None = None):
        super().__init__()
        self._initial_path = path
        self._initial_sheet = sheet

    def compose(self) -> ComposeResult:
        yield Static(
            f" ▓▒░ DATA SAMPLER ░▒▓  v{__version__} — representative samples, "
            "optionally anonymized",
            id="titlebar",
        )
        with Horizontal(id="file-main"):
            with VerticalScroll(id="file-form"):
                yield Label("file path or URL", classes="field-label")
                yield Input(
                    placeholder="local path or http(s) URL — .csv / .tsv / .json / .xlsx / .parquet",
                    id="path",
                )
                yield Label("excel sheet (blank = first)", classes="field-label")
                yield Input(placeholder="sheet name", id="sheet")
                yield Button("▶ load", id="load", variant="success")
                yield Static("", id="file-status")
                yield Label("names library (for the names anonymizer)", classes="field-label")
                yield Static(
                    "swap in your own names: export the current library, edit "
                    "it, then load it back (this session) — or set it permanently.",
                    id="names-lib-note",
                )
                yield Input(
                    placeholder="path to a custom names .py (load / export target)",
                    id="names-path",
                )
                with Horizontal(id="names-lib-row"):
                    yield Button("⬇ export sample", id="export-names", classes="mini-btn")
                    yield Button("⬆ load custom", id="load-names", classes="mini-btn")
                    yield Button("install permanently", id="install-names", classes="mini-btn")
                yield Static("", id="names-lib-status")
            with Vertical(id="file-browser"):
                yield Button("⟳ refresh", id="refresh-browser", classes="mini-btn")
                yield DirectoryTree(str(Path.cwd()), id="browser")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#file-form").border_title = "source file"
        self.query_one("#file-browser").border_title = "browse"
        if self._initial_path:
            self.query_one("#path", Input).value = self._initial_path
            if self._initial_sheet:
                self.query_one("#sheet", Input).value = self._initial_sheet
            self.action_load()
        else:
            self.query_one("#path", Input).focus()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.query_one("#path", Input).value = str(event.path)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_load()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "load":
            self.action_load()
        elif event.button.id == "refresh-browser":
            self.action_refresh_browser()
        elif event.button.id == "export-names":
            self._names_lib_action("export")
        elif event.button.id == "load-names":
            self._names_lib_action("load")
        elif event.button.id == "install-names":
            self._names_lib_action("install")

    def _names_lib_action(self, which: str) -> None:
        """Export / load / install a custom names library from the home screen."""
        status = self.query_one("#names-lib-status", Static)
        raw = self.query_one("#names-path", Input).value.strip().strip('"').strip("'")
        try:
            if which == "export":
                target = raw or "data_sampler_names.py"
                _names.export_library(target)
                status.update(Text(f"✓ exported current library → {target}", style=GREEN))
            elif which == "load":
                if not raw or not os.path.isfile(raw):
                    status.update(Text("enter the path to a custom names .py to load", style=YELLOW))
                    return
                _names.load_library(path=raw)
                status.update(Text(
                    f"✓ loaded custom library ({len(_names.ETHNICITIES)} ethnicities, "
                    "this session)", style=GREEN,
                ))
            else:  # install permanently into the package
                if not raw or not os.path.isfile(raw):
                    status.update(Text("enter the path to a custom names .py to install", style=YELLOW))
                    return
                target = _names.install_library(raw)
                status.update(Text(f"✓ installed into the package: {target}", style=GREEN))
        except Exception as exc:  # surfaced, never crash
            log.exception("names library action failed")
            status.update(Text(f"✗ {type(exc).__name__}: {exc}", style=RED))

    def action_refresh_browser(self) -> None:
        """Re-scan the working directory so files created since launch appear."""
        self.query_one("#browser", DirectoryTree).reload()
        self.query_one("#file-status", Static).update(
            Text("↻ file browser refreshed", style=CYAN)
        )

    def action_load(self) -> None:
        path = self.query_one("#path", Input).value.strip().strip('"').strip("'")
        sheet = self.query_one("#sheet", Input).value.strip() or None
        status = self.query_one("#file-status", Static)
        if not path:
            status.update(Text("enter a file path or URL, or pick one from the browser", style=YELLOW))
            return
        # a local path must exist; an http(s)/s3 URL is passed straight through
        if not io_is_url(path) and not os.path.isfile(path):
            status.update(Text(f"file not found: {path}", style=RED))
            return
        status.update(Text("loading…", style=CYAN))
        self.query_one("#load", Button).disabled = True
        self.run_worker(lambda: self._load(path, sheet), thread=True, exclusive=True)

    def _load(self, path: str, sheet: str | None) -> None:
        app: DataSamplerApp = self.app  # type: ignore[assignment]
        try:
            df = load_file(path, sheet=sheet)
            # stats stay in this worker thread too: on large frames they take
            # seconds, which would freeze the UI if run in on_mount
            stats = compute_stats(df)
        except Exception as exc:  # surfaced to the user, never crash the TUI
            log.exception("load failed")
            app.call_from_thread(self._load_failed, exc)
            return
        app.call_from_thread(self._loaded, path, sheet, df, stats)

    def _load_failed(self, exc: Exception) -> None:
        self.query_one("#load", Button).disabled = False
        self.query_one("#file-status", Static).update(
            Text(f"✗ {type(exc).__name__}: {exc}", style=RED)
        )

    def _loaded(
        self, path: str, sheet: str | None, df: pd.DataFrame, stats: list | None = None
    ) -> None:
        self.query_one("#load", Button).disabled = False
        self.query_one("#file-status", Static).update(
            Text(f"✓ {len(df):,} rows × {len(df.columns)} columns", style=GREEN)
        )
        app: DataSamplerApp = self.app  # type: ignore[assignment]
        app.df = df
        app.source_path = path
        app.sheet = sheet
        app.column_stats = stats
        app.push_screen(ColumnsScreen())


class MappingScreen(ModalScreen):
    """Modal to map a column's distinct values to gender / ethnicity groups.

    Prefilled from auto-detection (:func:`suggest_gender_mapping` /
    :func:`suggest_ethnicity_mapping`); every value stays editable, so it is
    the manual fallback / override. Dismisses with a ``{raw value: target}``
    dict (unmapped values omitted), or ``None`` on cancel.
    """

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, title: str, values: list, targets: list, current: dict):
        super().__init__()
        self._title = title
        self._values = list(values)
        self._targets = targets
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="map-dialog"):
            yield Static(self._title, id="map-title")
            yield Static(
                "auto-detected below — override any row (blank = fall back)",
                id="map-help",
            )
            options = [("(skip)", "")] + list(self._targets)
            with VerticalScroll(id="map-list"):
                for i, val in enumerate(self._values):
                    with Horizontal(classes="map-row"):
                        yield Label(str(val)[:22], classes="map-val")
                        yield Select(
                            options, value=self._current.get(val) or "",
                            allow_blank=False, id=f"map-{i}",
                        )
            with Horizontal(id="map-actions"):
                yield Button("cancel (esc)", id="map-cancel")
                yield Button("apply", id="map-apply", variant="success")

    def on_mount(self) -> None:
        self.query_one("#map-dialog").border_title = "map values"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "map-cancel":
            self.action_cancel()
        elif event.button.id == "map-apply":
            mapping: dict = {}
            for i, val in enumerate(self._values):
                value = self.query_one(f"#map-{i}", Select).value
                if value:  # "" == the "(skip)" option → leave unmapped
                    mapping[val] = value
            self.dismiss(mapping)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ColumnsDataTable(DataTable):
    """DataTable that reports modifier-clicks so rows can be multi-selected.

    Textual's DataTable turns any click into a plain cursor move. We keep that
    behaviour for unmodified clicks, but a ctrl- or shift-click is intercepted
    and re-emitted as :class:`ModifierClick` (after moving the cursor to the
    clicked row) so the screen can toggle / range-select rows — the desktop
    ctrl/shift-click idiom, in the terminal.
    """

    class ModifierClick(Message, namespace="columns_table"):
        def __init__(self, row_key: str, ctrl: bool, shift: bool) -> None:
            super().__init__()
            self.row_key = row_key
            self.ctrl = ctrl
            self.shift = shift

    async def _on_click(self, event) -> None:
        if event.ctrl or event.shift:
            meta = event.style.meta
            if "row" in meta and "column" in meta:
                row_index, column_index = meta["row"], meta["column"]
                if row_index >= 0 and column_index >= 0:
                    try:
                        cell_key = self.coordinate_to_cell_key(
                            Coordinate(row_index, column_index)
                        )
                    except Exception:
                        cell_key = None
                    if cell_key is not None and cell_key.row_key.value is not None:
                        self.cursor_coordinate = Coordinate(row_index, column_index)
                        self.post_message(
                            self.ModifierClick(
                                cell_key.row_key.value,
                                bool(event.ctrl),
                                bool(event.shift),
                            )
                        )
                        event.stop()
                        return
        await super()._on_click(event)


class ColumnsScreen(Screen):
    """Column stats dashboard + anonymizer / stratification configuration."""

    BINDINGS = [
        Binding("ctrl+r", "run", "run sample"),
        Binding("a", "suggest", "auto-suggest types"),
        Binding("s", "toggle_skip", "toggle strat skip"),
        Binding("d", "toggle_reduce_skip", "toggle reduce skip"),
        Binding("space", "toggle_select", "select row"),
        Binding("x", "clear_select", "clear selection"),
        Binding("ctrl+z", "undo", "undo"),
        Binding("ctrl+y", "redo", "redo"),
        Binding("escape", "back", "back to file"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.configs: dict[str, ColumnConfig] = {}
        self.stats: dict[str, ColumnStats] = {}
        self.selected: str | None = None
        self._syncing = False
        self._col_keys: list = []
        # rows explicitly multi-selected (via ctrl/shift-click or space); bulk
        # config edits apply to these plus the cursor row. Kept separate from
        # the DataTable's single-row cursor.
        self.selection: set[str] = set()
        self._anchor: str | None = None
        # config-only undo/redo (deep-copied snapshots of self.configs). The
        # brief asks for ≥10 steps; a generous cap keeps the memory bounded.
        self._undo: deque[dict[str, ColumnConfig]] = deque(maxlen=100)
        self._redo: deque[dict[str, ColumnConfig]] = deque(maxlen=100)
        # coalesces bursts of edits on one field into a single undo step
        self._last_coalesce_key: object | None = None

    # ── layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        app: DataSamplerApp = self.app  # type: ignore[assignment]
        n_rows = len(app.df) if app.df is not None else 0
        n_cols = len(app.df.columns) if app.df is not None else 0
        # column choices for the "gender/ethnicity from a column" pickers
        col_choices = [
            (str(c), str(c)) for c in (app.df.columns if app.df is not None else [])
        ]
        yield Static(
            f" ▓▒░ DATA SAMPLER ░▒▓  {Path(app.source_path or '?').name}"
            f" · {n_rows:,} rows × {n_cols} columns",
            id="titlebar",
        )
        with Horizontal(id="main"):
            with Container(id="columns-panel"):
                yield ColumnsDataTable(id="columns-table")
            with Vertical(id="side"):
                with VerticalScroll(id="detail-panel"):
                    yield Static("", id="detail")
                with VerticalScroll(id="config-panel"):
                    yield Label("anonymizer", classes="field-label")
                    yield Select(
                        ANON_CHOICES, value="none", allow_blank=False, id="anon-kind"
                    )
                    with ContentSwitcher(initial="opts-none", id="anon-options"):
                        yield Static(
                            "column is written through unchanged",
                            id="opts-none", classes="opts-note",
                        )
                        with Vertical(id="opts-names"):
                            yield Label("style", classes="field-label")
                            yield Select(
                                [
                                    ("First Last", "first_last"),
                                    ("First Middle Last", "first_middle_last"),
                                    ("Last, First", "last_first"),
                                    ("First only", "first"),
                                    ("Last only", "last"),
                                ],
                                value="first_last",
                                allow_blank=False,
                                id="opt-names-style",
                            )
                            yield Label("gender", classes="field-label")
                            yield Select(
                                GENDER_CHOICES, value="mixed", allow_blank=False,
                                id="opt-names-gender",
                            )
                            with Vertical(id="opt-names-gender-col-row", classes="col-map hidden"):
                                yield Select(
                                    [("(no column)", "")] + col_choices, value="",
                                    allow_blank=False, id="opt-names-gender-col",
                                )
                                with Horizontal(classes="optrow"):
                                    yield Switch(value=False, id="opt-names-gender-random")
                                    yield Label("randomize gender", classes="opt-label-wide")
                                yield Button("map values…", id="btn-gender-map", classes="mini-btn")
                            yield Label("ethnicity", classes="field-label")
                            yield Select(
                                ETHNICITY_CHOICES, value="all", allow_blank=False,
                                id="opt-names-ethnicity",
                            )
                            with Vertical(id="opt-names-eth-col-row", classes="col-map hidden"):
                                yield Select(
                                    [("(no column)", "")] + col_choices, value="",
                                    allow_blank=False, id="opt-names-eth-col",
                                )
                                yield Button("map values…", id="btn-eth-map", classes="mini-btn")
                        with Vertical(id="opts-sequential_id"):
                            with Horizontal(classes="optrow"):
                                yield Label("start", classes="opt-label")
                                yield Input("1", id="opt-seq-start", classes="opt-input")
                                yield Label("interval", classes="opt-label")
                                yield Input("1", id="opt-seq-interval", classes="opt-input")
                            with Horizontal(classes="optrow"):
                                yield Label("prefix", classes="opt-label")
                                yield Input("", id="opt-seq-prefix", classes="opt-input")
                                yield Label("pad width", classes="opt-label")
                                yield Input("0", id="opt-seq-width", classes="opt-input")
                        with Vertical(id="opts-numeric_jitter"):
                            with Horizontal(classes="optrow"):
                                yield Label("± percent", classes="opt-label")
                                yield Input("20", id="opt-jit-pct", classes="opt-input")
                                yield Label("round to", classes="opt-label")
                                yield Input("", id="opt-jit-round", classes="opt-input")
                        with Vertical(id="opts-datetime_jitter"):
                            with Horizontal(classes="optrow"):
                                yield Label("± window", classes="opt-label")
                                yield Input("7D", id="opt-dt-delta", classes="opt-input")
                                yield Label("unit", classes="opt-label")
                                yield Input("s", id="opt-dt-unit", classes="opt-input")
                        with Vertical(id="opts-random_string"):
                            with Horizontal(classes="optrow"):
                                yield Label("length", classes="opt-label")
                                yield Input("8", id="opt-str-length", classes="opt-input")
                                yield Label("prefix", classes="opt-label")
                                yield Input("", id="opt-str-prefix", classes="opt-input")
                        with Vertical(id="opts-hex"):
                            with Horizontal(classes="optrow"):
                                yield Label("length", classes="opt-label")
                                yield Input("8", id="opt-hex-length", classes="opt-input")
                    with Horizontal(id="skip-row"):
                        yield Switch(value=False, id="skip-strat")
                        yield Label("skip when stratifying", id="skip-label")
                    with Horizontal(id="reduce-skip-row"):
                        yield Switch(value=False, id="skip-reduce")
                        yield Label("skip from reduction (PCA)", id="reduce-skip-label")
        with Horizontal(id="runbar"):
            yield Label("rows", classes="run-label")
            yield Input("100", id="count", classes="run-input-s")
            yield Label("out dir", classes="run-label")
            yield Input("", placeholder="same folder as source", id="outdir", classes="run-input-l")
            yield Label("seed", classes="run-label")
            yield Input("", placeholder="—", id="seed", classes="run-input-s")
            yield Label("random", classes="run-label")
            yield Switch(value=False, id="random")
            yield Label("reduce", classes="run-label")
            yield Select(REDUCE_CHOICES, value="off", allow_blank=False, id="reduce-mode")
            yield Input("", placeholder="—", id="reduce-value")
            yield Button("▶ run sample", id="run", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        app: DataSamplerApp = self.app  # type: ignore[assignment]
        self.query_one("#columns-panel").border_title = "columns"
        self.query_one("#detail-panel").border_title = "detail"
        self.query_one("#config-panel").border_title = "anonymize · stratify · reduce"
        self.query_one("#runbar").border_title = "sample"

        df = app.df
        # stats were precomputed in the file-load worker thread; the inline
        # compute_stats is only a fallback for screens pushed directly (tests)
        stats_list = app.column_stats if app.column_stats is not None else compute_stats(df)
        self.stats = {s.name: s for s in stats_list}
        self.configs = {str(c): ColumnConfig() for c in df.columns}

        table = self.query_one("#columns-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        # actionable columns (anonymizer, strat, reduce) come right after type
        # so they stay visible; the wider distribution and the per-stat summary
        # columns (mean / median / mode / sd) trail them
        self._col_keys = list(
            table.add_columns(
                "column", "type", "anonymizer", "strat", "reduce",
                "miss%", "uniq", "distribution",
                "mean", "median", "mode", "sd",
            )
        )
        for name in self.configs:
            table.add_row(*self._row_cells(name), key=name)
        table.focus()
        if self.configs:
            self.selected = next(iter(self.configs))
            self._show_detail(self.selected)

    # ── table rendering ───────────────────────────────────────────────────────

    def _row_cells(self, name: str) -> list:
        s = self.stats[name]
        cfg = self.configs[name]
        color = KIND_COLORS.get(s.kind, DIM)
        miss_style = RED if s.missing else DIM
        if cfg.skip_strat:
            strat = Text("✗ skip", style=RED)
        elif s.stratifiable:
            strat = Text("✓ auto", style=GREEN)
        else:
            strat = Text("—", style=DIM)
        # reduction only ever touches numeric columns, so the skip state is
        # only meaningful (and only shown) for them — a skip flag on a
        # non-numeric column has no effect and must not claim otherwise
        if s.kind != "numeric":
            reduce_cell = Text("—", style=DIM)
        elif cfg.skip_reduce:
            reduce_cell = Text("✗ skip", style=RED)
        else:
            reduce_cell = Text("◇ pca", style=CYAN)
        selected = name in self.selection
        marker = "▌ " if selected else "  "
        name_cell = Text(marker + name, style=f"bold {YELLOW if selected else FG}")
        # mean / median / sd are numeric-only; mode applies to every kind
        num = s.kind == "numeric"
        return [
            name_cell,
            Text(s.kind, style=color),
            Text(anon_label(cfg), style=GREEN if cfg.kind != "none" else DIM),
            strat,
            reduce_cell,
            Text(f"{s.missing_pct:.1f}", style=miss_style, justify="right"),
            Text(f"{s.unique:,}", style=FG, justify="right"),
            Text(sparkline(s.histogram), style=color),
            Text(_fmt_stat(s.mean) if num else "—", style=CYAN if num else DIM, justify="right"),
            Text(_fmt_stat(s.median) if num else "—", style=CYAN if num else DIM, justify="right"),
            Text(_fmt_stat(s.mode), style=FG if s.mode is not None else DIM, justify="right"),
            Text(_fmt_stat(s.std) if num else "—", style=CYAN if num else DIM, justify="right"),
        ]

    def _refresh_row(self, name: str) -> None:
        table = self.query_one("#columns-table", DataTable)
        for col_key, cell in zip(self._col_keys, self._row_cells(name)):
            table.update_cell(name, col_key, cell)

    def _refresh_all_rows(self) -> None:
        for name in self.configs:
            self._refresh_row(name)

    def _update_selection_status(self) -> None:
        """Reflect the multi-selection count in the columns-panel title."""
        n = len(self.selection)
        title = f"columns — {n} selected (edits apply to all)" if n else "columns"
        self.query_one("#columns-panel").border_title = title

    # ── detail panel ─────────────────────────────────────────────────────────

    def _show_detail(self, name: str) -> None:
        s = self.stats[name]
        color = KIND_COLORS.get(s.kind, DIM)
        t = Text()
        t.append(f"{s.name}\n", style=f"bold {color}")
        t.append(f"{s.kind} · {s.dtype}\n\n", style=DIM)
        t.append(f"{s.count:,}", style=FG)
        t.append(" non-null · ", style=DIM)
        t.append(f"{s.missing:,}", style=RED if s.missing else FG)
        t.append(f" missing ({s.missing_pct:.1f}%) · ", style=DIM)
        t.append(f"{s.unique:,}", style=FG)
        t.append(" unique\n", style=DIM)
        if s.kind == "numeric" and s.min is not None:
            from ..stats import _fmt_num

            t.append("\nmin ", style=DIM)
            t.append(_fmt_num(s.min), style=CYAN)
            t.append("  median ", style=DIM)
            t.append(_fmt_num(s.median), style=CYAN)
            t.append("  mean ", style=DIM)
            t.append(_fmt_num(s.mean), style=CYAN)
            t.append("  max ", style=DIM)
            t.append(_fmt_num(s.max), style=CYAN)
            t.append("  σ ", style=DIM)
            t.append(_fmt_num(s.std), style=CYAN)
            t.append("\n")
        if s.histogram:
            t.append("\n")
            peak = max(s.histogram) or 1
            total = sum(s.histogram) or 1
            for label, count in zip(s.histogram_labels, s.histogram):
                width = int(count / peak * 24)
                t.append(f"{label[:18]:>18} ", style=FG)
                t.append("█" * width + "▏" * (0 if width else 1), style=color)
                t.append(f" {count:,} ({count / total * 100:.1f}%)\n", style=DIM)
        if not s.stratifiable:
            t.append("\nnot a stratification candidate ", style=DIM)
            t.append("(too many unique values, long text, or constant)", style=DIM)
        self.query_one("#detail", Static).update(t)

    # ── config panel sync ────────────────────────────────────────────────────

    _OPT_IDS = {
        "opt-seq-start": ("sequential_id", "start"),
        "opt-seq-interval": ("sequential_id", "interval"),
        "opt-seq-prefix": ("sequential_id", "prefix"),
        "opt-seq-width": ("sequential_id", "width"),
        "opt-jit-pct": ("numeric_jitter", "pct"),
        "opt-jit-round": ("numeric_jitter", "round_to"),
        "opt-dt-delta": ("datetime_jitter", "max_delta"),
        "opt-dt-unit": ("datetime_jitter", "unit"),
        "opt-str-length": ("random_string", "length"),
        "opt-str-prefix": ("random_string", "prefix"),
        "opt-hex-length": ("hex", "length"),
    }

    _OPT_DEFAULTS = {
        "opt-seq-start": "1", "opt-seq-interval": "1", "opt-seq-prefix": "",
        "opt-seq-width": "0", "opt-jit-pct": "20", "opt-jit-round": "",
        "opt-dt-delta": "7D", "opt-dt-unit": "s",
        "opt-str-length": "8", "opt-str-prefix": "", "opt-hex-length": "8",
    }

    def _sync_config_panel(self, name: str) -> None:
        cfg = self.configs[name]
        self._syncing = True
        try:
            # only assign widgets whose value actually differs: assignments
            # queue Changed messages that are processed AFTER _syncing clears,
            # so gratuitous ones can clobber a config edit still in flight
            kind_select = self.query_one("#anon-kind", Select)
            if kind_select.value != cfg.kind:
                kind_select.value = cfg.kind
            self.query_one("#anon-options", ContentSwitcher).current = f"opts-{cfg.kind}"
            style_select = self.query_one("#opt-names-style", Select)
            style_value = cfg.options.get("style") or "first_last"
            if style_select.value != style_value:
                style_select.value = style_value
            for widget_id, (kind, opt) in self._OPT_IDS.items():
                value = str(
                    cfg.options.get(opt, self._OPT_DEFAULTS[widget_id])
                    if cfg.kind == kind
                    else self._OPT_DEFAULTS[widget_id]
                )
                widget = self.query_one(f"#{widget_id}", Input)
                if widget.value != value:
                    widget.value = value
            skip = self.query_one("#skip-strat", Switch)
            if skip.value != cfg.skip_strat:
                skip.value = cfg.skip_strat
            skip_reduce = self.query_one("#skip-reduce", Switch)
            if skip_reduce.value != cfg.skip_reduce:
                skip_reduce.value = cfg.skip_reduce
            # names gender / ethnicity controls
            gsel = self.query_one("#opt-names-gender", Select)
            if gsel.value != cfg.gender:
                gsel.value = cfg.gender
            esel = self.query_one("#opt-names-ethnicity", Select)
            if esel.value != cfg.ethnicity:
                esel.value = cfg.ethnicity
            gcol = self.query_one("#opt-names-gender-col", Select)
            if gcol.value != cfg.gender_column:
                gcol.value = cfg.gender_column  # "" == the "(no column)" option
            ecol = self.query_one("#opt-names-eth-col", Select)
            if ecol.value != cfg.ethnicity_column:
                ecol.value = cfg.ethnicity_column
            grand = self.query_one("#opt-names-gender-random", Switch)
            if grand.value != cfg.randomize_gender:
                grand.value = cfg.randomize_gender
            self._update_names_visibility(cfg)
        finally:
            self._syncing = False

    # ── events ───────────────────────────────────────────────────────────────

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        if event.row_key.value == self.selected:
            # duplicate highlight for the row already shown (e.g. the initial
            # mount-time highlight arriving late): re-syncing here would reset
            # panel widgets and queue stale Changed messages that clobber any
            # user edit still in flight — the panel is already correct
            return
        self.selected = event.row_key.value
        self._show_detail(self.selected)
        self._sync_config_panel(self.selected)

    @staticmethod
    def _is_stale(event_value, widget_value) -> bool:
        """Whether a Changed message no longer reflects its widget's value.

        Every widget runs its own message pump, so a Changed can be delivered
        AFTER the value it reports was superseded (mount-time echoes, rapid
        edits, panel syncs). Applying it would roll the config back — the
        exact class of race that reset user edits on slow CI machines. A
        message whose value differs from the widget's current value is
        provably stale and safe to drop: the message carrying the current
        value is either this one or still on its way.
        """
        return str(event_value) != str(widget_value)

    # ── multi-selection + undo/redo ───────────────────────────────────────────

    def _targets(self) -> list[str]:
        """Columns a config edit applies to, in df order.

        When a multi-selection exists it is the exact target set (so a row
        toggled *off* is never edited, even while it is the cursor row);
        otherwise it falls back to just the cursor row.
        """
        if self.selection:
            return [c for c in self.configs if c in self.selection]
        if self.selected is not None:
            return [self.selected]
        return []

    def _snapshot(self) -> dict[str, ColumnConfig]:
        return {c: copy.deepcopy(cfg) for c, cfg in self.configs.items()}

    def _checkpoint(self, coalesce_key: object | None = None) -> None:
        """Push the current config onto the undo stack before a mutation.

        ``coalesce_key`` folds a burst of edits to one field (e.g. typing in an
        option box) into a single undo step: consecutive checkpoints sharing a
        non-None key are skipped, so the first snapshot of the burst is the one
        undo restores. Any structural change passes ``None`` and always records.
        """
        if coalesce_key is not None and coalesce_key == self._last_coalesce_key:
            return
        self._undo.append(self._snapshot())
        self._redo.clear()
        self._last_coalesce_key = coalesce_key

    def _restore(self, configs: dict[str, ColumnConfig]) -> None:
        self.configs = configs
        self._last_coalesce_key = None
        self._refresh_all_rows()
        if self.selected is not None:
            self._sync_config_panel(self.selected)

    def action_undo(self) -> None:
        if not self._undo:
            self.notify("nothing to undo", timeout=2)
            return
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        self.notify("undo", timeout=1)

    def action_redo(self) -> None:
        if not self._redo:
            self.notify("nothing to redo", timeout=2)
            return
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        self.notify("redo", timeout=1)

    def _toggle_in_selection(self, name: str) -> None:
        if name in self.selection:
            self.selection.discard(name)
        else:
            self.selection.add(name)
        self._anchor = name

    def on_columns_table_modifier_click(
        self, event: ColumnsDataTable.ModifierClick
    ) -> None:
        name = event.row_key
        if name not in self.configs:
            return
        if event.shift and self._anchor in self.configs:
            names = list(self.configs)
            i, j = names.index(self._anchor), names.index(name)
            lo, hi = (i, j) if i <= j else (j, i)
            self.selection = set(names[lo : hi + 1])
        elif event.ctrl:
            self._toggle_in_selection(name)
        else:  # shift-click without an anchor: start a fresh selection here
            self.selection = {name}
            self._anchor = name
        self._refresh_all_rows()
        self._update_selection_status()

    def action_toggle_select(self) -> None:
        """Keyboard equivalent of ctrl-click: toggle the cursor row."""
        if self.selected is None:
            return
        self._toggle_in_selection(self.selected)
        self._refresh_all_rows()
        self._update_selection_status()

    def action_clear_select(self) -> None:
        if not self.selection:
            return
        self.selection.clear()
        self._anchor = None
        self._refresh_all_rows()
        self._update_selection_status()

    # ── config edits (apply to every target column) ───────────────────────────

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._syncing or self.selected is None:
            return
        if self._is_stale(event.value, event.select.value):
            return
        sel = self.configs[self.selected]
        if event.select.id == "anon-kind":
            new_kind = str(event.value)
            if new_kind == sel.kind:
                return  # spurious/echoed change; keep existing options
            self._checkpoint()
            for c in self._targets():
                cfg = self.configs[c]
                cfg.kind = new_kind
                cfg.options = {}
                self._refresh_row(c)
            self.query_one("#anon-options", ContentSwitcher).current = f"opts-{new_kind}"
        elif event.select.id == "opt-names-style":
            style = str(event.value)
            if sel.kind != "names" or sel.options.get("style", "first_last") == style:
                return
            targets = self._targets()
            self._checkpoint(coalesce_key=(tuple(targets), "opt:names-style"))
            for c in targets:
                if self.configs[c].kind == "names":
                    self.configs[c].options["style"] = style
                    self._refresh_row(c)
        elif event.select.id == "opt-names-gender":
            self._apply_names_attr("gender", str(event.value))
            self._update_names_visibility(sel)
        elif event.select.id == "opt-names-ethnicity":
            self._apply_names_attr("ethnicity", str(event.value))
            self._update_names_visibility(sel)
        elif event.select.id == "opt-names-gender-col":
            self._apply_names_column("gender", str(event.value))
        elif event.select.id == "opt-names-eth-col":
            self._apply_names_column("ethnicity", str(event.value))

    def _apply_names_attr(self, attr: str, value: str) -> None:
        """Set a names attribute (gender/ethnicity) on every names target."""
        if getattr(self.configs[self.selected], attr) == value:
            return
        self._checkpoint()
        for c in self._targets():
            if self.configs[c].kind == "names":
                setattr(self.configs[c], attr, value)

    def _apply_names_column(self, which: str, column: str) -> None:
        """Set (or clear, when ``column`` is "") the gender/ethnicity source
        column, auto-detecting its value mapping."""
        col_attr = "gender_column" if which == "gender" else "ethnicity_column"
        map_attr = "gender_map" if which == "gender" else "ethnicity_map"
        if getattr(self.configs[self.selected], col_attr) == column:
            return  # echo / no-op for the shown row
        auto: dict = {}
        if column and self.app.df is not None and column in self.app.df.columns:
            distinct = list(self.app.df[column].dropna().unique())
            suggest = suggest_gender_mapping if which == "gender" else suggest_ethnicity_mapping
            auto = {k: v for k, v in suggest(distinct).items() if v}
        self._checkpoint()
        for c in self._targets():
            if self.configs[c].kind == "names":
                setattr(self.configs[c], col_attr, column)
                setattr(self.configs[c], map_attr, dict(auto))

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "opt-names-gender-random":
            if self._syncing or self.selected is None:
                return
            if self._is_stale(event.value, event.switch.value):
                return
            if self.configs[self.selected].randomize_gender == event.value:
                return
            self._checkpoint()
            for c in self._targets():
                if self.configs[c].kind == "names":
                    self.configs[c].randomize_gender = event.value
            return
        if event.switch.id not in ("skip-strat", "skip-reduce"):
            return  # ignore the run bar's #random switch
        if self._syncing or self.selected is None:
            return
        if self._is_stale(event.value, event.switch.value):
            return
        attr = "skip_strat" if event.switch.id == "skip-strat" else "skip_reduce"
        if getattr(self.configs[self.selected], attr) == event.value:
            return  # echo / no-op for the shown row
        self._checkpoint()
        for c in self._targets():
            setattr(self.configs[c], attr, event.value)
            self._refresh_row(c)

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._syncing or self.selected is None:
            return
        if self._is_stale(event.value, event.input.value):
            return
        mapping = self._OPT_IDS.get(event.input.id or "")
        if mapping is None:
            return  # a run-bar input (count/seed/…), not an anonymizer option
        kind, opt = mapping
        sel = self.configs[self.selected]
        default = self._OPT_DEFAULTS[event.input.id]
        if sel.kind != kind or str(sel.options.get(opt, default)) == str(event.value):
            return  # panel not on this kind, or an idempotent sync echo
        targets = self._targets()
        # the coalesce key includes the target set, so a burst of edits to the
        # same field on the same columns folds into one undo step — but editing
        # the same field on a *different* column (after moving the cursor) does
        # not, and stays a separate, recoverable step
        self._checkpoint(coalesce_key=(tuple(targets), f"opt:{event.input.id}"))
        for c in targets:
            cfg = self.configs[c]
            if cfg.kind == kind:
                cfg.options[opt] = event.value
                self._refresh_row(c)

    def _update_names_visibility(self, cfg: ColumnConfig) -> None:
        """Show the gender/ethnicity column pickers only in 'from a column' mode."""
        self.query_one("#opt-names-gender-col-row").set_class(
            cfg.gender != "column", "hidden"
        )
        self.query_one("#opt-names-eth-col-row").set_class(
            cfg.ethnicity != "column", "hidden"
        )

    def _open_mapping(self, which: str) -> None:
        """Open the value→group mapping modal for the gender/ethnicity column."""
        app = self.app  # type: ignore[assignment]
        if self.selected is None or app.df is None:
            return
        cfg = self.configs[self.selected]
        col = cfg.gender_column if which == "gender" else cfg.ethnicity_column
        if not col or col not in app.df.columns:
            self.notify(f"pick a {which} column first", severity="warning")
            return
        distinct = list(app.df[col].dropna().unique())
        if which == "gender":
            targets = GENDER_MAP_TARGETS
            current = {**suggest_gender_mapping(distinct), **cfg.gender_map}
            attr = "gender_map"
        else:
            targets = ETHNICITY_MAP_TARGETS
            current = {**suggest_ethnicity_mapping(distinct), **cfg.ethnicity_map}
            attr = "ethnicity_map"

        def done(mapping: dict | None) -> None:
            if mapping is None:
                return
            self._checkpoint()
            for c in self._targets():
                if self.configs[c].kind == "names":
                    setattr(self.configs[c], attr, dict(mapping))
            self.notify(f"{which} mapping: {len(mapping)} value(s) set", timeout=3)

        app.push_screen(
            MappingScreen(f"map '{col}' → {which}", distinct, targets, current), done
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run":
            self.action_run()
        elif event.button.id == "btn-gender-map":
            self._open_mapping("gender")
        elif event.button.id == "btn-eth-map":
            self._open_mapping("ethnicity")

    def action_toggle_skip(self) -> None:
        self._bulk_toggle("skip_strat")

    def action_toggle_reduce_skip(self) -> None:
        self._bulk_toggle("skip_reduce")

    def _bulk_toggle(self, attr: str) -> None:
        """Flip a boolean config flag on every target column (keyed off the
        first target's current value so the whole set moves together)."""
        targets = self._targets()
        if not targets:
            return
        self._checkpoint()
        new_value = not getattr(self.configs[targets[0]], attr)
        for c in targets:
            setattr(self.configs[c], attr, new_value)
            self._refresh_row(c)
        if self.selected is not None:
            self._sync_config_panel(self.selected)

    def action_suggest(self) -> None:
        """Auto-assign a suggested anonymizer type to every column."""
        from ..workflow import suggest_type

        planned = {name: suggest_type(stats) for name, stats in self.stats.items()}
        to_change = [n for n, k in planned.items() if self.configs[n].kind != k]
        # only record an undo step when something actually changes, so pressing
        # 'a' on an already-suggested frame doesn't push a phantom step or wipe
        # the redo history
        if to_change:
            self._checkpoint()
        for name, kind in planned.items():
            cfg = self.configs[name]
            if cfg.kind != kind:
                cfg.kind = kind
                cfg.options = {}
            self._refresh_row(name)
        if self.selected is not None:
            self._sync_config_panel(self.selected)
        self.notify(
            f"suggested anonymizer types ({len(to_change)} column(s) changed)",
            timeout=3,
        )

    def action_back(self) -> None:
        self.app.pop_screen()

    # ── run ──────────────────────────────────────────────────────────────────

    def action_run(self) -> None:
        app: DataSamplerApp = self.app  # type: ignore[assignment]
        try:
            count = int(self.query_one("#count", Input).value)
            if count < 1:
                raise ValueError
        except ValueError:
            self.notify("sample count must be a positive integer", severity="error")
            return
        seed_raw = self.query_one("#seed", Input).value.strip()
        try:
            seed = int(seed_raw) if seed_raw else None
        except ValueError:
            self.notify("seed must be an integer (or blank)", severity="error")
            return
        outdir = self.query_one("#outdir", Input).value.strip() or None
        use_random = self.query_one("#random", Switch).value
        reduce_mode = self.query_one("#reduce-mode", Select).value
        reduce_value = self.query_one("#reduce-value", Input).value.strip()
        reduce_kwargs: dict | None = None
        if reduce_mode == "components":
            try:
                n = int(reduce_value)
                if n < 1:
                    raise ValueError
            except ValueError:
                self.notify(
                    "reduce: component count must be a positive integer",
                    severity="error",
                )
                return
            reduce_kwargs = {"n_components": n}
        elif reduce_mode == "variance":
            try:
                r = float(reduce_value)
                if not 0.0 < r < 1.0:
                    raise ValueError
            except ValueError:
                self.notify(
                    "reduce: variance ratio must be between 0 and 1 (e.g. 0.9)",
                    severity="error",
                )
                return
            reduce_kwargs = {"variance_ratio": r}
        exclude = [c for c, cfg in self.configs.items() if cfg.skip_strat]
        # only numeric columns are reduction candidates, so a skip flag on any
        # other kind is a no-op — keep it out of the exclude list and report
        reduce_exclude = [
            c for c, cfg in self.configs.items()
            if cfg.skip_reduce and self.stats[c].kind == "numeric"
        ]
        try:
            spec = {
                c: build_anonymizer(cfg)
                for c, cfg in self.configs.items()
                if cfg.kind != "none"
            }
        except (ValueError, TypeError) as exc:
            self.notify(f"anonymizer options: {exc}", severity="error")
            return

        self.query_one("#run", Button).disabled = True
        self.notify("sampling…", timeout=2)
        self.run_worker(
            lambda: self._do_run(
                count, use_random, exclude, spec, seed, outdir,
                reduce_kwargs, reduce_exclude,
            ),
            thread=True,
            exclusive=True,
        )

    def _build_snippet(
        self, *, count, use_random, exclude, seed, outdir,
        reduce_kwargs, reduce_exclude, source_path, tag,
    ) -> str:
        """The Python that reproduces this run — shown on the report screen so
        the TUI is a launchpad for the scriptable API, not a dead end."""
        src = Path(source_path).name if source_path else "data.csv"
        out: list[str] = ["import data_sampler as ds", "", f"df = ds.load_file({src!r})"]

        args = ["df", str(count)]
        if use_random:
            args.append("use_random=True")
        if exclude:
            args.append(f"exclude_columns={exclude!r}")
        if seed is not None:
            args.append(f"random_state={seed}")
        out.append(f"result = ds.sample({', '.join(args)})")
        out.append("data = result.data")

        anon = [(c, cfg) for c, cfg in self.configs.items() if cfg.kind != "none"]
        if anon:
            out.append("data = ds.anonymize(data, {")
            for col, cfg in anon:
                kind, kwargs = anon_api_kwargs(cfg)
                spec = f"({kind!r}, {kwargs!r})" if kwargs else f"{kind!r}"
                out.append(f"    {col!r}: {spec},")
            out.append(f"}}, seed={seed!r})")

        if reduce_kwargs:
            rargs = ", ".join(f"{k}={v!r}" for k, v in reduce_kwargs.items())
            if reduce_exclude:
                rargs += f", exclude={reduce_exclude!r}"
            if seed is not None:
                rargs += f", seed={seed}"
            out.append(f"red = ds.reduce_columns(data, {rargs})")
            out.append("data = red.data")

        outdir_arg = f", output_folder={outdir!r}" if outdir else ""
        out.append(f"ds.save_output(data, {src!r}, tag={tag!r}{outdir_arg})")
        return "\n".join(out)

    def _do_run(
        self, count, use_random, exclude, spec, seed, outdir,
        reduce_kwargs=None, reduce_exclude=(),
    ) -> None:
        app: DataSamplerApp = self.app  # type: ignore[assignment]
        try:
            from ..anonymize import anonymize
            from ..report import column_histogram_data

            result = sample(
                app.df, count,
                use_random=use_random,
                exclude_columns=exclude,
                random_state=seed,
            )
            # source-vs-sample histograms use the pre-anonymization sample
            hist_data = column_histogram_data(app.df, result.data)
            data = result.data
            if spec:
                data = anonymize(data, spec, seed=seed)
            reduction = None
            if reduce_kwargs:
                reduction = reduce_columns(
                    data, seed=seed, exclude=reduce_exclude, **reduce_kwargs
                )
                data = reduction.data
            tag = (
                f"sample_{count}"
                + ("_anon" if spec else "")
                + (
                    f"_pca{reduction.n_components}"
                    if reduction is not None and reduction.n_components
                    else ""
                )
            )
            out_path = save_output(data, app.source_path, tag, output_folder=outdir)

            # lead with the reproduce-in-Python snippet so it's the first thing
            # on the report (and at the top of the saved .txt), not below a long
            # stratification report
            snippet = self._build_snippet(
                count=count, use_random=use_random, exclude=exclude, seed=seed,
                outdir=outdir, reduce_kwargs=reduce_kwargs,
                reduce_exclude=reduce_exclude, source_path=app.source_path, tag=tag,
            )
            lines = [
                "╔═════════════════════════════════════════════════════════════════════╗",
                "║                       REPRODUCE THIS IN PYTHON                      ║",
                "╚═════════════════════════════════════════════════════════════════════╝",
                "",
                snippet,
                "",
            ]
            lines += list(result.notes)
            if exclude:
                lines.append(f"Columns excluded from stratification: {', '.join(exclude)}")
            report = format_stratification_report(app.df, result)
            if result.method == "stratified":
                lines.append("")
                lines.append(report)
            if spec:
                lines.append("")
                lines.append("ANONYMIZED COLUMNS")
                for col in spec:
                    lines.append(f"  {col}  →  {anon_label(self.configs[col])}")
            if reduction is not None:
                lines.append("")
                if reduce_exclude:
                    lines.append(
                        "Columns excluded from reduction: "
                        + ", ".join(reduce_exclude)
                    )
                lines.append(format_reduction_report(reduction))
            lines.append("")
            lines.append(f"Sampled {len(data)} rows.")
            lines.append(f"Output saved to: {out_path}")
            app.call_from_thread(
                self._run_done, "\n".join(lines), str(out_path), hist_data
            )
        except Exception as exc:  # surfaced to the user, never crash the TUI
            log.exception("run failed")
            app.call_from_thread(self._run_failed, exc)

    def _run_failed(self, exc: Exception) -> None:
        self.query_one("#run", Button).disabled = False
        self.notify(f"{type(exc).__name__}: {exc}", severity="error", timeout=10)

    def _run_done(self, report: str, out_path: str, hist_data: list) -> None:
        self.query_one("#run", Button).disabled = False
        self.app.push_screen(ReportScreen(report, out_path, hist_data))


class ReportScreen(Screen):
    """Post-run report: stratification comparison, per-column histograms,
    anonymization summary."""

    BINDINGS = [
        Binding("escape", "back", "back to columns"),
        Binding("ctrl+s", "save_text", "save report + histograms"),
        Binding("n", "new_file", "new file"),
    ]

    HIST_BAR = 12  # width of each histogram bar in the panel

    def __init__(self, report: str, out_path: str, hist_data: list | None = None):
        super().__init__()
        self._report = report
        self._out_path = out_path
        self._hist_data = hist_data or []

    def compose(self) -> ComposeResult:
        yield Static(" ▓▒░ DATA SAMPLER ░▒▓  sample complete", id="titlebar")
        with Horizontal(id="report-main"):
            with VerticalScroll(id="report-panel"):
                yield Static(Text(self._report), id="report-text")
            with VerticalScroll(id="hist-panel"):
                yield Static(self._build_histograms(), id="hist-text")
        with Horizontal(id="report-actions"):
            yield Button("◀ back (esc)", id="back")
            yield Button("💾 save report + histograms (ctrl+s)", id="save-text", variant="success")
            yield Button("new file (n)", id="new-file", variant="primary")
            yield Static("", id="save-status")
        yield Footer()

    def on_mount(self) -> None:
        report_panel = self.query_one("#report-panel")
        report_panel.border_title = "report"
        self.query_one("#hist-panel").border_title = "column histograms"
        report_panel.focus()

    def _build_histograms(self) -> Text:
        if not self._hist_data:
            return Text("no columns to chart", style=DIM)
        t = Text()
        t.append("source ", style=DIM)
        t.append("█", style=DIM)
        t.append("   sample ", style=DIM)
        t.append("█", style=CYAN)
        t.append("  (% of non-null values)\n", style=DIM)
        for d in self._hist_data:
            labels = d["labels"]
            if not labels:
                continue
            color = KIND_COLORS.get(d["kind"], DIM)
            label_w = min(16, max(len(l) for l in labels))
            peak = max([*d["source_pct"], *d["sample_pct"], 1e-9])
            t.append(f"\n{d['name']} ", style=f"bold {color}")
            t.append(f"({d['kind']})\n", style=DIM)
            for label, s_pct, m_pct in zip(labels, d["source_pct"], d["sample_pct"]):
                lbl = label if len(label) <= label_w else label[: label_w - 1] + "…"
                s_len = int(s_pct / peak * self.HIST_BAR)
                m_len = int(m_pct / peak * self.HIST_BAR)
                t.append(f"{lbl:>{label_w}} ", style=FG)
                t.append("█" * s_len + "░" * (self.HIST_BAR - s_len), style=DIM)
                t.append(f"{s_pct:4.0f}% ", style=DIM)
                t.append("█" * m_len + "░" * (self.HIST_BAR - m_len), style=color)
                t.append(f"{m_pct:4.0f}%\n", style=DIM)
        return t

    def _histograms_text(self) -> str:
        """Plain-text (no color markup) render of the column histograms, for
        the saved report file."""
        if not self._hist_data:
            return "no columns to chart"
        bar = self.HIST_BAR
        lines = ["COLUMN DISTRIBUTIONS (source vs sample, % of non-null values)"]
        for d in self._hist_data:
            labels = d["labels"]
            if not labels:
                continue
            label_w = min(16, max(len(l) for l in labels))
            peak = max([*d["source_pct"], *d["sample_pct"], 1e-9])
            lines.append(f"\n{d['name']} ({d['kind']})")
            for label, s_pct, m_pct in zip(labels, d["source_pct"], d["sample_pct"]):
                lbl = label if len(label) <= label_w else label[: label_w - 1] + "…"
                s_len = int(s_pct / peak * bar)
                m_len = int(m_pct / peak * bar)
                lines.append(
                    f"  {lbl:>{label_w}}  src {'█' * s_len}{'░' * (bar - s_len)} "
                    f"{s_pct:5.1f}%   sam {'█' * m_len}{'░' * (bar - m_len)} {m_pct:5.1f}%"
                )
        return "\n".join(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "save-text":
            self.action_save_text()
        elif event.button.id == "new-file":
            self.action_new_file()

    def action_save_text(self) -> None:
        """Write the report AND the column histograms to a .txt beside the sample."""
        try:
            out = Path(self._out_path)
            txt_path = out.with_name(f"{out.stem}_report.txt")
            content = (
                self._report
                + "\n\n"
                + self._histograms_text()  # the "COLUMN DISTRIBUTIONS" section
                + "\n"
            )
            txt_path.write_text(content, encoding="utf-8")
        except Exception as exc:  # surfaced to the user, never crash the TUI
            log.exception("save report failed")
            self.query_one("#save-status", Static).update(
                Text(f"✗ save failed: {exc}", style=RED)
            )
            self.notify(f"save failed: {exc}", severity="error", timeout=8)
            return
        self.query_one("#save-status", Static).update(
            Text(f"✓ saved report + histograms → {txt_path.name}", style=GREEN)
        )
        self.notify(f"saved report + histograms to {txt_path}", timeout=5)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_new_file(self) -> None:
        app: DataSamplerApp = self.app  # type: ignore[assignment]
        app.pop_screen()
        app.pop_screen()


# ── app ───────────────────────────────────────────────────────────────────────


class DataSamplerApp(App):
    """Colorful terminal UI for sampling + anonymizing tabular data."""

    TITLE = "Data Sampler"

    BINDINGS = [Binding("ctrl+q", "quit", "quit")]

    CSS = f"""
    Screen {{
        background: #0b0e14;
        color: {FG};
    }}
    #titlebar {{
        height: 1;
        background: #10141c;
        color: {CYAN};
        text-style: bold;
    }}
    Footer {{
        background: #10141c;
    }}

    /* file screen */
    #file-main {{ height: 1fr; }}
    #file-form {{
        width: 50%;
        border: round {CYAN};
        border-title-color: {CYAN};
        padding: 1 2;
    }}
    #file-browser {{
        width: 50%;
        border: round {MAGENTA};
        border-title-color: {MAGENTA};
    }}
    #browser {{ background: transparent; height: 1fr; }}
    #refresh-browser {{ width: auto; height: 3; margin: 0 1; }}
    #file-status {{ margin-top: 1; }}
    #load {{ margin-top: 1; }}
    #names-lib-note {{ color: {DIM}; margin-top: 1; }}
    #names-lib-row {{ height: 3; margin-top: 1; }}
    #names-lib-row Button {{ margin-right: 1; }}
    #names-lib-status {{ margin-top: 1; }}
    .field-label {{ color: {DIM}; margin-top: 1; }}

    /* columns screen */
    #main {{ height: 1fr; }}
    #columns-panel {{
        width: 58%;
        border: round {CYAN};
        border-title-color: {CYAN};
    }}
    #columns-table {{
        background: transparent;
        height: 1fr;
    }}
    #side {{ width: 42%; }}
    #detail-panel {{
        height: 45%;
        border: round {MAGENTA};
        border-title-color: {MAGENTA};
        padding: 0 1;
    }}
    #config-panel {{
        height: 55%;
        border: round {YELLOW};
        border-title-color: {YELLOW};
        padding: 0 1;
    }}
    /* Vertical/Horizontal default to 1fr height; inside the scrollable
       config panel that swallows the space and pushes #skip-row out of
       view — force content-sized heights */
    #anon-options {{ height: auto; }}
    #anon-options > Vertical {{ height: auto; }}
    .opts-note {{ color: {DIM}; margin: 1 0; }}
    .optrow {{ height: 3; }}
    .opt-label {{ width: 10; margin-top: 1; color: {DIM}; }}
    .opt-input {{ width: 1fr; }}
    #skip-row {{ height: 3; margin-top: 1; }}
    #skip-label {{ margin-top: 1; color: {FG}; }}
    #reduce-skip-row {{ height: 3; }}
    #reduce-skip-label {{ margin-top: 1; color: {FG}; }}
    .hidden {{ display: none; }}
    .col-map {{ height: auto; padding: 0 0 0 2; }}
    .col-map > Button {{ margin: 0; }}
    .opt-label-wide {{ width: auto; margin-top: 1; margin-left: 1; color: {FG}; }}
    .mini-btn {{ width: auto; height: 3; }}

    /* run bar */
    #runbar {{
        height: 5;
        border: round {GREEN};
        border-title-color: {GREEN};
        padding: 0 1;
    }}
    .run-label {{ margin-top: 1; margin-right: 1; color: {DIM}; }}
    .run-input-s {{ width: 12; }}
    .run-input-l {{ width: 1fr; }}
    #random {{ margin-top: 0; }}
    #reduce-mode {{ width: 20; }}
    #reduce-value {{ width: 10; }}
    #run {{ margin-left: 2; }}

    /* report screen */
    #report-main {{ height: 1fr; }}
    #report-panel {{
        width: 55%;
        border: round {GREEN};
        border-title-color: {GREEN};
        padding: 0 2;
    }}
    #hist-panel {{
        width: 45%;
        border: round {CYAN};
        border-title-color: {CYAN};
        padding: 0 1;
    }}
    #report-actions {{ height: 3; padding: 0 2; }}
    #report-actions Button {{ margin-right: 2; }}
    #save-status {{ margin-top: 1; width: 1fr; }}

    /* value-mapping modal */
    MappingScreen {{ align: center middle; background: #0b0e14 70%; }}
    #map-dialog {{
        width: 64; height: 80%;
        background: #10141c;
        border: round {MAGENTA};
        border-title-color: {MAGENTA};
        padding: 1 2;
    }}
    #map-title {{ color: {FG}; text-style: bold; }}
    #map-help {{ color: {DIM}; margin-bottom: 1; }}
    #map-list {{ height: 1fr; }}
    .map-row {{ height: 3; }}
    .map-val {{ width: 24; margin-top: 1; color: {FG}; }}
    #map-actions {{ height: 3; margin-top: 1; }}
    #map-actions Button {{ margin-right: 2; }}

    /* inputs / selects / switches — rounded, quiet at rest, cyan on focus
       (btop-style chrome that matches the panels' round borders) */
    Input {{
        background: {INPUT_BG};
        border: round {BORDER};
        padding: 0 1;
        height: 3;
    }}
    Input:focus {{
        border: round {CYAN};
        background: {INPUT_BG_FOCUS};
    }}
    Input.-invalid {{ border: round {RED}; }}
    Input.-invalid:focus {{ border: round {RED}; }}
    Input > .input--placeholder {{ color: {DIM}; text-style: italic; }}
    Input > .input--cursor {{ background: {CYAN}; color: #0b0e14; }}

    Select {{ height: 3; }}
    SelectCurrent {{
        background: {INPUT_BG};
        border: round {BORDER};
        padding: 0 1;
    }}
    Select:focus > SelectCurrent {{ border: round {CYAN}; }}
    Select.-expanded > SelectCurrent {{ border: round {CYAN}; }}
    SelectCurrent .arrow {{ color: {CYAN}; }}
    SelectOverlay {{
        background: {INPUT_BG};
        border: round {CYAN};
    }}
    SelectOverlay > .option-list--option-highlighted {{
        background: {INPUT_BG_FOCUS};
        color: {CYAN};
        text-style: bold;
    }}

    Switch {{
        background: {INPUT_BG};
        border: round {BORDER};
        padding: 0 1;
    }}
    Switch:focus {{ border: round {CYAN}; }}
    Switch > .switch--slider {{ color: {DIM}; background: #0b0e14; }}
    Switch.-on > .switch--slider {{ color: {GREEN}; }}

    /* buttons — outlined, no solid fill (btop-style): a rounded accent border
       + bold accent text over the transparent panel, brightening on hover */
    Button {{
        height: 3;
        min-width: 10;
        background: transparent;
        color: {FG};
        text-style: bold;
        border: round {DIM};
        border-top: round {DIM};
        border-bottom: round {DIM};
    }}
    Button:hover {{ background: transparent; color: {CYAN}; border: round {CYAN}; }}
    Button:focus {{ background: transparent; border: round {CYAN}; text-style: bold; }}
    Button.-active {{ background: transparent; tint: {CYAN} 12%; }}
    Button.-success {{ color: {GREEN}; border: round {GREEN}; background: transparent; }}
    Button.-success:hover {{ color: {GREEN}; border: round {GREEN}; background: transparent; }}
    Button.-primary {{ color: {CYAN}; border: round {CYAN}; background: transparent; }}
    Button.-primary:hover {{ color: {CYAN}; border: round {CYAN}; background: transparent; }}

    DataTable > .datatable--header {{
        color: {CYAN};
        text-style: bold;
        background: #10141c;
    }}
    """

    def __init__(self, path: str | None = None, sheet: str | None = None):
        super().__init__()
        self._initial_path = path
        self._initial_sheet = sheet
        self.df: pd.DataFrame | None = None
        self.source_path: str | None = None
        self.sheet: str | None = None
        self.column_stats: list | None = None  # precomputed by the load worker

    def on_mount(self) -> None:
        # keep log lines off the live display
        log_path = Path(tempfile.gettempdir()) / "data_sampler_tui.log"
        redirect_to_file(str(log_path))
        self.push_screen(FileScreen(self._initial_path, self._initial_sheet))
