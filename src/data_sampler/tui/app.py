"""The data-sampler terminal UI.

A panel-based, color-coded dashboard (in the spirit of btop / lazydocker):

- File screen — pick a data file (typed path or directory browser).
- Columns screen — Data Wrangler-style column stats table, a detail panel
  with distribution bars, per-column anonymizer configuration, and
  stratification skip toggles.
- Report screen — stratification report, anonymization summary, output path.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
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

from .. import __version__
from .._logging import get_logger, redirect_to_file
from ..anonymize import make_anonymizer
from ..io import load_file, save_output
from ..report import format_stratification_report
from ..sampling import sample
from ..stats import ColumnStats, compute_stats, sparkline

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


@dataclass
class ColumnConfig:
    """Per-column TUI state: chosen anonymizer + stratification skip."""

    kind: str = "none"
    options: dict[str, str] = field(default_factory=dict)
    skip_strat: bool = False


def build_anonymizer(cfg: ColumnConfig):
    """Turn a ColumnConfig's raw option strings into an anonymizer instance."""
    o = cfg.options
    if cfg.kind == "names":
        return make_anonymizer("names", style=o.get("style") or "first_last")
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


# ── screens ───────────────────────────────────────────────────────────────────


class FileScreen(Screen):
    """Pick the source data file."""

    BINDINGS = [Binding("ctrl+l", "load", "load file")]

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
            with Vertical(id="file-form"):
                yield Label("file path", classes="field-label")
                yield Input(
                    placeholder="path to .csv / .tsv / .json / .xlsx / .parquet",
                    id="path",
                )
                yield Label("excel sheet (blank = first)", classes="field-label")
                yield Input(placeholder="sheet name", id="sheet")
                yield Button("▶ load", id="load", variant="success")
                yield Static("", id="file-status")
            with Vertical(id="file-browser"):
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

    def action_load(self) -> None:
        path = self.query_one("#path", Input).value.strip().strip('"').strip("'")
        sheet = self.query_one("#sheet", Input).value.strip() or None
        status = self.query_one("#file-status", Static)
        if not path:
            status.update(Text("enter a file path or pick one from the browser", style=YELLOW))
            return
        if not os.path.isfile(path):
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


class ColumnsScreen(Screen):
    """Column stats dashboard + anonymizer / stratification configuration."""

    BINDINGS = [
        Binding("ctrl+r", "run", "run sample"),
        Binding("a", "suggest", "auto-suggest types"),
        Binding("s", "toggle_skip", "toggle strat skip"),
        Binding("escape", "back", "back to file"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.configs: dict[str, ColumnConfig] = {}
        self.stats: dict[str, ColumnStats] = {}
        self.selected: str | None = None
        self._syncing = False
        self._col_keys: list = []

    # ── layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        app: DataSamplerApp = self.app  # type: ignore[assignment]
        n_rows = len(app.df) if app.df is not None else 0
        n_cols = len(app.df.columns) if app.df is not None else 0
        yield Static(
            f" ▓▒░ DATA SAMPLER ░▒▓  {Path(app.source_path or '?').name}"
            f" · {n_rows:,} rows × {n_cols} columns",
            id="titlebar",
        )
        with Horizontal(id="main"):
            with Container(id="columns-panel"):
                yield DataTable(id="columns-table")
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
        with Horizontal(id="runbar"):
            yield Label("rows", classes="run-label")
            yield Input("100", id="count", classes="run-input-s")
            yield Label("out dir", classes="run-label")
            yield Input("", placeholder="same folder as source", id="outdir", classes="run-input-l")
            yield Label("seed", classes="run-label")
            yield Input("", placeholder="—", id="seed", classes="run-input-s")
            yield Label("random", classes="run-label")
            yield Switch(value=False, id="random")
            yield Button("▶ run sample", id="run", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        app: DataSamplerApp = self.app  # type: ignore[assignment]
        self.query_one("#columns-panel").border_title = "columns"
        self.query_one("#detail-panel").border_title = "detail"
        self.query_one("#config-panel").border_title = "anonymize · stratify"
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
        # actionable columns (anonymizer, strat) come right after type so they
        # stay visible; the wider distribution/summary columns trail them
        self._col_keys = list(
            table.add_columns(
                "column", "type", "anonymizer", "strat",
                "miss%", "uniq", "distribution", "summary",
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
        return [
            Text(name, style=f"bold {FG}"),
            Text(s.kind, style=color),
            Text(anon_label(cfg), style=GREEN if cfg.kind != "none" else DIM),
            strat,
            Text(f"{s.missing_pct:.1f}", style=miss_style, justify="right"),
            Text(f"{s.unique:,}", style=FG, justify="right"),
            Text(sparkline(s.histogram), style=color),
            Text(s.summary(), style=DIM),
        ]

    def _refresh_row(self, name: str) -> None:
        table = self.query_one("#columns-table", DataTable)
        for col_key, cell in zip(self._col_keys, self._row_cells(name)):
            table.update_cell(name, col_key, cell)

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

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._syncing or self.selected is None:
            return
        cfg = self.configs[self.selected]
        if event.select.id == "anon-kind":
            if str(event.value) == cfg.kind:
                return  # spurious/echoed change; keep existing options
            cfg.kind = str(event.value)
            cfg.options = {}
            self.query_one("#anon-options", ContentSwitcher).current = f"opts-{cfg.kind}"
            self._refresh_row(self.selected)
        elif event.select.id == "opt-names-style":
            cfg.options["style"] = str(event.value)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "skip-strat":
            if self._syncing or self.selected is None:
                return
            self.configs[self.selected].skip_strat = event.value
            self._refresh_row(self.selected)

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._syncing or self.selected is None:
            return
        mapping = self._OPT_IDS.get(event.input.id or "")
        if mapping is None:
            return
        kind, opt = mapping
        cfg = self.configs[self.selected]
        if cfg.kind == kind:
            cfg.options[opt] = event.value
            self._refresh_row(self.selected)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run":
            self.action_run()

    def action_toggle_skip(self) -> None:
        if self.selected is None:
            return
        cfg = self.configs[self.selected]
        cfg.skip_strat = not cfg.skip_strat
        self._sync_config_panel(self.selected)
        self._refresh_row(self.selected)

    def action_suggest(self) -> None:
        """Auto-assign a suggested anonymizer type to every column."""
        from ..workflow import suggest_type

        changed = 0
        for name, stats in self.stats.items():
            kind = suggest_type(stats)
            cfg = self.configs[name]
            if cfg.kind != kind:
                cfg.kind = kind
                cfg.options = {}
                changed += 1
            self._refresh_row(name)
        if self.selected is not None:
            self._sync_config_panel(self.selected)
        self.notify(
            f"suggested anonymizer types ({changed} column(s) changed)",
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
        exclude = [c for c, cfg in self.configs.items() if cfg.skip_strat]
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
            lambda: self._do_run(count, use_random, exclude, spec, seed, outdir),
            thread=True,
            exclusive=True,
        )

    def _do_run(self, count, use_random, exclude, spec, seed, outdir) -> None:
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
            tag = f"sample_{count}" + ("_anon" if spec else "")
            out_path = save_output(data, app.source_path, tag, output_folder=outdir)

            lines = list(result.notes)
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
            yield Button("new file (n)", id="new-file", variant="primary")
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "new-file":
            self.action_new_file()

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
    #browser {{ background: transparent; }}
    #file-status {{ margin-top: 1; }}
    #load {{ margin-top: 1; }}
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

    Input {{
        background: #151a24;
    }}
    Input:focus {{
        border: tall {CYAN};
    }}
    Select {{
        background: #151a24;
    }}
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
