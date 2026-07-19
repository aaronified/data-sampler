# Roadmap

Feature blocks for the v3.0 rewrite (package + TUI + anonymizers). Each block
gets its own commit after tests pass and docs are updated.

## v3.0

- [x] **Block 1 — core package.** Restructure into a src-layout package
  (`src/data_sampler/`): file I/O, Data Wrangler-style column stats,
  stratified sampling engine (now with user-excluded columns, seedable RNG,
  and a `SampleResult` object), string-based reports, central logging, and
  packaging (`pyproject.toml`, hatchling).
- [x] **Block 2 — anonymizers.** Optional per-column anonymization with
  consistent value mapping: `names` (bundled first/middle/last library),
  `sequential_id` (start + interval), `numeric_jitter` (±20% by default),
  `random_string` (alphanumeric or hex). Public `anonymize()` +
  `make_anonymizer()` API.
- [x] **Block 3 — terminal UI.** Colorful, panel-based Textual TUI
  (btop/lazydocker style): file picker, column stats dashboard, per-column
  anonymizer config, stratification skip toggles, run + report screens.
  `run_tui()` entry point.
- [x] **Block 4 — CLI + release readiness.** `data-sampler` console script
  (no args → TUI; args → headless sampling with `--skip`/`--anon`
  options), `python -m data_sampler`, README/CHANGELOG rewrite, wheel build
  verification. PyPI upload stays manual, after extensive testing.

## Later

- Optional PyInstaller EXE build of the TUI (replaces the old Tkinter EXE).
- Datetime jitter anonymizer.
- Column-level histograms in the report screen.
