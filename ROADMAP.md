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

## v3.1

- [x] **Block A — datetime jitter anonymizer.** `datetime_jitter` kind
  (`DatetimeJitterAnonymizer`): shifts each date/time by a random offset
  within a ±window (±7 days by default), consistent-mapping preserved, NaT
  untouched, string-date columns coerced, timezones kept. Wired into the CLI
  (`--anon "col=datetime_jitter:max_delta=30D"`) and the TUI config panel.
- [x] **Block B — anonymiser workflow.** A guided column-type workflow
  (`data_sampler.workflow`): name a set of columns, then assign each a type by
  choosing from options (interactive `choose_interactively` wizard, CLI
  `--interactive`), pre-specifying through a function (`AnonymizationPlan`
  API + `suggest`/`assign`, CLI `--suggest`), or clicking in the TUI (with an
  `a` auto-suggest action). `suggest_type` infers a type per column from its
  stats. Types drive the anonymizers via `AnonymizationPlan.apply`.
- [ ] **Block C — column-level histograms in the report screen.** Per-column
  source-vs-sample distribution histograms on the report screen (and CLI
  text output).

## Later

- Optional PyInstaller EXE build of the TUI (replaces the old Tkinter EXE).
