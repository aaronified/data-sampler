# Changelog

## v3.2 — unreleased

- **Vectorized every anonymizer** (Block P1 of the v3.2 performance & scale
  effort): the transform pipeline now does a single `pd.factorize`
  (dictionary-encode) plus a vectorized gather instead of `pd.unique` + a
  per-unique Python dict + `Series.map` (the in-process equivalent of a native
  join against a mapping table). Sequential IDs use `np.arange`;
  numeric/datetime jitter use vectorized numpy draws. Benchmarks: ~4–6× faster
  on `sequential_id` and up to ~3.3× on `numeric_jitter`, with `sequential_id`
  output verified bit-identical to the old path. The public `build_mapping`
  API, consistent-mapping guarantee, seed reproducibility, and NaN/dtype
  handling are preserved; nullable string dtypes now round-trip. Docstrings
  clarified that relabelling anonymizers (`names`, `sequential_id`,
  `random_string`, `hex`) are bijective while jitter anonymizers
  (`numeric_jitter`, `datetime_jitter`) are bounded noise (distinct nearby
  values may collide).
- **Added an optional DuckDB out-of-core engine** (`data_sampler.engine`,
  Blocks P2–P4 of the v3.2 performance & scale effort): install
  `pip install "data-sampler[large]"` to push loading, stratification, and
  sampling into DuckDB instead of pandas. Multi-threaded (`PRAGMA threads`)
  and memory-limited with spill-to-disk; reads CSV/TSV/JSON and Parquet
  natively (Parquet with projection pushdown, the biggest I/O win) plus
  pandas DataFrames — only the resulting sample is ever materialized.
  Sampling: reservoir sampling for the random case (exact count, single
  pass, reproducible via `REPEATABLE`) and two-pass proportional stratified
  sampling, with auto-stratification picking low-cardinality columns via
  HyperLogLog (`approx_count_distinct`). Seed-reproducible end to end
  (seeded stratified runs go single-threaded, since DuckDB's `random()`
  ordering is only reproducible that way); NaN strata are joined with
  `IS NOT DISTINCT FROM` so they aren't silently dropped; column identifiers
  are quoted against injection. New `should_use_engine` auto-selects the
  engine for Parquet/large inputs, and `large_materialization_warning` warns
  before loading a large dataset fully into pandas — the pandas path stays
  the default for small/medium data. Adversarially verified across 4 review
  lenses with zero findings; a 2M-row Parquet file sampled to 1000 rows in
  ~1s, out-of-core.
- **Added approximate stats at scale** (`DuckDBEngine.stats()` and
  module-level `engine.stats()`, Block P5 of the v3.2 performance & scale
  effort): per-column `ColumnStats` computed in DuckDB — distinct counts via
  HyperLogLog (`approx_count_distinct`), median via `approx_quantile`, plus
  min/max/mean/std, missing counts, equal-width numeric histograms, and
  categorical/datetime top-values. Scalar aggregates run in one streaming
  pass so stats stay cheap over billions of rows; `approximate=False` gives
  exact counts/quantiles for small inputs, and `distributions=False` skips
  the per-column passes for a single cheap scalar pass across very wide
  inputs. A new `ColumnStats.approximate` flag marks approximate results.

## v3.1 — unreleased

- **Added a `datetime_jitter` anonymizer** (`DatetimeJitterAnonymizer`):
  shifts each date/time by a random offset within a ±window (±7 days by
  default), with consistent value mapping preserved so the column's
  distribution shape survives. `NaT` is left untouched, string-date columns
  are coerced via `pandas.to_datetime`, and timezone-aware inputs keep their
  zone. Raises `ValueError` if the window is finer than its `unit`. Reachable
  as kinds `datetime_jitter` / `datetime` / `dates`, and wired into the CLI
  `--anon` option and the TUI anonymizer config panel.
- **Added a guided anonymization workflow** (`data_sampler.workflow`): an
  `AnonymizationPlan` maps each column to a "type" (anonymizer), buildable
  three ways — programmatically (`assign`/`suggest`), interactively via a
  menu wizard (`choose_interactively`, CLI `-i`/`--interactive`), or by
  clicking in the TUI (new `a` auto-suggest action). `suggest_type` infers a
  type per column from its stats, including date-named string columns →
  `datetime_jitter`, using token-based matching so "candidate"/"mandate"
  aren't misread as dates. New public API: `AnonymizationPlan`,
  `suggest_type`, `TYPE_OPTIONS`. CLI gains `--suggest`. The TUI columns
  table was reordered so the anonymizer/strat columns stay visible.
- **Added column-level histograms** (`column_histogram_data`,
  `format_column_histograms` in `report`): per-column source-vs-sample
  distributions — numeric columns share bin edges, other columns use the
  source's top categories — computed from the pre-anonymization sample so
  they show how faithfully the sample preserved each column. Surfaced as a
  new right-hand "column histograms" panel on the TUI report screen and a
  "COLUMN DISTRIBUTIONS" section in the CLI output. New public API:
  `column_histogram_data`, `format_column_histograms`.

## v3.0.1 — 2026-07-19

- **Fixed the bundled name library.** The v3.0.0 library contained invented
  non-name strings (e.g. "Argon", "Ardwin"), so the `names` anonymizer could
  produce gibberish. Replaced with a curated list of 200 first, 60 middle,
  and 220 last real names, globally diverse.
- **Added a bundled example dataset** (`examples/employees.csv`, 1,000 rows,
  stratifiable) and a worked example section in the README covering the
  TUI, the Python API, and the CLI.

## v3.0 — 2026-07-19

### Package restructure

- **Proper Python package.** The flat scripts are gone; everything now lives
  in a src-layout package (`src/data_sampler/`) built with hatchling and
  installable via `pip install -e .` (wheel-ready for a future PyPI
  release — the release itself stays manual, after extensive testing).
- **Public API.** `load_file`, `list_sheets`, `save_output`,
  `compute_stats`/`ColumnStats`, `sample`/`SampleResult`,
  `stratified_sample`, `find_stratification_columns`,
  `format_stratification_report`, `anonymize`, `make_anonymizer`, and
  `run_tui` are importable from `data_sampler`.
- **Old Tkinter GUI removed** (`data-sampler-gui.py`), replaced by the TUI.
- **Central logging** behind `DATA_SAMPLER_LOG` / `DATA_SAMPLER_LOG_FILE`.

### Terminal UI (new)

- Colorful, panel-based Textual dashboard (btop / lazydocker style):
  file picker with directory browser, Data Wrangler-style column stats
  table (type, missing %, unique count, distribution sparkline, summary),
  per-column detail panel with distribution bars, anonymizer configuration,
  stratification skip toggles, and a post-run report screen.
- Launch via `data-sampler` (no args), `data-sampler-tui`,
  `python -m data_sampler`, or `data_sampler.run_tui()`.

### Anonymizers (new)

- Optional per-column anonymization with **consistent value mapping**
  (repeated values stay repeated, so distributions survive) and NaN
  passthrough: `names` (bundled first/middle/last name library, five
  styles), `sequential_id` (start + interval, optional prefix/zero-pad),
  `numeric_jitter` (±20 % by default), `random_string` / `hex`
  (configurable length/charset). Seedable for reproducible runs.

### Sampling engine

- `sample()` accepts `exclude_columns` (columns the user marks as skipped
  never become stratifiers) and `random_state` for reproducible sampling.
- Sampling functions no longer print; they return a `SampleResult` that the
  CLI/TUI render via `format_stratification_report`.
- Fixed string-column classification under pandas 3.0 (new default
  `StringDtype` no longer matched the `object` dtype checks).

### CLI

- `data-sampler <source> <count>` now supports `--seed`,
  `--skip COL[,COL]`, and repeatable `--anon "COL=KIND[:k=v,...]"`
  anonymization options; with no arguments it opens the TUI.

## v2.0 — 2026-04-13

### Sampling engine

- **Joint intersection stratification.** Stratification now groups rows by the *simultaneous* combination of all selected columns (e.g. `A=x AND B=y AND C=z`), rather than applying stratifiers sequentially. Proportional allocation is computed once per intersection group, ensuring the sample reflects the true joint distribution.

- **Missing values included as a category.** Rows with `NaN` in any stratifier column are no longer silently dropped. They are treated as their own `(missing)` category and sampled proportionally alongside non-null values.

- **NaN-safe indexing throughout.** All allocation and sampling operations now use positional (integer) indexing to avoid `KeyError` failures when a MultiIndex contains `NaN` keys — a limitation of pandas label-based `.at[]` / `.get_loc()` accessors.

### Stratification report

- **Method statement.** The report now opens with a plain-language explanation of how groups were formed, so it is clear whether allocation was joint or single-column.

- **Fixed 13-character value labels.** All category labels in the report are right-justified to exactly 13 characters. Values longer than 13 characters are truncated to `10 chars + "..."`. Missing values display as `(missing)`.

- **Zero-allocation flags.** Any category that received 0 samples is marked inline with `← not represented`.

- **Per-category diagnosis.** When categories are unrepresented, a warning block appears showing — for each such value — the number of intersection groups it appeared in, the size of the largest group, and the minimum group size required to earn at least 1 sample. This confirms whether the cause is data sparsity or cross-column splitting.

### GUI

- **"Open output folder" button.** Opens the output directory in Windows Explorer after a successful run. Disabled until a run completes.

- **"Copy log" button.** Copies the full contents of the log panel to the clipboard in one click.

- **GUI no longer duplicates sampling logic.** The GUI now loads `data-sampler.py` directly at runtime via `importlib`, so both the CLI and the GUI always execute identical code. This applies to the built EXE as well — `data-sampler.py` is bundled alongside the executable and loaded from there.

---

## v1.0 — 2024

- Initial release with GUI front-end and CLI.
- Stratified sampling with automatic column selection.
- Supports CSV, TSV, JSON, Excel, and Parquet.
- Side-by-side distribution bar charts in the stratification report.
- Standalone Windows EXE built with PyInstaller.
