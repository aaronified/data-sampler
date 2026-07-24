# Changelog

## v3.5.1 — 2026-07-24

- **The "skip from reduction (PCA)" toggle is hidden for non-numeric columns.**
  PCA reduction only ever operates on continuous numeric columns (booleans and
  datetimes are explicitly rejected; strings/categoricals never qualify), so the
  toggle is meaningless on those columns. It now appears in the detail panel only
  when the selected column is numeric — matching the columns table, which already
  renders `—` in the reduce cell for non-numeric columns. No behavior change to
  the reduction itself: skipped columns were, and remain, held out of the PCA
  math entirely and passed through to the output unchanged.

## v3.5.0 — 2026-07-24

- **Names anonymizer is now gender- and ethnicity-aware.** The bundled library
  (`_names.py`) is regrouped into real, hand-checked names keyed by
  `<ethnicity>_<gender>` across **33 ethnic groups** (multiple Indian
  regions/cultures, Chinese, Japanese, Korean, Vietnamese, Anglo, Italian,
  French, German, Russian, Polish, Scandinavian, Greek, Turkish, Persian,
  Arab-Levantine, Sudanese, Algerian, Moroccan, Ethiopian, Rwandan, Yoruba,
  Igbo, Ghanaian-Akan, Hispanic, Brazilian, Filipino, Indonesian). Surnames are
  grouped too — gendered where real (Russian Ivanov/Ivanova, Polish -ski/-ska)
  and with distinctly-female additions (North-Indian Devi/Kumari) over unisex
  bases (Kumar/Das). `NameAnonymizer(gender=…, ethnicity=…)` fixes a gender
  (`male`/`female`/`third`/`undisclosed`) and/or an ethnic group; **third
  gender** mixes given names within an ethnicity, **undisclosed** draws from any
  ethnicity.
- **Gender/ethnicity can be read from another column** and mapped:
  `NameAnonymizer(gender_column=…, ethnicity_column=…)` with per-value
  `gender_map` / `ethnicity_map` overrides. Values in any encoding (M/F, 0/1 per
  ISO-5218, `male`/`female`, other languages) are auto-detected
  (`suggest_gender_mapping` / `suggest_ethnicity_mapping`) with a manual
  fallback. `randomize_gender=True` reassigns genders and rewrites the gender
  column so both fields are anonymized together.
- **Custom name libraries.** `export_names_library()` writes the current library
  as an editable module; `load_names_library(path=…)` (or the
  `DATA_SAMPLER_NAMES` env var) activates one for the session; and
  `install_names_library(path=…)` installs one permanently into the package.
- **Report screen shows a "reproduce this in Python" snippet** — the exact
  `ds.load_file / sample / anonymize / reduce_columns / save_output` calls that
  reproduce the run — and it's included in the saved report `.txt`. Every TUI
  capability now maps to the public API.
- **Remote inputs.** `load_file` (and the TUI file screen) accept `http(s)` URLs
  (e.g. GitHub raw links). The DuckDB engine loads `httpfs` on demand so it can
  sample remote `s3://` / `https://` **Parquet with HTTP range requests** — a
  multi-GB remote file is sampled out-of-core without downloading it whole.
- **TUI restyle** — btop-inspired rounded, quiet-until-focused inputs / selects
  / switches and outlined (non-solid) buttons; the columns table's per-stat
  `mean`/`median`/`mode`/`sd` columns; the report save clarifies it includes the
  histograms.

## v3.4.1 — 2026-07-24

- **TUI columns table now breaks the single summary cell into separate
  `mean` / `median` / `mode` / `sd` columns.** Numeric columns fill all four;
  non-numeric columns show `mode` (their most frequent value) with dashes for
  the numeric-only stats. `ColumnStats` gained a `mode` field, populated for
  every column kind by `compute_column_stats`.
- **Bulk configuration via multi-select on the columns screen.** Ctrl-click
  toggles individual rows, shift-click selects a contiguous range (keyboard:
  `space` toggles the cursor row, `x` clears the selection). Any anonymizer
  choice, stratification-skip, or reduction-skip then applies to every
  selected column at once; the panel title shows the selection count.
- **Columns can now be excluded from the PCA reduction from the TUI.** A new
  “skip from reduction (PCA)” switch (and the `d` key) sets a per-column
  `skip_reduce` flag that is passed to `reduce_columns(exclude=…)`, so chosen
  numeric columns pass through the reduction unchanged. The columns table
  gained a `reduce` column showing each column's candidacy / skip state, and
  excluded columns are listed in the run report.
- **Undo / redo for column configuration** (`ctrl+z` / `ctrl+y`), keeping well
  over the last ten steps. Bursts of edits to one option field coalesce into a
  single undo step.
- **File screen: refresh the directory browser** with `ctrl+r` or the new
  ⟳ refresh button, so files created after launch appear without restarting.
- **Report screen: save the report + column histograms to a text file** with
  `ctrl+s` or the 💾 save .txt button; it writes `<sample-name>_report.txt`
  next to the sample output.

## v3.4.0 — 2026-07-24

- **Added optional PCA column reduction** (`data_sampler.reduce`): after
  sampling (and anonymization, if any), the numeric block of the outgoing
  sample can be collapsed into its first *k* principal components — so the
  delivered file is narrow as well as short. *k* is controlled one of two
  ways: a component count (`--reduce-components N` /
  `reduce_columns(df, n_components=N)`, capped at the number of usable
  numeric columns) or a variance target (`--reduce-variance R` /
  `variance_ratio=R`: the fewest components whose cumulative
  explained-variance ratio reaches R). Non-numeric columns are always
  preserved; `--reduce-exclude` passes chosen numeric columns (e.g.
  identifiers) through unchanged, `--reduce-prefix` renames the `PC*`
  output columns when the source already has one, and the tool warns when
  an included column looks like an identifier. Reduce flags are validated
  before sampling starts, so a typo never costs a full out-of-core run. Missing values are mean-imputed (the row count
  never changes), constant/all-missing columns pass through with a note, and
  columns are z-scored by default (`--reduce-no-standardize` opts out) so
  large-unit columns cannot dominate. The exact-SVD implementation is pure
  numpy — no new dependency — and deterministic across runs and platforms
  (sign-fixed singular vectors). New public API: `reduce_columns`,
  `ReductionResult`, `format_reduction_report`.
- **Every reduction reports its rationale**, not just the variance kept: the
  report lists per-component and cumulative explained variance, **the groups
  of correlated columns that move together** (connected components of the
  |r| ≥ 0.7 correlation graph — on standardized data PCA diagonalizes
  exactly that correlation matrix, so the groups are a faithful account of
  what compressed), each component's top driving columns, and any
  imputation/passthrough/identifier notes. The full labelled correlation
  matrix is available on `ReductionResult.correlation_matrix`.
- Wired into all three surfaces: CLI flags on both the pandas and DuckDB
  engine paths (the reduction always runs on the small sampled frame, never
  the full source), a `reduce` control in the TUI run bar with the rationale
  shown on the report screen, and the Python API. Output files gain a
  `_pca{k}` tag suffix (e.g. `data_sample_500_anon_pca3.csv`).

## v3.3.1 — 2026-07-23

- **Fixed the second CI-only TUI race** (caught, again, by the release CI —
  it blocked v3.3.0's PyPI publish): a stale `Changed` message (mount-time
  echo or superseded edit, delivered late because every Textual widget runs
  its own message pump) could roll an anonymizer choice back to `none` and
  wipe its options. All Select/Input/Switch handlers now drop any `Changed`
  whose value no longer matches the widget's current value — provably stale
  — which also subsumes the v3.2.1 duplicate-highlight fix. v3.3.0 was never
  published to PyPI; this release carries all of its changes.

## v3.3.0 — 2026-07-23

- **Two-phase narrow sampling in the DuckDB engine.** The expensive phase of a
  sample (the per-stratum window sort, or the reservoir buffer) now runs over
  only the stratification columns plus a stable row id — `file_row_number` for
  single-file Parquet sources, a positional id for DataFrames — and a second
  pass fetches the winning rows with every column. Measured on 300k rows ×
  401 columns: stratified sampling 2.5× faster on Parquet and 9.5× faster for
  DataFrame sources (whose wide payload never enters the SQL engine; winners
  return as dtype-preserving pandas slices). CSV/TSV/JSON and multi-file
  Parquet globs keep the previous single-pass shape (per-file row numbers are
  not a global id — verification caught that a glob would otherwise be
  silently over-sampled — and text formats must re-parse per scan anyway).
  No behavioral change to counts, allocation, NaN strata, or determinism;
  note that *which* rows a given seed selects changes vs v3.2.1 for
  single-file Parquet and DataFrame sources.

## v3.2.1 — 2026-07-23

- **Fixed a TUI race that only surfaced on slow machines** (caught by the
  v3.2.0 release CI on Windows runners; it blocked the PyPI publish): a late
  mount-time row-highlight for the already-selected column could re-sync the
  config panel and queue a stale `Changed` message that reset a just-made
  anonymizer choice back to `none`. Duplicate highlights are now no-ops and
  the panel sync no longer queues gratuitous widget updates. This is the
  first release published to PyPI (v3.2.0's publish run failed on this bug).

## v3.2.0 — 2026-07-23

- **Measured parallel scaling** (20M-row Parquet, 12-core machine,
  10k-row sample): stratified sampling 14.5 s → 3.9 s from 1 → 8 threads
  (3.7×), `stats()` 9.5 s → 1.7 s (5.6×), reservoir sampling 0.10 s —
  ~50× faster than the pandas path on the same file, with no ~0.9 GB
  in-memory materialization.

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
- **Wired the DuckDB engine into the CLI** (Block P6 of the v3.2 performance
  & scale effort): new `--engine {auto,pandas,duckdb}` flag (default `auto`
  — DuckDB for Parquet/large inputs, pandas otherwise), plus `--threads` and
  `--memory-limit` to tune the engine. The engine path supports
  `--random`/`--skip`/`--seed`/`--anon` and `--suggest` (suggestions computed
  from the engine's approximate stats, without materializing the full
  input); `--interactive` remains pandas-only. When the pandas path is used
  on a large input, a note suggests the engine. Auto-selection falls back to
  pandas if the optional `duckdb` dependency is absent.
- **Fixed 22 issues found in a pre-release six-lens adversarial audit** (one
  accepted divergence documented in `TROUBLESHOOTING.md`): the DuckDB
  engine's `stats()` no longer crashes on LIST/STRUCT columns or on float
  files containing real NaN values (every NaN-sensitive aggregate is now
  NaN-filtered); `TIME`/`INTERVAL` and nested types (`LIST`, `STRUCT`,
  `MAP`, …) classify as `"other"` and are never treated as numeric or
  datetime; seeded stratified sampling is now deterministic even under
  allocation-remainder ties (stratum order is pinned before allocation);
  columns-oriented JSON (the pandas `to_json()` default) is detected and
  refused with guidance instead of silently sampling a single row; `--skip`
  column names are validated against the source; `--engine auto` now
  genuinely falls back to pandas on a DuckDB read failure instead of
  raising.
- Alias anonymizer kinds (e.g. `"seq"` for `sequential_id`) now canonicalize
  on `assign`, so the interactive wizard no longer silently resets them to
  `none`; accepting a seeded assignment in the wizard preserves its options
  instead of discarding them. HyperLogLog-based unique counts are clamped to
  the row count so they can no longer push `unique_pct` past 100%.
- Performance/robustness: row counts are cached per source (3–4 fewer full
  scans per CLI run); `compute_stats` skips the useless top-values pass for
  numeric columns (~1.9× faster); the TUI now computes stats off the event
  loop so loading a large file no longer freezes the UI; report histograms
  exclude ±inf from their percentage denominators and skip near-unique
  columns instead of drawing a meaningless top-8.

## v3.1.0 — 2026-07-23 (released with v3.2.0)

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
