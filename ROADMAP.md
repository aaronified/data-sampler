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
- [x] **Block C — column-level histograms in the report screen.** Per-column
  source-vs-sample distribution histograms (`column_histogram_data` /
  `format_column_histograms`): numeric columns share bin edges, others use the
  source's top categories, computed from the pre-anonymization sample. Shown
  in a new right-hand panel on the TUI report screen and in a "COLUMN
  DISTRIBUTIONS" section of the CLI output.

## v3.2 — performance & scale

Goal: handle very large inputs (toward billions of rows × thousands of columns)
as fast as possible without a C rewrite. The wins are algorithmic and
architectural (vectorization, out-of-core columnar engine, streaming sampling,
approximate stats), not language-level. Each block is independent and ships on
its own commit.

- [ ] **Block P1 — vectorize the anonymizers.** Replace the per-unique Python
  loop + `Series.map(dict)` with a single `pd.factorize` (dictionary-encode)
  and a vectorized gather (the in-process equivalent of a native join against a
  mapping table). Sequential IDs become `np.arange`; numeric/datetime jitter
  become vectorized numpy draws. Preserves the consistent-mapping guarantee,
  seed reproducibility, NaN/dtype handling, and the public `build_mapping`
  API. Pure win, no new dependencies.
- [ ] **Block P2 — DuckDB out-of-core engine.** An optional engine that pushes
  loading, stratification, and sampling into DuckDB (vectorized, multi-threaded,
  larger-than-memory). Set `PRAGMA threads` for parallelism and a memory limit
  so it spills to disk instead of OOM-ing. Selected automatically for large
  inputs (or via a flag), with the pandas path kept as the default for small/
  medium data. CSV/Excel/JSON still supported (DuckDB reads CSV/Parquet
  natively; Excel is converted).
- [ ] **Block P3 — Parquet fast path (separate).** A dedicated Parquet code
  path with column pruning and predicate/projection pushdown so only the needed
  columns/rows are read — the biggest I/O win, independent of the compute
  language. CSV/Excel keep working via the compatibility path. Surface a warning
  about the memory speedbump when a Parquet dataset is too large to materialize,
  steering the user to the streaming/DuckDB path.
- [ ] **Block P4 — streaming sampling algorithms.** Reservoir and Bernoulli
  sampling so the full dataset is never materialized. Stratified: two-pass
  (count strata → reservoir-sample per stratum) or single-pass weighted
  reservoir. This is the algorithmic key to billion-row inputs; exposed through
  the engine so both the CLI and TUI benefit.
- [ ] **Block P5 — approximate stats at scale.** HyperLogLog for distinct
  counts and approximate quantiles/histograms (DuckDB `approx_count_distinct`,
  `approx_quantile`, histogram aggregation) so per-column stats over billions of
  rows stay cheap. Fall back to exact stats for small inputs.

## Later

- Optional PyInstaller EXE build of the TUI (replaces the old Tkinter EXE).
- Rust/Polars-on-Arrow native engine (pyo3) as an alternative to DuckDB.
- GPU acceleration (RAPIDS cuDF) for the aggregation-heavy paths.
- Distributed backend (Dask / Ray Data / Spark) for multi-machine scale.
