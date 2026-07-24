# data-sampler

[![PyPI](https://img.shields.io/pypi/v/data-sampler)](https://pypi.org/project/data-sampler/)
[![Python](https://img.shields.io/pypi/pyversions/data-sampler)](https://pypi.org/project/data-sampler/)

**Hand someone data that looks and behaves like your production data, isn't
your production data, and provably kept its statistical variety — in one
command.**

```sh
data-sampler customers.xlsx 500 --suggest
```

Use it to:

- **Share a realistic slice with a vendor or contractor** — names, ids,
  emails, salaries, and dates anonymized, but every distribution, duplicate,
  and group structure intact.
- **Attach a repro to a bug report** without leaking the real records that
  trigger it.
- **Cut a 2 GB export down to 500 rows** for a demo or prototype that still
  behaves like the real thing.
- **Build test fixtures that mirror production skew** instead of uniform toy
  data.
- **Give a class or workshop realistic data** without a data-sharing
  agreement.
- **Pull 10,000 representative rows from a 100M-row Parquet file** without
  loading it — the optional DuckDB engine samples out-of-core, in parallel.
- **Prove the sample is representative**: every run produces a side-by-side
  source-vs-sample distribution report, per column.

How: stratified sampling preserves the statistical variety of your data
(strata are detected automatically), and every anonymizer maps each unique
original value to exactly one replacement — so repeated values stay repeated
and the joint distributions survive anonymization. Everything ships as a
single Python package: a colorful terminal UI for non-programmers, a headless
CLI, and a plain Python API.

## Install

```sh
pip install data-sampler             # from PyPI
pip install "data-sampler[large]"    # + the out-of-core DuckDB engine for huge files
```

Requires Python 3.10+. For development, clone the repo and
`pip install -e ".[dev]"`.

## Terminal UI

```sh
data-sampler            # no arguments → opens the TUI
data-sampler-tui        # explicit TUI entry point
python -m data_sampler  # same as data-sampler
```

Or from Python:

```python
import data_sampler

data_sampler.run_tui()                 # file picker first
data_sampler.run_tui("data.csv")       # pre-load a file
```

The TUI is a panel-based dashboard (think btop / lazydocker):

1. **File screen** — type a path or pick a file from the directory browser
   (**`ctrl+r`** or the **⟳ refresh** button re-scans the folder so files
   created since launch show up); Excel files take an optional sheet name.
2. **Columns screen** — every column with its type, missing %, unique count,
   distribution sparkline, and per-stat **mean / median / mode / sd** columns
   (modelled after the Data Wrangler VS Code extension). Select a column to
   see full stats and distribution bars, choose an anonymizer for it, and
   toggle whether it should be skipped when preserving statistical variety
   (stratification) or excluded from the PCA reduction. **Multi-select** rows
   to configure them in bulk: **ctrl-click** toggles individual rows,
   **shift-click** selects a range (**space** toggles the cursor row from the
   keyboard, **`x`** clears) — then any anonymizer / skip / reduce choice
   applies to every selected column at once. **`ctrl+z` / `ctrl+y`** undo and
   redo column-config changes (at least the last ten steps). Set the sample
   size, output folder, optional seed, an optional PCA column reduction
   (`reduce`: N components or a variance target), and run.
3. **Report screen** — the stratification comparison and anonymization summary
   on the left, and a **column histograms** panel on the right showing every
   column's source-vs-sample distribution (numeric columns share bin edges;
   others use the source's top categories) so you can see at a glance how well
   the sample preserved each column. The output path is shown too, and
   **`ctrl+s`** (or **💾 save .txt**) writes the report and histograms to a
   `*_report.txt` beside the sample.

Key bindings — columns screen: `ctrl+r` run sample, `a` auto-suggest
anonymizer types, `s` toggle stratification skip, `d` toggle reduction skip,
`space` select row, `x` clear selection, `ctrl+z`/`ctrl+y` undo/redo,
`escape` back. Report screen: `ctrl+s` save report to text, `n` new file.
Everywhere: `ctrl+q` quit.

## CLI (headless)

```sh
data-sampler <source> <count> [options]
```

| Option | Description |
| --- | --- |
| `--sheet NAME` | Sheet name for Excel files (default: first sheet) |
| `--outdir DIR` | Output folder (default: same folder as source file) |
| `--random` | Pure random sampling instead of stratified |
| `--seed N` | Seed for reproducible sampling and anonymization |
| `--skip COL[,COL]` | Exclude column(s) from stratification (repeatable) |
| `--anon COL=KIND[:k=v,...]` | Anonymize a column (repeatable) |
| `-i`, `--interactive` | Guided workflow: choose an anonymizer type per column from a menu |
| `--suggest` | Auto-assign a suggested anonymizer type to each column from its stats |
| `--reduce-components N` | PCA column reduction: replace the numeric columns with the first `N` principal components |
| `--reduce-variance R` | PCA column reduction: keep the fewest components whose cumulative explained variance reaches `R` (0 < R < 1) |
| `--reduce-exclude COL[,COL]` | Numeric column(s) to keep out of the reduction, e.g. identifiers (repeatable) |
| `--reduce-prefix PREFIX` | Name prefix for the component columns (default `PC` → `PC1`, `PC2`, …) |
| `--reduce-no-standardize` | Skip the per-column z-scoring before PCA |
| `--engine {auto,pandas,duckdb}` | Sampling engine (default `auto`: DuckDB for Parquet/large inputs, pandas otherwise) |
| `--threads N` | DuckDB engine: number of threads (default: all cores) |
| `--memory-limit SIZE` | DuckDB engine: memory limit before spilling to disk (e.g. `8GB`) |
| `--tui` | Open the TUI (optionally preloading `source`) |

Examples:

```sh
data-sampler data.csv 500
data-sampler report.xlsx 200 --sheet "Sheet2" --outdir C:\samples
data-sampler data.csv 100 --skip region,notes --seed 7 \
    --anon "name=names" \
    --anon "cust_id=sequential_id:start=1000,interval=7" \
    --anon "salary=numeric_jitter:pct=0.1" \
    --anon "email=hex:length=12"

# large / out-of-core: sample a Parquet file in parallel with DuckDB
data-sampler huge.parquet 10000 --engine duckdb --threads 8 --memory-limit 8GB --suggest

# narrow the sample too: collapse the numeric columns into 3 principal
# components (or keep however many retain 90% of the variance)
data-sampler wide.csv 500 --reduce-components 3 --reduce-exclude cust_id
data-sampler wide.csv 500 --reduce-variance 0.9
```

## Python API

```python
import data_sampler as ds

df = ds.load_file("data.xlsx", sheet="Sheet2")

# Data Wrangler-style column stats
for s in ds.compute_stats(df):
    print(s.name, s.kind, s.unique, s.summary())

# representative sample; 'notes' never used for stratification
result = ds.sample(df, 500, exclude_columns=["notes"], random_state=7)
print(ds.format_stratification_report(df, result))

# per-column source-vs-sample histograms (or ds.column_histogram_data for the raw numbers)
print(ds.format_column_histograms(df, result.data))

# anonymize chosen columns of the sample (consistent mapping, NaN preserved)
anon = ds.anonymize(
    result.data,
    {
        "name": "names",
        "cust_id": ("sequential_id", {"start": 1000, "interval": 7}),
        "salary": ("numeric_jitter", {"pct": 0.1}),
        "email": {"kind": "hex", "length": 12},
    },
    seed=7,
)

# optionally collapse the numeric columns into principal components
red = ds.reduce_columns(anon, variance_ratio=0.9, exclude=["cust_id"])
print(ds.format_reduction_report(red))   # variance kept + correlated groups

ds.save_output(red.data, "data.xlsx", tag="sample_500_anon_pca")
```

## Try it: bundled example

The repo ships a 1,000-row dummy dataset, [examples/employees.csv](examples/employees.csv),
built to be stratifiable: `department`, `region`, and `employment_type` have
skewed categorical distributions, `performance_rating` is low-cardinality
numeric, and `employee_id`/`full_name`/`email`/`salary` are there to
anonymize.

```text
employee_id,full_name,email,department,region,employment_type,performance_rating,salary
E1001,Emily Lee,emily.lee001@example.com,Sales,North,Full-time,4,62000
E1002,Joshua Clark,joshua.clark002@example.com,Finance,South,Full-time,4,50000
E1003,Donald Martin,donald.martin003@example.com,Operations,East,Contract,3,68500
```

### In the TUI

```sh
data-sampler examples/employees.csv --tui
```

The columns screen opens with the stats table. Try: press **`a`** to
auto-suggest an anonymizer type for every column, then adjust — select
`full_name` and set its anonymizer to **names**; select `employee_id` and
choose **sequential id** (start 1000); select `salary` and choose **numeric
jitter**; select `performance_rating` and flip **skip when stratifying** to
keep it out of the variety-preservation logic. Set rows to `100`, seed to
`42`, and press `ctrl+r` — the report screen shows how closely the sample
tracks the original distributions.

### With the Python functions

```python
import data_sampler as ds

df = ds.load_file("examples/employees.csv")

result = ds.sample(df, 100, random_state=42)          # stratifies automatically
print(ds.format_stratification_report(df, result))

anon = ds.anonymize(
    result.data,
    {
        "full_name": "names",
        "employee_id": ("sequential_id", {"start": 1000}),
        "salary": "numeric_jitter",
        "email": {"kind": "hex", "length": 10},
    },
    seed=42,
)
ds.save_output(anon, "examples/employees.csv", tag="sample_100_anon")
```

### From the CLI

```sh
data-sampler examples/employees.csv 100 --seed 42 \
    --anon "full_name=names" \
    --anon "employee_id=sequential_id:start=1000" \
    --anon "salary=numeric_jitter" \
    --anon "email=hex:length=10"
```

The run stratifies on `employment_type`, `region`, and `department`, and the
report shows original vs. sample side by side (excerpt):

```text
  Column: 'employment_type' (3 categories)
          Value          Original                    Sample
  ─────────────────────────────────────────────────────────────────────
       Contract  ██░░░░░░░░░░░░░  10.1%  █░░░░░░░░░░░░░░   9.0%
      Full-time  ███████████████  68.9%  ███████████████  68.0%
      Part-time  ████░░░░░░░░░░░  21.0%  █████░░░░░░░░░░  23.0%
  ─────────────────────────────────────────────────────────────────────
         Totals                   1000                     100
```

The anonymized sample keeps the structure but none of the identities —
repeated values still repeat, salaries stay within ±20 % of the originals:

```text
employee_id,full_name,email,department,region,employment_type,performance_rating,salary
1000,Ravi Andersen,6a78c49ea2,Engineering,South,Full-time,1,62264
1001,Thomas Gomez,0e32684b27,Engineering,North,Part-time,3,102743
1002,Fatima Singh,b95e909348,Operations,North,Full-time,3,46793
```

### Notebook and launcher scripts

- [examples/using_data_sampler.ipynb](examples/using_data_sampler.ipynb) —
  the full package walkthrough as an executed Jupyter notebook (load →
  stats → sample → anonymize → save, with outputs included).
- [scripts/run-tui.sh](scripts/run-tui.sh) — opens the TUI on Linux (any
  distro) and macOS; falls back from the `data-sampler` command to
  `python3 -m data_sampler` and prints install instructions if neither is
  available.
- [scripts/run-tui.bat](scripts/run-tui.bat) — the same for Windows
  (double-clickable).

Both scripts pass arguments through, e.g. `./scripts/run-tui.sh data.csv`.

## Anonymizers

Every anonymizer maps each unique original value to exactly one replacement,
so repeated values stay repeated and the column's distribution — the
statistical variety this tool exists to preserve — survives anonymization.
Missing values are left as missing. All anonymizers accept a seed (via
`anonymize(..., seed=N)` or `--seed`) for reproducible output.

| Kind | Replaces values with | Options (defaults) |
| --- | --- | --- |
| `names` | Realistic names from a bundled library of first, middle, and last names | `style`: `first_last`, `first_middle_last`, `last_first`, `first`, `last` |
| `sequential_id` | `start`, `start+interval`, ... in order of first appearance | `start` (1), `interval` (1), `prefix` (`""`), `width` (0, zero-pads) |
| `numeric_jitter` | A random number within ±`pct` of the original | `pct` (0.2 = ±20 %), `round_to` (decimal places) |
| `datetime_jitter` | A date/time shifted by a random offset within ±`max_delta` | `max_delta` (`"7D"`; any `pandas.Timedelta` string), `unit` (`"s"`; jitter resolution) |
| `random_string` | Random character sequences, unique per value | `length` (8), `charset` (`alphanumeric`, `letters`, `digits`, `hex`), `prefix` (`""`) |
| `hex` | Shorthand for `random_string` with `charset="hex"` | `length` (8) |

## Anonymization workflow

Rather than spell out every column by hand, you can drive a guided workflow —
give it your columns and pick a *type* for each. The three ways to do it share
one engine (`AnonymizationPlan`) and the same auto-suggestion (`suggest_type`),
which infers a type from each column's stats (datetime → datetime jitter,
name/email columns → names/hex, id-ish high-uniqueness columns → sequential id,
numbers → numeric jitter, free text → random string; categorical/boolean columns
are left alone so the categories you stratify on survive).

- **Choose from options (interactive):** `data-sampler data.csv 100 --interactive`
  walks each column and offers a numbered menu, defaulting to the suggested
  type — press Enter to accept or type a number to override.
- **Pre-specify through a function (Python):**

  ```python
  import data_sampler as ds

  df = ds.load_file("data.csv")
  plan = ds.AnonymizationPlan.suggest(df)          # auto-infer every column…
  plan.assign("salary", "numeric_jitter", pct=0.1) # …then override as needed
  plan.clear("region")
  anon = plan.apply(df, seed=7)                     # runs ds.anonymize under the hood
  ```

- **Click in the TUI:** open the columns screen, select a column, and pick its
  anonymizer — or press **`a`** to auto-suggest a type for every column at once,
  then tweak. The `anonymizer` column shows each choice at a glance.

`--suggest` applies the suggestions non-interactively (columns you also set with
`--anon` keep your explicit choice).

## How sampling works

**Stratified (default):** columns suitable for stratification are detected
automatically — categorical or low-cardinality columns with 2–100 unique
values; long text and ID-like numeric columns are avoided, as are any
columns you mark as skipped. Rows are grouped by the joint combination of
all selected columns and sampled proportionally per group, so the sample
mirrors the original joint distribution. Missing values count as their own
category. A side-by-side distribution report is produced for every run.

**Pure random (`--random`):** rows are drawn uniformly at random.

If no suitable stratification columns exist, the tool falls back to pure
random sampling automatically.

### Reducing columns (PCA)

Sampling narrows the *rows*; the optional PCA step narrows the *columns* of
the outgoing sample. It replaces the numeric block with its first *k*
principal components (`PC1..PCk`), controlled one of two ways:

- **`--reduce-components N`** — `N` components in the output (capped at the
  number of usable numeric columns, with a note when fewer are possible);
- **`--reduce-variance R`** — the fewest components whose cumulative
  explained-variance ratio reaches `R` (e.g. `0.9` keeps ≥ 90 % of the
  numeric variance).

Non-numeric columns (ids, categories, text, booleans, datetimes) are always
preserved, and `--reduce-exclude` keeps chosen numeric columns out too —
identifiers should be excluded, since an all-unique id forms its own
artificial component (the tool warns when it spots one). Missing values are
mean-imputed so the row count never changes; constant columns carry no signal
and pass through unchanged. Columns are z-scored first by default (PCA on the
correlation matrix), so a large-unit column such as a salary cannot dominate
the components; `--reduce-no-standardize` turns that off.

Every reduction prints its rationale: the variance each component retains,
**the groups of correlated columns** that move together (which is exactly the
redundancy PCA collapses — on standardized data PCA diagonalizes the
correlation matrix), and each component's top driving columns. The reduction
runs after anonymization, on the already-sampled rows, so the stratification
and histogram reports still describe the original columns.

## Large data: the out-of-core DuckDB engine

The default pandas path loads the whole file into memory. For inputs that are
too big for that (toward billions of rows, especially Parquet), install the
optional engine and let **DuckDB** do the work — multi-threaded, and able to
spill to disk, so only the resulting sample is ever materialized:

```sh
pip install "data-sampler[large]"
```

```python
from data_sampler.engine import DuckDBEngine, should_use_engine

# reads Parquet/CSV natively; only the sample (count rows) comes back as a DataFrame
with DuckDBEngine(threads=8, memory_limit="8GB") as engine:
    result = engine.sample("huge.parquet", 10_000, seed=42)   # stratifies automatically
    result.data.to_parquet("sample.parquet", index=False)

should_use_engine("huge.parquet")   # True — Parquet always benefits from pushdown
```

- **Parallel + out-of-core:** all cores by default; a `memory_limit` makes it
  spill instead of running out of memory.
- **Native readers:** Parquet is read with projection pushdown (only the scanned
  columns); CSV/TSV/JSON and pandas DataFrames work too. Excel still goes through
  the pandas path.
- **Streaming sampling:** reservoir sampling for the random case (exact count,
  single pass) and two-pass proportional sampling for the stratified case.
- **Reproducible:** pass `seed=` (seeded stratified runs go single-threaded so
  the result is deterministic; the distribution is preserved either way).

`large_materialization_warning(n_rows, n_cols)` returns a heads-up when a dataset
is big enough that loading it fully into pandas may exhaust memory — Parquet in
particular expands well beyond its compressed on-disk size.

Measured on a 20M-row Parquet file (5 columns, 12-core machine), sampling
10,000 rows:

| threads | stratified sample | reservoir sample | `stats()` |
| ---: | ---: | ---: | ---: |
| 1 | 14.5 s | 0.39 s | 9.5 s |
| 4 | 5.1 s | 0.16 s | 2.8 s |
| 8 | 3.9 s | 0.13 s | 2.1 s |
| 12 | 4.0 s | 0.10 s | 1.7 s |

The pandas path on the same file: 5.6 s total while materializing a ~0.9 GB
frame in RAM — the engine's reservoir sampling is ~50× faster and never
materializes the source at all.

## How it scales: the algorithms

Every trick used to handle millions-to-billions of rows and thousands of
columns, in one place.

### Millions (to billions) of rows

- **Out-of-core execution.** With the DuckDB engine, loading, stratification,
  and sampling run inside a vectorized, multi-threaded SQL engine with a
  memory limit and a temp directory — it spills to disk instead of OOM-ing,
  and only the resulting sample ever becomes a DataFrame.
- **Reservoir sampling** for the random case: a single streaming pass with
  O(sample size) memory, an exact row count, and `REPEATABLE(seed)`
  reproducibility that is independent of the file format and thread count.
- **Two-pass stratified sampling.** Pass 1 is a `GROUP BY` over the
  stratification columns — one row per stratum, tiny regardless of source
  size. Largest-remainder proportional allocation is computed on that tiny
  table in numpy. Pass 2 ranks rows per stratum with
  `row_number() OVER (PARTITION BY strata ORDER BY random())` and keeps the
  first *allocation* rows of each, joining the allocation table with
  `IS NOT DISTINCT FROM` so missing-value strata are sampled too. DuckDB
  parallelizes the partitions and spills the window sort if needed.
- **Parquet projection pushdown.** Pass 1 and the stats queries touch only
  the columns they reference, so a wide Parquet file is never read in full.
- **Two-phase narrow sampling.** The expensive phase (the per-stratum window
  sort, or the reservoir buffer) runs over *only the stratification columns
  plus a stable row id* — `file_row_number` for single-file Parquet, a
  positional id for DataFrames — and a second pass fetches just the winning
  rows with every column. The sort never carries the wide payload: measured
  2.5× on a 400-column Parquet file and 9.5× for wide DataFrames (whose
  payload never enters the SQL engine at all — winners come back as
  dtype-preserving pandas slices). CSV/JSON and multi-file Parquet globs
  keep the single-pass shape, which is the correct one there (text must be
  re-parsed per scan, and per-file row numbers aren't a global id).
- **Determinism engineering.** DuckDB's `GROUP BY` output order is
  nondeterministic, so the stratum order is pinned with `ORDER BY … NULLS
  LAST` before allocation (otherwise remainder ties break differently run to
  run); seeded stratified runs drop to a single thread because `random()`
  ordering is only reproducible that way, then restore the thread count.
- **Row-count caching.** `count(*)` runs once per source per engine session
  instead of once per operation.
- **Vectorized anonymizers.** The column is dictionary-encoded once with
  `pd.factorize`, each *unique* value gets one replacement, and the result is
  assembled by a fancy-index gather — the in-process equivalent of a native
  join against a mapping table. Cost scales with unique values plus one
  vectorized pass, never a Python loop over rows: sequential IDs are an
  `np.arange`, numeric/datetime jitter are single vectorized RNG draws.

### Thousands of columns

- **One scan for all scalar stats.** `DuckDBEngine.stats()` computes count,
  distinct, min/max/mean/std, and median for *every column in a single
  aggregate query* — one streaming pass over the data regardless of how many
  columns there are.
- **Sketches instead of sorts.** Distinct counts use HyperLogLog
  (`approx_count_distinct`) and medians use `approx_quantile` — streaming
  approximations with fixed memory per column, no per-column sort or full
  hash table. Exact mode (`approximate=False`) exists for small data, and
  approximate results are flagged on the stats object.
- **`distributions=False`** skips the per-column histogram/top-k passes
  entirely, so very wide tables get exactly one scan (this is what the CLI's
  `--suggest` uses to pick anonymizer types).
- **Bounded stratification search.** Candidate columns are screened in one
  aggregate pass (HLL cardinality + average text length), then a greedy
  fewest-categories-first selection keeps the *joint* stratum count at or
  below the sample size — so the group count stays bounded no matter how
  many columns the file has.
- **Pandas-path trims.** Numeric columns skip the top-values stringification
  (~1.9× faster stats), report histograms skip near-unique columns instead
  of hashing millions of ids for a meaningless top-8, and the TUI computes
  stats in a worker thread so the UI never freezes on load.
- **PCA column reduction on the way out.** `--reduce-components` /
  `--reduce-variance` collapse a wide numeric block into a handful of
  principal components *after* sampling, so the SVD runs on the small sampled
  frame (never the full source) and the delivered file is narrow as well as
  short.

### Correctness under scale

Details that only bite on real data, all regression-tested: DuckDB treats NaN
as a value rather than NULL, so every NaN-sensitive aggregate is filtered
(`FILTER (WHERE NOT isnan(…))`); missing-value strata survive the allocation
join via `IS NOT DISTINCT FROM`; HyperLogLog estimates are clamped to each
column's non-null count; and columns-oriented JSON (the pandas `to_json()`
default, which SQL engines parse as one giant row) is detected and refused
with guidance.

Known trade-offs: CSV sources are re-parsed per query (the streaming design —
convert to Parquet for repeated work), and a seeded stratified run gives up
multi-threading for reproducibility (unseeded runs use all cores).

## Supported formats

| Format | Extensions |
| --- | --- |
| CSV | `.csv` |
| TSV | `.tsv` |
| JSON | `.json` |
| Excel | `.xlsx`, `.xls` |
| Parquet | `.parquet` |

Output keeps the source format and is named
`{stem}_sample_{count}{ext}` — with an `_anon` suffix when anonymization ran
and a `_pca{k}` suffix when PCA column reduction ran
(e.g. `data_sample_500_anon_pca3.csv`).

## Development

```sh
pip install -e ".[dev]"
pytest                 # full suite, incl. headless TUI tests
python -m build        # build the wheel + sdist into dist/
```

Logging is controlled by `DATA_SAMPLER_LOG` (`quiet`/`info`/`verbose`) and
`DATA_SAMPLER_LOG_FILE`. See `ROADMAP.md` for planned work and
`TROUBLESHOOTING.md` for known failure modes.

Releases go to [PyPI](https://pypi.org/project/data-sampler/) via a
human-triggered, test-gated GitHub Actions workflow — see `RELEASING.md`.

---

Built with the assistance of [Claude Code](https://claude.ai/code) (Anthropic).
