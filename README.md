# data-sampler

Creates representative samples from data files, using stratified sampling to
preserve the statistical variety of your data — with optional per-column
anonymization and a colorful terminal UI.

Everything ships as a single Python package: launch the TUI with one
function call or command, or use the sampling/anonymization functions
directly from Python.

## Install

```sh
pip install data-sampler        # once released on PyPI
# from a clone, today:
pip install -e ".[dev]"
```

Requires Python 3.10+.

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

1. **File screen** — type a path or pick a file from the directory browser;
   Excel files take an optional sheet name.
2. **Columns screen** — every column with its type, missing %, unique count,
   distribution sparkline, and summary (modelled after the Data Wrangler
   VS Code extension). Select a column to see full stats and distribution
   bars, choose an anonymizer for it, and toggle whether it should be
   skipped when preserving statistical variety (stratification). Set the
   sample size, output folder, optional seed, and run.
3. **Report screen** — the stratification comparison and anonymization summary
   on the left, and a **column histograms** panel on the right showing every
   column's source-vs-sample distribution (numeric columns share bin edges;
   others use the source's top categories) so you can see at a glance how well
   the sample preserved each column. The output path is shown too.

Key bindings: `ctrl+r` run sample, `a` auto-suggest anonymizer types,
`s` toggle stratification skip,
`escape` back, `ctrl+q` quit.

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

ds.save_output(anon, "data.xlsx", tag="sample_500_anon")
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
(e.g. `data_sample_500_anon.csv`).

## Development

```sh
pip install -e ".[dev]"
pytest                 # full suite, incl. headless TUI tests
python -m build        # build the wheel + sdist into dist/
```

Logging is controlled by `DATA_SAMPLER_LOG` (`quiet`/`info`/`verbose`) and
`DATA_SAMPLER_LOG_FILE`. See `ROADMAP.md` for planned work and
`TROUBLESHOOTING.md` for known failure modes.

PyPI releases are manual and happen only after extensive testing.

---

Built with the assistance of [Claude Code](https://claude.ai/code) (Anthropic).
