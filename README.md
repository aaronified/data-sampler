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
3. **Report screen** — side-by-side original-vs-sample distribution bars per
   stratification column, the anonymization summary, and the output path.

Key bindings: `ctrl+r` run sample, `s` toggle stratification skip,
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
| `random_string` | Random character sequences, unique per value | `length` (8), `charset` (`alphanumeric`, `letters`, `digits`, `hex`), `prefix` (`""`) |
| `hex` | Shorthand for `random_string` with `charset="hex"` | `length` (8) |

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
