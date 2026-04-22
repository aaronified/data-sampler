# data-sampler

Creates representative samples from data files, automatically using stratified sampling to preserve the statistical distribution of categorical columns.

## GUI (Windows EXE)

`dist/DataSampler.exe` is a standalone Windows application — no Python installation required. Double-click to launch.

1. **Input file** — click Browse to pick your data file.
2. **Sheet name** — appears automatically for Excel files; leave blank for the first sheet.
3. **Sample count** — number of rows to sample.
4. **Output folder** — where to save the result; defaults to the same folder as the source file.
5. **Sampling mode** — Stratified (default) or Random.
6. Click **Run Sample**. Progress and the stratification report stream into the log window.

To rebuild the EXE after modifying the scripts:

```sh
# from the repo root, with the venv active
venv\Scripts\activate
pyinstaller --onedir --windowed --noupx --name "DataSampler" data-sampler-gui.py
# output: dist\DataSampler\  — zip the contents and share DataSampler.zip
# recipients extract the zip and run DataSampler.exe from the extracted folder
```

---

## Python usage

### Interactive mode (no arguments)

```sh
python data_sampler.py
```

Prompts for file path, sample count, sheet name (Excel only), and sampling mode.

### CLI mode

```sh
python data_sampler.py <source> <count> [--sheet SHEET] [--outdir DIR] [--random]
```

| Argument | Description |
| --- | --- |
| `source` | Path to the source data file |
| `count` | Number of rows to sample |
| `--sheet` | Sheet name for Excel files (default: first sheet) |
| `--outdir` | Output folder (default: same folder as source file) |
| `--random` | Use pure random sampling instead of stratified |

#### Examples

```sh
python data_sampler.py data.csv 500
python data_sampler.py report.xlsx 200 --sheet "Sheet2"
python data_sampler.py data.csv 100 --outdir C:\samples --random
```

---

## Supported file formats

| Format | Extensions |
| --- | --- |
| CSV | `.csv` |
| TSV | `.tsv` |
| JSON | `.json` |
| Excel | `.xlsx`, `.xls` |
| Parquet | `.parquet` |

## How it works

**Stratified sampling (default):** Automatically identifies columns suitable for stratification — categorical or low-cardinality columns with 2–100 unique values. It avoids long text columns and columns that look like IDs. Rows are then sampled proportionally from each combination of stratification groups, ensuring the sample mirrors the original distribution. A stratification report with side-by-side bar charts is printed to the terminal.

**Pure random sampling (`--random`):** Draws rows uniformly at random with no stratification.

If no suitable stratification columns are found, the tool falls back to pure random sampling automatically.

## Output

The sampled file is saved in the output folder (or next to the source file if none is specified), named `{original_stem}_sample_{count}{ext}` (e.g., `data_sample_500.csv`).

## Python requirements

```sh
pip install -r requirements.txt
# or manually: pip install pandas openpyxl pyarrow
```

---

Built with the assistance of [Claude Code](https://claude.ai/code) (Anthropic).
