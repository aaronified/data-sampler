# Changelog

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
