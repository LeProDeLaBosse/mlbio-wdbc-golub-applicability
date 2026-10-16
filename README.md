# Golub weighted voting on WDBC

Applicability study of the Golub et al. (1999) weighted-voting classifier on the
Wisconsin Diagnostic Breast Cancer data. Personal seed: **32**.

## Run everything (one command)

```bash
uv run python scripts/run_all.py
```

This runs the analysis on `data/wdbc.csv` (all tables and figures in
`results/`), then compiles the LaTeX report (needs `pdflatex` installed) into `report.pdf`.

## Files

- `scripts/pipeline.py`: exploration, weighted voting (10 features, fold-wise estimation),
  5-fold x 3 repeated stratified CV, selection stability, logistic-regression comparison.
- `report/report.tex`: LaTeX source of the report; the numbers and tables it uses are
  written by `pipeline.py` into `report/generated/`, the figures come from `results/`.
- `scripts/run_all.py`: runs `pipeline.py`, then compiles `report/report.tex`.

## References

Golub et al. (1999) Science 286:531-537; WDBC, UCI, DOI 10.24432/C5DW2B;
Wolberg et al. (1994) Cancer Letters 77:163-171; Pedregosa et al. (2011) JMLR 12:2825-2830.
No other code was reused.
