# Warning Lead Time for Clinical AI Reliability Under Distribution Shift

Code accompanying the manuscript *"Warning Lead Time for Clinical AI Reliability Under Distribution Shift."*

The study measures **Warning Lead Time (Δα_shell)** — the interval along an ordered shift trajectory between a monitoring alert and the Critical Failure Point at which model performance crosses a prespecified acceptability threshold (τ = 0.70).

## What this code does

Four experiments in a 2 × 2 design: two shift mechanisms (Fourier Domain Adaptation and pixel-space linear blending) crossed with two imaging domains (chest radiography and histopathology), each across three architectures (ResNet18, DenseNet121, ViT-B/16) and three seeds — 36 runs in total.

For each run, models are fine-tuned on a source domain, then evaluated across 20 progressively increasing shift levels. At every level, five monitoring indicators are recorded alongside task performance, so that alert onset times can be compared against the failure point.

## Repository structure

```
src/                              Reusable modules
  enhanced_ews_pipeline.py        Core analysis: onset detection, threshold
                                  calibration, failure detection, Safety Shell,
                                  false-alarm surrogates, Warning Lead Time
  cross_domain_shift.py           Fourier Domain Adaptation shift operator
  histopath_data.py               Histopathology data loading (PCam → BreakHis)
  publication_utils.py            Table and figure styling helpers

scripts/                          Experiment runners (one per experiment)
  corrected_main_pipeline.py                        FDA + radiography
  corrected_main_pipeline_linearshift.py            Linear blend + radiography
  corrected_main_pipeline_histopath.py              FDA + histopathology
  corrected_main_pipeline_histopath_linearshift.py  Linear blend + histopathology

analysis/                         Post-hoc analyses (run on saved .npz outputs)
  rerun_two_family_final.py       Primary Warning Lead Time and coverage results
  compute_auroc_and_interception.py  Detection AUROC; intercepted-failure fractions
  extract_supplementary_tables.py    Supplementary Tables S1–S3
  strengthening_experiments.py       Hyperparameter sensitivity; Page-Hinkley baseline
  recover_histopath_threshold.py     Verifies the histopathology decision threshold
```

## Two-stage workflow

The experiment runners are GPU-bound and write per-run `.npz` files containing
raw per-level signals. Every analysis in the manuscript then runs **against those
saved files**, with no retraining and no GPU required.

**Stage 1 — run the experiments** (GPU, hours per experiment):

```bash
python scripts/corrected_main_pipeline.py
```

Each runner writes `.npz` files, one per (architecture, seed).

**Stage 2 — reproduce the manuscript's numbers** (CPU, minutes):

```bash
python analysis/rerun_two_family_final.py          # primary results
python analysis/compute_auroc_and_interception.py  # Sections 5.4, 5.5
python analysis/extract_supplementary_tables.py    # Tables S1–S3
```

## Environment note

The experiment runners in `scripts/` were written and executed on **Kaggle
Notebooks**, and contain hardcoded `/kaggle/input/...` dataset paths. To run
them elsewhere, update those paths to point at local copies of the datasets
listed below. The `analysis/` scripts have no such dependency — they read only
the `.npz` files produced in Stage 1, and run anywhere.

## Datasets

All four are publicly available:

| Role | Dataset | Source |
|---|---|---|
| Radiography source | PneumoniaMNIST (MedMNIST v2) | https://medmnist.com/ |
| Radiography target | RSNA Pneumonia Detection Challenge | https://www.kaggle.com/c/rsna-pneumonia-detection-challenge |
| Histopathology source | PatchCamelyon (Histopathologic Cancer Detection) | https://www.kaggle.com/c/histopathologic-cancer-detection |
| Histopathology target | BreakHis | https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/ |

## Installation

```bash
pip install -r requirements.txt
```

## Reproducibility

Each run is seeded (seeds 0, 1, 2). Thresholds are calibrated by a
leave-one-run-out procedure using only nominal-region data, never the failure
outcomes they are subsequently evaluated against. False-alarm rates are
estimated from 400 dependence-preserving surrogate replicates per run, drawn
with a shared resampling pattern across detectors so that real cross-detector
dependence is preserved.

Exact GPU determinism is not guaranteed across hardware; the manuscript's
conclusions rest on effects substantially larger than seed-level variation, and
all per-seed values are reported in the Supplementary Material.

## Citation

Citation details will be added upon publication.

## License

See `LICENSE`.
