# NIMIS

Neural framework for 1H-NMR mixture demixing using chemistry-informed transformer attention and slot attention clustering.

NIMIS is a structure-agnostic deep learning framework that separates complex 1H-NMR mixture spectra into individual molecular components (one peak list per component) without requiring prior structural information.

## Repository structure

```text
NIMIS/
├── src/                          model code
│   ├── __init__.py
│   └── nimis_model.py            NMRPeakSlotAttn (ScalarEmbedding, RatioAttention, Transformer, SlotAttention),
│                                 slot_hungarian_loss, entropy_regularization
│
├── scripts/                      training, collation and dataset processing
│   ├── train.py                  training entry point (DDP-capable, W&B logging, noise sweep on the test set)
│   ├── collate.py                collate128 (batching, integral noise, con / con_J matrices) and build_con_J_hard_overlap
│   ├── dataset_nimis_parquet.py  NimisMixtureParquetDataset and split_dataset (random / molecule_disjoint splits)
│   └── dataset_analysis.py       concentration-bucket statistics of a mixtures parquet
│
├── helpers/                      shared utilities
│   └── violin_parser.py          assignment-log parser (parse_assignment_file) and plotting:
│                                 PR curve, confidence violins, accuracy-by-mixture-size (+ 2x2 grid versions)
│
├── eval/                         evaluation, baselines and ablations
│   ├── baselines.py              non-learned baselines: J-coupling-only, position-only, random floor
│   ├── eval_checkpoint.py        re-evaluate a trained checkpoint on another test parquet
│   ├── stratified_test_conj.py   conJ benefit stratified by how badly peaks overlap in shift (from assignment logs)
│   ├── concentration_accuracy.py accuracy by true component concentration (1-5% ... >20%) and mixture size
│   ├── nimis_score_threshold.py  operating threshold on the NIMIS score (Youden's J, precision-recall)
│   ├── attention_match_quality.py        does attention separate genuine from spurious J-matches as tolerance loosens
│   ├── j_coupling_dropout_ablation.py    zero out a fraction of con_J matches at inference (no retraining)
│   ├── peak_matching_tolerance_ablation.py  sweep the con_J match tolerance at inference (no retraining)
│   └── plot_case_studies.py      best / worst mixture case-study figures (needs rdkit)
│
├── gui/
│   └── nimis_gui.py              Streamlit app for running the released model
│
├── weights/
│   └── nimis_final_weights.pt    final checkpoint (run 16A_A, best); loaded by the GUI
│
├── data/                         (not tracked) mixtures parquet files
├── outputs/                      (not tracked) checkpoints, assignment logs, plots, csvs written by train/eval
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/annamarijasumrova/NIMIS.git
cd NIMIS
pip install -r requirements.txt
```

All commands below are run from the repository root. Scripts are packages, so launch them with `python -m`.

## GUI

```bash
streamlit run gui/nimis_gui.py
```

Input modes: SI-style 1H-NMR strings, or manual peak entry.

Example SI-style input:

```
7.26 (d, J = 8.1 Hz, 2H), 7.12–7.05 (m, 3H), 2.31 (s, 3H)
```

Input is a peak list for a 1H-NMR mixture spectrum. Output is the same peak list annotated with an arbitrary component number per peak and a NIMIS confidence score from 0 to 1 (closer to 1 = more confident).

## Dataset

The mixture datasets (train, validation and test parquet files) are hosted on Zenodo: <ZENODO_LINK>. Download them into `data/`.

They are built from NMRexp experimental spectra:

Wang, JJ., Jin, Y., Zhi, CY. et al. NMRexp: A database of 3.3 million experimental NMR spectra. Sci Data 12, 1954 (2025). https://doi.org/10.1038/s41597-025-06245-5

Training and evaluation read these prebuilt mixture parquet files. Each row is one peak-aligned mixture with columns including `n_components`, `mix_peak_positions`, `mix_integrals`, `target`, `j_vals`, `j_mask`, `smiles`, `scales`, `NMR_solvent`, `NMR_frequency` (see `scripts/dataset_nimis_parquet.py`).

## Training

```bash
python -m scripts.train --run-id 1 \
    --train-dataset data/<train>.parquet \
    --val-dataset   data/<val>.parquet \
    --test-dataset  data/<test>.parquet
```

Outputs go to `outputs/` (override with `--output-dir`): `NMRexp_conJ_Slot_attn_<run-id>[_best].pt`, `assignments_nmrexp_{val,test}_slot_conj_<run-id>.txt`, loss curve and `test_run_id_<run-id>/` plots. Metrics are logged to Weights & Biases (`--wandb-mode disabled` to turn off).

Ablation flags: `--vanilla-attn`, `--drop-con`, `--drop-conj`, `--per-head-bias`. Other defaults: `--K 5`, `--integral-noise-std 0.10`.

## Evaluation

```bash
python -m eval.baselines --test-dataset data/<test>.parquet           # non-learned baselines
python -m eval.eval_checkpoint --checkpoint outputs/<ckpt>.pt --test-parquet data/<test>.parquet
python -m eval.stratified_test_conj
python -m eval.concentration_accuracy
python -m eval.nimis_score_threshold
python -m eval.attention_match_quality
python -m eval.j_coupling_dropout_ablation
python -m eval.peak_matching_tolerance_ablation
python -m eval.plot_case_studies
```

Use the same `--split-mode`, `--split-seed` and `--integral-noise-std` as the training run you compare against. Several eval scripts have run ids and file paths set near the top of `main()` (e.g. `outputs/assignments_nmrexp_test_slot_conj_8.txt`); edit these to point at your runs. Scripts with a built-in check run it via `--self-test` or `--selfcheck`.

## License

version 1.0

Copyright (c) 2026 Anna Marija Sumrova

distributed under MIT license
