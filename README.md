# NIMIS

Neural framework for 1H-NMR mixture demixing using chemistry-informed transformer attention and slot attention clustering.

NIMIS is a structure-agnostic deep learning framework designed to separate complex 1H-NMR mixture spectra into individual molecular components (individual peak list per component) without requiring prior structural information.

-------------------------------------------------------------------------------
## REPOSITORY STRUCTURE

```text
NIMIS/
├── model/
│   ├── __init__.py
│   ├── nimis_model.py
│   └── nimis_weights.pt
│
├── gui/
│   └── nimis_gui.py
│
├── data/
│   ├── parquet_to_h5.py
│   └── dataset_nmrexp.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```
-------------------------------------------------------------------------------

## INSTALLATION GUIDE

Clone the repository:

```bash
git clone https://github.com/annamarijasumrova/NIMIS.git
cd NIMIS
```

Install required packages:

```bash
pip install -r requirements.txt
```
-------------------------------------------------------------------------------

## LAUNCH GUI

From repository root:

```bash
streamlit run gui/nimis_gui.py
```

The GUI supports two input modes:
1. SI-style 1H-NMR strings
2. Manual peak entry

-------------------------------------------------------------------------------

## EXAMPLE SI-STYLE INPUT

7.26 (d, J = 8.1 Hz, 2H), 7.12–7.05 (m, 3H), 2.31 (s, 3H)

-------------------------------------------------------------------------------

## OUTPUT

Input:
Peak list corresponding to a 1H-NMR mixture spectrum.

Output:
The same peak list annotated with predicted mixture component assignments. Each peak is assigned an arbitrary component number together with a NIMIS confidence score ranging from 0 to 1, where values closer to 1 indicate higher assignment confidence and values closer to 0 indicate lower confidence in the predicted assignment.

-------------------------------------------------------------------------------

## DATASET PREPARATION

The original NMRexp parquet file is not included in this repository and can be downloaded at the source publication:

Wang, JJ., Jin, Y., Zhi, CY. et al. NMRexp: A database of 3.3 million experimental NMR spectra. Sci Data 12, 1954 (2025). https://doi.org/10.1038/s41597-025-06245-5

https://zenodo.org/records/17296666

To prepare the training dataset:

1. Download the NMRexp parquet file

2. Open: 
```bash
./parse/parquet_to_h5.py
```

and specify path to downloaded parquet file by changing

```python
file_path = Path("PATH_TO_NMREXP_PARQUET")
```
3. Run:
```bash
python data/parquet_to_h5.py
```
This generates the HDF5 dataset used for model training.

-------------------------------------------------------------------------------

## LICENSE

version 1.0

Copyright (c) 2026 Anna Marija Sumrova

distributed under MIT license


