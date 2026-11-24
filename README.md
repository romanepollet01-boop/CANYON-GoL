
# CANYON-GoL
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24503301.svg)](https://doi.org/10.5281/zenodo.24503301)

**CANYON-GoL: a deep-learning reconstruction of the Gulf of Lion carbonate system**

**Authors:** R. Pollet¹, R. Sauzède², L. Delaigue¹, E. Labourdette¹, L. Coppola¹˒³

1. Sorbonne Université, CNRS, Laboratoire d'Océanographie de Villefranche (LOV), Villefranche-sur-Mer, France
2. Sorbonne Université, CNRS, Institut de la Mer de Villefranche (IMEV), Villefranche, France
3. Sorbonne Université, CNRS, Observatoire des Sciences de l'Univers (OSU) Station Marines (STAMAR), Paris, France

## Overview
CANYON-GoL is a deep-learning model developed to reconstruct vertical profiles of carbonate system variables in the Gulf of Lion (Northwestern Mediterranean Sea).

The model predicts three carbonate system parameters:

- Total alkalinity (AT)
- Dissolved inorganic carbon (DIC)
- pH at total scale

This repository provides the pretrained neural network models, associated scalers, and Python scripts required to generate predictions.

A detailed description of the model development, training, and validation is provided in the associated scientific manuscript, submitted to *JGR: Oceans*.

## Repository contents
The repository is organized as follows:

```text
CANYON-GoL/
├── AT/
│   ├── cgol_at_models.pt
│   ├── input_scaler_at.pt
│   ├── output_scaler_at.pt
│   └── CANYON_GoL_AT.py
│
├── DIC/
│   ├── cgol_dic_models.pt
│   ├── input_scaler_dic.pt
│   ├── output_scaler_dic.pt
│   └── CANYON_GoL_DIC.py
│
├── pH/
│   ├── cgol_ph_models.pt
│   ├── input_scaler_ph.pt
│   ├── output_scaler_ph.pt
│   └── CANYON_GoL_pH.py
│
├── gebco_2025_n44.0_s41.0_w2.5_e7.0.nc
├── coastline_points_gol.csv
└── README.md
```

### Model files
Each directory (`AT/`, `DIC/`, `pH/`) contains:

- **Model weights (`.pt`):** trained neural network parameters for the corresponding carbonate system variable.
- **Input scaler (`.pt`):** parameters used to normalize the input variables.
- **Output scaler (`.pt`):** parameters used to transform model predictions back to their original scale.
- **Python script (`.py`):** script to load and run the corresponding CANYON-GoL model.

### Additional files
- `gebco_2025_n44.0_s41.0_w2.5_e7.0.nc`: GEBCO 2025 bathymetric data covering the Gulf of Lion region.
- `coastline_points_gol.csv`: coastline points for the Gulf of Lion region.

## Usage
Each carbonate system variable can be predicted independently using its corresponding Python script:

- `AT/CANYON_GoL_AT.py` for total alkalinity.
- `DIC/CANYON_GoL_DIC.py` for dissolved inorganic carbon.
- `pH/CANYON_GoL_pH.py` for pH.

The pretrained model weights and associated input and output scalers are provided in the corresponding directories.

## Data availability
The observational dataset used to develop and train CANYON-GoL will be made publicly available through [SEANOE](https://www.seanoe.org/).

**Data repository:** SEANOE (data publication in progress).

The corresponding DOI will be added once the dataset is published.

## Reference

### Software citation

If you use CANYON-GoL in your research, please cite the corresponding software release:

Pollet, R., Sauzède, R., Delaigue, L., Labourdette, E., & Coppola, L. (2026). CANYON-GoL: a deep-learning reconstruction of the Gulf of Lion carbonate system (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24503301

### Associated scientific publication

A detailed description of the model development, training, and validation is provided in the associated scientific manuscript:

Pollet, R., Sauzède, R., Delaigue, L., Labourdette, E., and Coppola, L. CANYON-GoL: a deep-learning reconstruction of the Gulf of Lion carbonate system. Manuscript submitted to Journal of Geophysical Research: Oceans.

The complete citation and DOI of the scientific publication will be provided once available.
## License

CANYON-GoL is distributed under the terms of the GNU General Public License version 3.0 (GPL-3.0).

See the [LICENSE](LICENSE) file for the full license terms.
