# nanoparticle
Machine learning project for predicting nanoparticle geometry from emissivity spectra.

## Prerequisite
Download the training data and put it in `data/simulated_data.mat`.

## Generate data from random forest
In the nanoparticle directory, run

`python src/inference_RF_DTGEN_scalar_spectral_cleaned_5_feat_import.py`

with desired `n_gen_to_data_ratio` (line 52, default to 1).

This script trains a random forest model based on the simulated data, and
generates more data based on the random forest model.

The generated data will be in the `data` directory.
