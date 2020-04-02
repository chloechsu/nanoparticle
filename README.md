# nanoparticle
Machine learning project for predicting nanoparticle geometry from emissivity spectra.

Download the training data and put it in `data/simulated_data.mat`.

In the nanoparticle directory,

1. Run `python src/inference_RF_DTGEN_scalar_spectral_cleaned_5_feat_import.py`

2. Run `python src/InverseDesign_DT_DTGEN_scalar_spectral_cleaned_2.py`

The generated data will be in `nanoparticle/cache`.
