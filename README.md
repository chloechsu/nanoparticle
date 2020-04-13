# nanoparticle
Machine learning project for predicting nanoparticle geometry from emissivity spectra.

## Prerequisite
Download the training data and put it in `data/simulated_data.mat`.

Install Anaconda and install the conda environment by running
`conda env create -f conda.yml`

## Generate data from random forest
In the nanoparticle directory, run

`python src/train_RF_and_gen_data.py`

with desired `n_gen_to_data_ratio` (line 52, default to 20).

This script trains a random forest model based on the simulated data, and
generates more data based on the random forest model.

The generated data in csv form will be in the `data` directory.

## Load generated data and original simulated data.
See example script `src/load_sim_and_gen_data.py`.

## Train neural network as inverse model
Example:

`python src/train_inv_model.py --model_name=alexnet --lr=1e-4`

See `python src/train_inv_model.py -h` for all command line options.

During training, the script logs training progress to a tensorboard directory
under `runs/`. To bring up tensorboard visualization,

`tensorboard --logdir=runs`
