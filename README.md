# Automated Analysis of Nanoparticle Emissivity Spectra Using Machine Learning

 A crucial step in nanoparticle synthesis is to verify whether the synthesized particles are of the desired shapes and sizes, since nanoparticles' morphology largely determine their function. Currently, such verification depends on complex and time-consuming analytical measurements.
     
To simplify the measurement procedure, we show that machine learning models can effectively predict nanoparticles' shapes and sizes based on their emissivity spectra. The emissivity spectra reflect particles' optical properties, and are simpler to measure than direct size measurements with more advanced techniques such as transition electron microscopy (TEM). As optical properties are determined by particle shapes and sizes, machine learning models can extract morphology information from emissivity spectra.
     
We compare the effectiveness of three types of models: random forests, fully connected neural networks, and convolutional neural networks, and compare different training procedures. We find that a ResNet-based convolutional neural network performs the best on shape classification, while random forests and convolutional neural networks are comparable on size regression. We also include studies on the effects of data augmentation, multi-task training, and joint training on materials.

For more details, see [`report.pdf`](report.pdf) in the repository.

## Usage

### Prerequisite
With Python 3, install requirements on pip by running `pip install -r requirements.txt`.

Put the training data from [Elzouka et al. (arXiv:2002.04223)](https://arxiv.org/abs/2002.04223) in `data/simulated_data.mat`.

### Data augmentation
In the nanoparticle directory, run

`python src/train_RF_and_gen_data.py`

with desired `n_gen_to_data_ratio` (line 36, default to 20).

This script trains a random forest model based on the simulated data, and
generates more data based on the random forest model.

The generated data in csv form will be in the `data` directory, and also
additionally numpy arrays in joblib format in the `cache` directory.

### Load generated data and original simulated data.
See example script `src/load_sim_and_gen_data.py`.

### Train neural network as inverse model
Example:

`python src/train_inv_model.py --model_name=resnet18 --lr=1e-4`

See `python src/train_inv_model.py -h` for all command line options.

During training, the script logs training progress to a tensorboard directory
under `runs/`. To bring up tensorboard visualization,

`tensorboard --logdir=runs`

The script also logs model weights and performance metrics on validation set to
the `model/`directory.

### Evaluate neural network performance on test set
Run the script `src/test_inv_model.py`.

### Random forest models
See the Jupyter notebooks `Build_RF_Models_UPDATED_051420.ipynb` and
`Paper Figures.ipynb`.
