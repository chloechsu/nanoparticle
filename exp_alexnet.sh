#!/bin/bash

python src/train_inv_model.py --model_name=twolayerfc --lr=1e-4;
python src/train_inv_model.py --model_name=twolayerfc --lr=5e-4;
python src/train_inv_model.py --model_name=twolayerfc --lr=5e-4 --material=SiN;
python src/train_inv_model.py --model_name=twolayerfc --lr=5e-4 --material=SiO2;
python src/train_inv_model.py --model_name=threelayerfc --lr=1e-4;
python src/train_inv_model.py --model_name=threelayerfc --lr=5e-4;
python src/train_inv_model.py --model_name=alexnet --lr=1e-4;
python src/train_inv_model.py --model_name=alexnet --lr=1e-4 --material=SiN;
python src/train_inv_model.py --model_name=alexnet --lr=1e-4 --material=SiO2;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=SiN;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=SiO2;
