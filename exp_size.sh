#!/bin/bash

python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --n_epochs=20 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --n_epochs=20 --size_only;
python src/train_inv_model.py --model_name=alexnet --lr=1e-4 --optimizer_name=Adam --material=all --n_epochs=20 --size_only;
python src/train_inv_model.py --model_name=onelayerfc --lr=1e-4 --optimizer_name=Adam --material=all --n_epochs=20 --size_only;
python src/train_inv_model.py --model_name=twolayerfc --lr=1e-4 --optimizer_name=Adam --material=all --n_epochs=20 --size_only;
python src/train_inv_model.py --model_name=threelayerfc --lr=1e-4 --optimizer_name=Adam --material=all --n_epochs=20 --size_only;
