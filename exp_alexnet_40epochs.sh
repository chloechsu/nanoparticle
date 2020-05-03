#!/bin/bash

python src/train_inv_model.py --model_name=alexnet --lr=1e-4 --material=Au --joint_obj --n_epochs=40;
python src/train_inv_model.py --model_name=alexnet --lr=1e-5 --material=Au --joint_obj --n_epochs=40;
python src/train_inv_model.py --model_name=alexnet --lr=5e-5 --material=Au --joint_obj --n_epochs=40;
python src/train_inv_model.py --model_name=alexnet --lr=1e-4 --material=SiN --joint_obj --n_epochs=40;
python src/train_inv_model.py --model_name=alexnet --lr=1e-5 --material=SiN --joint_obj --n_epochs=40;
python src/train_inv_model.py --model_name=alexnet --lr=5e-5 --material=SiN --joint_obj --n_epochs=40;
python src/train_inv_model.py --model_name=alexnet --lr=1e-4 --material=SiO2 --joint_obj --n_epochs=40;
python src/train_inv_model.py --model_name=alexnet --lr=1e-5 --material=SiO2 --joint_obj --n_epochs=40;
python src/train_inv_model.py --model_name=alexnet --lr=5e-5 --material=SiO2 --joint_obj --n_epochs=40;
