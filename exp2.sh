#!/bin/bash

python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=all --n_epochs=40;
python src/train_inv_model.py --model_name=alexnet --lr=1e-3 --material=all;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=all --joint_obj;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=Au --joint_obj;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=SiN --joint_obj;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=SiO2 --joint_obj;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=all --joint_obj --multistage;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=Au --joint_obj --multistage;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=SiN --joint_obj --multistage;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=SiO2 --joint_obj --multistage;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=all --multistage;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=Au --multistage;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=SiN --multistage;
python src/train_inv_model.py --model_name=alexnet --lr=5e-4 --material=SiO2 --multistage;
