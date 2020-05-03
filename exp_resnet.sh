#!/bin/bash

python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --material=all --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --material=Au --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --material=SiN --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --material=SiO2 --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=5e-4 --material=all --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=5e-4 --material=Au --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=5e-4 --material=SiN --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=5e-4 --material=SiO2 --n_epochs=20;
python src/train_inv_model.py --model_name=resnet34 --lr=5e-4 --material=all --n_epochs=20;
python src/train_inv_model.py --model_name=resnet34 --lr=5e-4 --material=Au --n_epochs=20;
python src/train_inv_model.py --model_name=resnet34 --lr=5e-4 --material=SiN --n_epochs=20;
python src/train_inv_model.py --model_name=resnet34 --lr=5e-4 --material=SiO2 --n_epochs=20;
python src/train_inv_model.py --model_name=resnet50 --lr=5e-4 --material=all --n_epochs=20;
python src/train_inv_model.py --model_name=resnet50 --lr=5e-4 --material=Au --n_epochs=20;
python src/train_inv_model.py --model_name=resnet50 --lr=5e-4 --material=SiN --n_epochs=20;
python src/train_inv_model.py --model_name=resnet50 --lr=5e-4 --material=SiO2 --n_epochs=20;
