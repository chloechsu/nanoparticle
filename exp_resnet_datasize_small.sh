#!/bin/bash

python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=1.0 --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.5 --n_epochs=40;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.25 --n_epochs=80;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.1 --n_epochs=100;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --exclude_gen_data --n_epochs=100;

python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=1.0 --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.5 --n_epochs=40;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.25 --n_epochs=80;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.1 --n_epochs=100;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --exclude_gen_data --n_epochs=100;
