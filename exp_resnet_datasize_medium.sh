#!/bin/bash

python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.5 --n_epochs=8;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.4 --n_epochs=10;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.3 --n_epochs=14;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.2 --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.1 --n_epochs=40;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.05 --n_epochs=80;

python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.5 --n_epochs=8;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.4 --n_epochs=10;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.3 --n_epochs=14;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.2 --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.1 --n_epochs=40;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.05 --n_epochs=80;

python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.5 --n_epochs=8 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.4 --n_epochs=10 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.3 --n_epochs=14 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.2 --n_epochs=20 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.1 --n_epochs=40 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.05 --n_epochs=80 --joint_obj;
