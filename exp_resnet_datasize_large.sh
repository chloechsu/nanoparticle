#!/bin/bash

python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=1.0 --n_epochs=4;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.6 --n_epochs=7;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.2 --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=1.0 --n_epochs=4;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.6 --n_epochs=7;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.2 --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=1.0 --n_epochs=4 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.6 --n_epochs=7 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=Au --gen_data_fraction=0.2 --n_epochs=20 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=1.0 --n_epochs=4 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.6 --n_epochs=7 --joint_obj;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=all --gen_data_fraction=0.2 --n_epochs=20 --joint_obj;
