#!/bin/bash

python src/train_inv_model.py --model_name=resnatt18 --lr=1e-4 --optimizer_name=Adam --material=Au --n_epochs=40;
python src/train_inv_model.py --model_name=resnatt50 --lr=1e-4 --optimizer_name=Adam --material=Au --n_epochs=40;
python src/train_inv_model.py --model_name=resnatt18 --lr=1e-3 --optimizer_name=SGD --material=Au --n_epochs=40;
python src/train_inv_model.py --model_name=resnatt50 --lr=1e-3 --optimizer_name=SGD --material=Au --n_epochs=40;
python src/train_inv_model.py --model_name=resnatt18 --lr=1e-4 --optimizer_name=Adam --material=Au --n_epochs=40 --joint_obj;
python src/train_inv_model.py --model_name=resnatt50 --lr=1e-4 --optimizer_name=Adam --material=Au --n_epochs=40 --joint_obj;
python src/train_inv_model.py --model_name=resnatt18 --lr=1e-4 --optimizer_name=Adam --material=all --n_epochs=40;
python src/train_inv_model.py --model_name=resnatt18 --lr=1e-4 --optimizer_name=Adam --material=all --n_epochs=40 --joint_obj;
python src/train_inv_model.py --model_name=resnatt50 --lr=1e-4 --optimizer_name=Adam --material=all --n_epochs=40;
python src/train_inv_model.py --model_name=resnatt50 --lr=1e-4 --optimizer_name=Adam --material=all --n_epochs=40 --joint_obj;
python src/train_inv_model.py --model_name=resnatt18 --lr=1e-3 --optimizer_name=SGD --material=all --n_epochs=40;
python src/train_inv_model.py --model_name=resnatt50 --lr=1e-3 --optimizer_name=SGD --material=all --n_epochs=40;
