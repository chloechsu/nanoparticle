#!/bin/bash

python src/train_inv_model.py --model_name=resnet18 --lr=1e-3 --optimizer_name=SGD --material=Au --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-2 --optimizer_name=SGD --material=Au --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=SGD --material=Au --n_epochs=20;
python src/train_inv_model.py --model_name=resnet34 --lr=1e-3 --optimizer_name=SGD --material=Au --n_epochs=20;
python src/train_inv_model.py --model_name=resnet50 --lr=1e-3 --optimizer_name=SGD --material=Au --n_epochs=20;
