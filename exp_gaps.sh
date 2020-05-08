#!/bin/bash

python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=SiN --n_epochs=20;
python src/train_inv_model.py --model_name=resnet18 --lr=1e-4 --optimizer_name=Adam --material=SiO2 --n_epochs=20;
