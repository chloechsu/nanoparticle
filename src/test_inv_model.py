import argparse
import csv
import glob
import os

from load_sim_and_gen_data import TestDataset
from train_inv_model import evaluate

from alexnet import AlexNet1D
from resnet import RESNET_NAME_TO_MODEL_MAP
from attention import RESNATT_NAME_TO_MODEL_MAP
from fc import OneLayerFC, TwoLayerFC, ThreeLayerFC


MODEL_NAME_TO_CLASS_MAP = {
    'alexnet': AlexNet1D,
    'onelayerfc': OneLayerFC,
    'twolayerfc': TwoLayerFC,
    'threelayerfc': ThreeLayerFC,
}
MODEL_NAME_TO_CLASS_MAP.update(RESNET_NAME_TO_MODEL_MAP)
MODEL_NAME_TO_CLASS_MAP.update(RESNATT_NAME_TO_MODEL_MAP)


def main():
    parser = argparse.ArgumentParser(description='Training config.')
    parser.add_argument('--model_path', type=str, default='model/')
    args = parser.parse_args()

    # Store test sets for faster reuse.
    testsets = dict()
    for m in ['Au', 'SiN', 'SiO2', 'all']:
        for fe in [True, False]:
            testsets[(m, fe)] = TestDataset(m, feature_engineering=fe)

    for model_path in glob.glob(args.model_path+'*.pth'):

        metrics_path = model_path[:-4] + '_test_metrics.csv'
        if os.path.isfile(metrics_path):
            continue
        if 'Adam' not in model_path:
            continue

        material = 'all'
        for m in ['Au', 'SiN', 'SiO2']:
            if m in model_path:
                material = m
        fe = ('nofeature' not in model_path)
        test_set = testsets[(material, fe)]
        for m in MODEL_NAME_TO_CLASS_MAP.keys():
            if m in model_path:
                model_name = m
        try:
            model_cls = MODEL_NAME_TO_CLASS_MAP[model_name.lower()]
        except:
            print('Unrecognized model name.')
            return
        model = model_cls(n_logits=test_set.n_logits())

        evaluate(model_path, test_set, model_cls, metrics_path)


if __name__ == "__main__":
    main()
