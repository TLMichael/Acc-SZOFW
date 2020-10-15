import torch
import argparse
import os
import os.path as osp
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

from query_model import QueryModel
from datasets import get_dataset
from methods.zscg import ZSCG
from methods.zo_sfw import ZOSFW
from methods.acc_szofw import AccSZOFW
from methods.acc_szofwp import AccSZOFWP

NUM_FEATURES = {
    'phishing': 68,
    'a9a': 123,
    'w8a': 300,
    'covtype': 54,
}


def main(dataset, method, estimator):
    parser = argparse.ArgumentParser(description='Robust Lasso Regression by ZSCG, ZO-SFW, Acc-SZOFW and Acc-SZOFW*')
    args = parser.parse_args()

    args.dataset = dataset
    args.method = method
    args.estimator = estimator

    args.writer = True
    # args.writer = False

    args.Q = 1
    args.num_features = NUM_FEATURES[args.dataset]

    args.result_path = './results/'
    args.tensorboard_path = osp.join(args.result_path, 'tsdata', args.dataset, args.method, args.estimator)

    pprint(vars(args))

    # Starting algorithm
    train_data = get_dataset(args.dataset, train=True)
    test_data = get_dataset(args.dataset, train=False)
    model = QueryModel(args.num_features, args.Q, args.estimator)

    if 'ZSCG' in args.method:
        attacker = ZSCG(args.dataset, train_data, test_data, model)
    elif 'ZO-SFW' in args.method:
        attacker = ZOSFW(args.dataset, train_data, test_data, model)
    elif 'Acc-SZOFW*' in args.method:
        attacker = AccSZOFWP(args.dataset, train_data, test_data, model)
    elif 'Acc-SZOFW' in args.method:
        is_vr, is_xm, is_gm = True, True, False
        attacker = AccSZOFW(args.dataset, train_data, test_data, model, is_vr=is_vr, is_gm=is_gm, is_vm=is_xm)
    
    writer = SummaryWriter(args.tensorboard_path) if args.writer else None

    train_loss, test_loss = attacker.attack(writer=writer)

    print('Done')


if __name__ == '__main__':
    
    datasets = ['phishing', 'a9a', 'w8a', 'covtype']
    algorithms = [
        ('ZSCG', 'GauGE'),
        ('ZO-SFW', 'GauGE'),
        ('Acc-SZOFW', 'UniGE'),
        ('Acc-SZOFW*', 'UniGE'),
        ('Acc-SZOFW', 'CooGE'),
        ('Acc-SZOFW*', 'CooGE'),
    ]
    
    for ds in datasets:
        for meth, est in algorithms:
            main(ds, meth, est)

