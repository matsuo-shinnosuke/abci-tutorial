import argparse
from pathlib import Path
from utils import mkdir

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam','sgd'])
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--is_train', default=1, type=int)
    parser.add_argument('--is_test', default=1, type=int)

    parser.add_argument('--dataset_dir', default='data/', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cifar100'])
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--output_dir', default='result/debug/', type=str)

    args = parser.parse_args()
    mkdir(args.output_dir)
 
    return args
