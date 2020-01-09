# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Params. These are all required.
    parser.add_argument("train_json",
                        help='comma seperated training jsons '
                        'e.g., "train1.json,train2.json" or "train.json". '
                        'For None, use -1.')
    parser.add_argument("valid_json",
                        help='comma seperated validation jsons. For none, use -1.')
    parser.add_argument("test_json",
                        help='comma seperated testing jsons. For none, use -1.')
    parser.add_argument("image_feat_tsv", default=None,
                        help='comma seperated tsv for files containing extracted '
                        'image features. '
                        'e.g., "data/feats1.tsv,data/feats2.tsv" or "vg_gqa_obj36.tsv"')

    parser.add_argument('output_dir', type=str, default='output')
    
    # classifier arguments
    parser.add_argument("--ans2label", default=None,
                        help='json dictionary mapping from strings to ints. '
                        'the strings are the names of the classes, and the '
                        'ints are their indices.')

    
    # Training Hyper-parameters
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--optimize_metric', default='acc',
                        help='which metric to optimize over the validation set?')
    
    # set to gqa defaults
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=True, const=True)

    # Model Loading
    parser.add_argument('--load_finetune', type=str, default=None,
                        help='Load the finetuned model for testing.')
    
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    
    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')
    parser.add_argument("--model_type", default='full', type=str,
                        help='What LXMERT model type should be used?')

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    # Parse the arguments.
    args = parser.parse_args()
    
    # we need to set these, but we dont want to make them mutable
    args.from_scratch = False
    if args.load_lxmert and '_LXRT.pth' in args.load_lxmert:
        args.load_lxmert = args.load_lxmert.replace('_LXRT.pth', '')
    if args.load_finetune and '.pth' in args.load_finetune:
        args.load_finetune = args.load_finetune.replace('.pth', '')
    
    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args

args = parse_args()
