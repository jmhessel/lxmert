# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch

import param

def parse_args():
    parser = argparse.ArgumentParser()

    # Data Params. These are all required.
    parser.add_argument("--train_json", default=None,
                        'comma seperated training jsons '
                        'e.g., "train1.json,train2.json" or "train.json".')
    parser.add_argument("--valid_json", default=None,
                        'comma seperated validation jsons.')
    parser.add_argument("--test_json", default=None,
                        'comma seperated testing jsons.')
    parser.add_argument("--ans2label", default=None,
                        'json dictionary mapping from strings to ints. '
                        'the strings are the names of the classes, and the '
                        'ints are their indices.')
    parser.add_argument("--image_feat_tsvs", default=None,
                        'comma seperated tsv for files containing extracted '
                        'image features. '
                        'e.g., "data/feats1.tsv,data/feats2.tsv" or "vg_gqa_obj36.tsv"')

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    # set to gqa defaults
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output_dir', type=str, default='classifier_outputs')
    parser.add_argument('--output_file', type=str, default='predictions.json')
    
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

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

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = param.get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
