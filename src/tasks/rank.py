# coding=utf-8
# Copyleft 2019 project LXRT.
import sys
import os
sys.path.append(os.getcwd() + '/src/')
import json
import collections
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from finetune_param import args
from tasks.rank_model import RankModel
from tasks.rank_data import RankDataset, RankTorchDataset, RankEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = RankDataset(splits)
    tset = RankTorchDataset(dset)
    evaluator = RankEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class Rank:
    def __init__(self):
        if args.train_json != "-1":
            self.train_tuple = get_tuple(
                args.train_json, bs=args.batch_size, shuffle=True, drop_last=True
            )
        else:
            self.train_tuple = None
        
        if args.valid_json != "-1":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid_json, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.model = RankModel(model_type=args.model_type)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        
        # Losses and optimizer, only if training
        if args.train_json != "-1":
            self.rank_loss = nn.BCEWithLogitsLoss()
            if 'bert' in args.optim:
                batch_per_epoch = len(self.train_tuple.loader)
                t_total = int(batch_per_epoch * args.epochs)
                print("Total Iters: %d" % t_total)
                from lxrt.optimization import BertAdam
                self.optim = BertAdam(list(self.model.parameters()),
                                      lr=args.lr,
                                      warmup=0.1,
                                      t_total=t_total)
            else:
                self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output_dir
        self.best_name = None
        
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            instance_id2pred = {}
            for i, (instance_ids, f0, b0, f1, b1, sent0, sent1, logit_in, label) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()
                f0, b0 = f0.cuda(), b0.cuda()
                f1, b1 = f1.cuda(), b1.cuda()

                label = label.cuda()
                logit_in = logit_in.cuda()

                score0 = self.model(f0, b0, sent0)
                score1 = self.model(f1, b1, sent1)
                logit = score0 - score1 + logit_in

                if label.dim() == 1: #expand targets in binary mode
                    assert logit.size(1) == 1
                    label = label.unsqueeze(1)

                loss = self.rank_loss(logit, label)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                
                predict = (logit > 0).float()
                
                for instance_id, l, score, c_score_0, c_score_1 in zip(instance_ids, predict.cpu().numpy(), logit.detach().cpu().numpy(), score0.detach().cpu().numpy(), score1.detach().cpu().numpy()):
                    instance_id2pred[instance_id] = {'label': float(l), 'scores': score, 'score0':float(c_score_0[0]), 'score1':float(c_score_1[0])}

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(instance_id2pred) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.best_name = 'epoch_{}_valscore_{:.5f}_argshash_{}'.format(epoch, valid_score * 100., args.args_hash)
                    self.save(self.best_name)

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        instance_id2pred = {}
        for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
            instance_ids, f0, b0, f1, b1, sent0, sent1, logit_in = datum_tuple[:-1]
            with torch.no_grad():
                f0, b0 = f0.cuda(), b0.cuda()
                f1, b1 = f1.cuda(), b1.cuda()

                score0 = self.model(f0, b0, sent0)
                score1 = self.model(f1, b1, sent1)

                logit_in = logit_in.cuda()
                
                logit = score0 - score1 + logit_in
                predict = (logit > 0).float()

                for instance_id, l, score, c_score_0, c_score_1 in zip(instance_ids, predict.cpu().numpy(), logit.detach().cpu().numpy(), score0.cpu().numpy(), score1.cpu().numpy()):
                    instance_id2pred[instance_id] = {'label': float(l), 'scores': score, 'score0':float(c_score_0[0]), 'score1':float(c_score_1[0])}
                    
                                
        if dump is not None:
            evaluator.dump_result(instance_id2pred, dump)
        
        return instance_id2pred

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        instance_id2pred = self.predict(eval_tuple, dump)
        return evaluator.evaluate(instance_id2pred)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        instance_id2pred = {}
        for i, (instance_ids, f0, b0, f1, b1, sent0, sent1, logit_in, label) in enumerate(loader):
            for instance_id, l in zip(instance_ids, label.cpu().numpy()):
                instance_id2pred[instance_id] = {'label': float(l)}
        return evaluator.evaluate(instance_id2pred)
    
    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":

    args_as_tuple = tuple(sorted({k: v for k, v in vars(args).items() if v is not None}.items()))
    args_hash = hash(args_as_tuple)
    args.args_hash = args_hash

    if not args.load_finetune and not args.load_lxmert:
        raise NotImplementedError(
            'Error: you appear to be loading *no pretrained weights*. '
            'You want to be loading either the lxmert weights for finetuning '
            'or the finetuned weights for testing.')
        quit()
    
    # Build ranker
    rank = Rank()

    # Load Model
    if args.load_finetune is not None:
        rank.load(args.load_finetune)

    trained_this_run = False
    if args.train_json != '-1':
        trained_this_run = True
        print('Splits in Train data:', rank.train_tuple.dataset.splits)
        if rank.valid_tuple is not None:
            print('Splits in Valid data:', rank.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (rank.oracle_score(rank.valid_tuple) * 100))
        else:
            print('Warning: we are not using a validation set! this is discouraged.')
        rank.train(rank.train_tuple, rank.valid_tuple)
        
    if args.test_json != '-1':
        if trained_this_run:
            print('loading from {}!'.format(rank.best_name))
            rank.load(os.path.join(rank.output, rank.best_name))
            args.load_finetune = rank.best_name

        prediction_name_prefix = args.load_finetune.split('/')[-1]
            
        print('Testing!')
        args.fast = args.tiny = False       # Always loading all data in test
        rank.predict(
            get_tuple(args.test_json, bs=args.batch_size,
                      shuffle=False, drop_last=False),
            dump=os.path.join(args.output_dir, prediction_name_prefix + '_test_predictions.json')
        )
