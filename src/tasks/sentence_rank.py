# coding=utf-8
# Copyleft 2019 project LXRT.
import sys
import os
sys.path.append(os.getcwd() + '/src/')
import json
import collections

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from finetune_param import args
from tasks.sentence_rank_model import SentenceRankModel
from tasks.sentence_rank_data import SentenceRankDataset, SentenceRankTorchDataset, SentenceRankEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = SentenceRankDataset(splits)
    tset = SentenceRankTorchDataset(dset)
    evaluator = SentenceRankEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class SentenceRank:
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

        self.model = SentenceRankModel()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        
        # Losses and optimizer, only if training
        if args.train_json != "-1":
            self.rank_loss = nn.MarginRankingLoss(margin=1.0)
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
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            instance_id2pred = {}
            for i, (instance_ids, feats, boxes, sent0, sent1, label) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()
                feats, boxes, label = feats.cuda(), boxes.cuda(), label.cuda()
                score0 = self.model(feats, boxes, sent0)
                score1 = self.model(feats, boxes, sent1)
                
                loss = self.rank_loss(score0, score1, label)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score = score0 - score1
                predict = (score0 > score1) * 2 - 1
                
                for instance_id, l in zip(instance_ids, predict.cpu().numpy()):
                    instance_id2pred[instance_id] = l

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(instance_id2pred) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

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
        for i, datum_tuple in enumerate(loader):
            instance_ids, feats, boxes, sent0, sent1 = datum_tuple[:5]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                score0 = self.model(feats, boxes, sent0)
                score1 = self.model(feats, boxes, sent1)

                score = score0 - score1
                predict = (score0 > score1) * 2 - 1
                
                for instance_id, l in zip(instance_ids, predict.cpu().numpy()):
                    instance_id2pred[instance_id] = l
                
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
        for i, (instance_ids, feats, boxes, sent0, sent1, label) in enumerate(loader):
            for instance_id, l in zip(instance_ids, label.cpu().numpy()):
                instance_id2pred[instance_id] = l
        return evaluator.evaluate(instance_id2pred)
    
    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":

    if not args.load_finetune and not args.load_lxmert:
        raise NotImplementedError(
            'Error: you appear to be loading *no pretrained weights*. '
            'You want to be loading either the lxmert weights for finetuning '
            'or the finetuned weights for testing.')
        quit()
    
    # Build ranker
    sentence_rank = SentenceRank()

    # Load Model
    if args.load_finetune is not None:
        sentence_rank.load(args.load_finetune)

    trained_this_run = False
    if args.train_json != '-1':
        trained_this_run = True
        print('Splits in Train data:', sentence_rank.train_tuple.dataset.splits)
        if sentence_rank.valid_tuple is not None:
            print('Splits in Valid data:', sentence_rank.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (sentence_rank.oracle_score(sentence_rank.valid_tuple) * 100))
        else:
            print('Warning: we are not using a validation set! this is discouraged.')
        sentence_rank.train(sentence_rank.train_tuple, sentence_rank.valid_tuple)
        
    if args.test_json != '-1':
        if trained_this_run:
            print('loading from best!')
            sentence_rank.load(os.path.join(sentence_rank.output, 'BEST'))
        print('Testing!')
        args.fast = args.tiny = False       # Always loading all data in test
        sentence_rank.predict(
            get_tuple(args.test_json, bs=args.batch_size,
                      shuffle=False, drop_last=False),
            dump=os.path.join(args.output_dir, 'test_predictions.json')
        )