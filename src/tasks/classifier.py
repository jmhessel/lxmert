# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from finetune_param import args
from tasks.classifier_model import ClassifierModel
from tasks.classifier_data import ClassifierDataset, ClassifierTorchDataset, ClassifierEvaluator


DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = ClassifierDataset(splits)
    tset = ClassifierTorchDataset(dset)
    evaluator = ClassifierEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class Classifier:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.model = ClassifierModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
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

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            instance_id2pred = {}
            for i, (instance_ids, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else:
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for instance_id, l in zip(instance_ids, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    instance_id2pred[instance_id] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch,
                                                     evaluator.evaluate(instance_id2pred) * 100.)

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
            instance_ids, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for instance_id, l in zip(instance_ids, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    instance_id2pred[instance_id] = ans
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
        for i, (instance_id2pred, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for instance_id, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                instance_id2pred[instance_id] = ans
        return evaluator.evaluate(instance_id2pred)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    classifier = Classifier()

    # Load Model
    if args.load is not None:
        classifier.load(args.load)

    if not args.load and not args.load_lxmert:
        raise NotImplementedError(
            'Error: you appear to be loading *no pretrained weights*. '
            'You want to be loading either the lxmert weights for finetuning '
            'or the finetuned weights for testing.')
        quit()

    if args.train is not None:
        print('Splits in Train data:', classifier.train_tuple.dataset.splits)
        if classifier.valid_tuple is not None:
            print('Splits in Valid data:', classifier.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (classifier.oracle_score(classifier.valid_tuple) * 100))
        else:
            print('Warning: we are not using a validation set! this is discouraged.')
        classifier.train(classifier.train_tuple, classifier.valid_tuple)
        
    if args.test is not None:
        print('Testing!')
        args.fast = args.tiny = False       # Always loading all data in test
        classifier.predict(
            get_tuple(args.test, bs=args.batch_size,
                      shuffle=False, drop_last=False),
            dump=os.path.join(args.output_dir, args.output_file)
        )
        


