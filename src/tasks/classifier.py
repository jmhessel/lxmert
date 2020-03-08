# coding=utf-8
# Copyleft 2019 project LXRT.

import sys
import os
sys.path.append(os.getcwd() + '/src/')
import json

import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from finetune_param import args
from classifier_model import ClassifierModel
from classifier_data import ClassifierDataset, ClassifierTorchDataset, ClassifierEvaluator


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

        n_answers = len(json.load(open(args.ans2label)))

        print(args.model_type)
        self.model = ClassifierModel(n_answers, model_type=args.model_type)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer, only if training
        if args.train_json != "-1":
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

        self.output = args.output_dir
        self.best_name = None
        
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            instance_id2pred = {}
            for i, (instance_ids, feats, boxes, sent, logit_in, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                # for gradient checking
                # feats.requires_grad = True
                
                feats, boxes, logit_in, target = feats.cuda(), boxes.cuda(), logit_in.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent) + logit_in

                # for gradient checks --- this errors for concat model (no dependence)
                # but gives a gradient for the full model
                # # text feature gradient
                # dldf = torch.autograd.grad(
                #     torch.sum(logit),
                #     self.model.lxrt_encoder.model.bert.tmp_cur_embedding_output,
                #     create_graph=True)[0]
                # # image feature double gradient
                # print(torch.autograd.grad(torch.sum(dldf), feats)[0])

                if target.dim() == 1: #expand targets in binary mode
                    assert logit.size(1) == 1
                    target = target.unsqueeze(1)
                assert logit.dim() == target.dim() == 2

                if logit.size(1) > 1: # multiclass, mce loss
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else: # binary
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                if logit.size()[1] > 1:
                    score, label = logit.max(1)
                else:
                    score = logit.flatten()
                    label = (logit > 0).float().flatten()
                
                for instance_id, l, scores in zip(instance_ids, label.cpu().numpy(), logit.detach().cpu().numpy()):
                    ans = dset.label2ans[l]
                    instance_id2pred[instance_id] = {'answer': ans, 'label': l, 'scores': scores}

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch,
                                                     evaluator.evaluate(instance_id2pred) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.best_name = 'epoch_{}_valscore_{:.5f}'.format(epoch, valid_score * 100.)
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
            instance_ids, feats, boxes, sent, logit_in = datum_tuple[:5]   # avoid handling target
            with torch.no_grad():
                feats, boxes, logit_in = feats.cuda(), boxes.cuda(), logit_in.cuda()
                
                logit = self.model(feats, boxes, sent) + logit_in
                if logit.size()[1] > 1:
                    score, label = logit.max(1)
                else:
                    score = logit.flatten()
                    label = (logit > 0).float().flatten()
                for instance_id, l, scores in zip(instance_ids, label.cpu().numpy(), logit.detach().cpu().numpy()):
                    ans = dset.label2ans[l]
                    instance_id2pred[instance_id] = {'answer': ans, 'label': l, 'scores': scores}
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
        for i, (instance_ids, feats, boxes, sent, logit_in, target) in enumerate(loader):
            if len(target.size()) > 1 and target.size()[1] > 1:
                _, label = target.max(1)
            else:
                label = torch.flatten(target)
            for instance_id, l in zip(instance_ids, label.cpu().numpy()):
                ans = dset.label2ans[l]
                instance_id2pred[instance_id] = {'answer': ans, 'label': l}
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

    if not args.load_finetune and not args.load_lxmert:
        raise NotImplementedError(
            'Error: you appear to be loading *no pretrained weights*. '
            'You want to be loading either the lxmert weights for finetuning '
            'or the finetuned weights for testing.')
        quit()

    # Build Classifier
    classifier = Classifier()

    # Load Model
    if args.load_finetune is not None:
        classifier.load(args.load_finetune)

    trained_this_run = False
    if args.train_json != '-1':
        trained_this_run = True
        print('Splits in Train data:', classifier.train_tuple.dataset.splits)
        if classifier.valid_tuple is not None:
            print('Splits in Valid data:', classifier.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (classifier.oracle_score(classifier.valid_tuple) * 100))
        else:
            print('Warning: we are not using a validation set! this is discouraged.')
        classifier.train(classifier.train_tuple, classifier.valid_tuple)
        
    if args.test_json != '-1':
        if trained_this_run:
            print('loading from {}!'.format(classifier.best_name))
            classifier.load(os.path.join(classifier.output, classifier.best_name))
            args.load_finetune = classifier.best_name

        prediction_name_prefix = args.load_finetune.split('/')[-1]

        print('Testing!')
        args.fast = args.tiny = False       # Always loading all data in test
        classifier.predict(
            get_tuple(args.test_json, bs=args.batch_size,
                      shuffle=False, drop_last=False),
            dump=os.path.join(args.output_dir, prediction_name_prefix + '_test_predictions.json')
        )
