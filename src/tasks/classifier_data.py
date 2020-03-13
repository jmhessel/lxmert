# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
import torch
import sklearn.preprocessing
from torch.utils.data import Dataset

from finetune_param import args
from utils import load_obj_tsv
import eval_utils

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class ClassifierDataset:
    """
    A Classifier data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "0": 1.0
        },
        "instance_id": "07333408",
        "sent": "What is on the white wall?",
        (optionally) "logit": [list of logits]
    }

    "label" is a dictionary mapping 'answers' (which may be strings representing
    the class names, or just string placeholders for ints) to class probabilities.
    """
    def __init__(self, splits: str):

        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['instance_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open(args.ans2label))
        self.label2ans = {v: k for k, v in self.ans2label.items()}

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class ClassifierBufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, path, number):
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]


classifier_buffer_loader = ClassifierBufferLoader()


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class ClassifierTorchDataset(Dataset):
    def __init__(self, dataset: ClassifierDataset):
        super().__init__()
        self.raw_dataset = dataset

        # note --- limited loading will happen even at test time.
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # This could be more memory efficient. For example, the buffer could
        # load only image data that has a corresponding datapoint, rather
        # than the other way around.
        img_data = []

        for path in args.image_feat_tsv.split(','):
            img_data.extend(classifier_buffer_loader.load_data(path, topk))
        
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        instance_id = datum['instance_id']
        sent = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        #resample if less than 36 boxes
        if len(feats) != 36:
            to_resample = 36 - len(feats)
            all_idxs = list(range(len(feats))) + list(np.random.choice(len(feats), size=to_resample))
            feats = feats[all_idxs]
            boxes = boxes[all_idxs]
            print('resampled bboxes. this should be rare.')

        
        # create logits
        if 'logit' in datum and args.use_logits:
            logit_in = torch.FloatTensor(datum['logit'])
        else:
            if self.raw_dataset.num_answers > 2: # multiclass
                logit_in = torch.zeros(self.raw_dataset.num_answers)
            else: # binary
                logit_in = torch.zeros(1)

        # Create target
        if 'label' in datum:
            label = datum['label']
            if self.raw_dataset.num_answers > 2: # multiclass mode
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    if ans in self.raw_dataset.ans2label:
                        target[self.raw_dataset.ans2label[ans]] = score
            else: # binary mode
                target = torch.zeros(1)
                assert len(label) == 1, 'binary can only have one label'
                target = float(self.raw_dataset.ans2label[list(label.keys())[0]])
                
            return instance_id, feats, boxes, sent, logit_in, target
        else:
            return instance_id, feats, boxes, sent, logit_in


class ClassifierEvaluator:
    def __init__(self, dataset: ClassifierDataset):
        self.dataset = dataset

    def convert_dict_to_hard_label(self, label_dict):
        top_ans_pred = self.convert_dict_to_hard_answer(label_dict)
        return self.dataset.ans2label[top_ans_pred]

    def convert_dict_to_hard_answer(self, label_dict):
        sorted_items = sorted(list(label_dict.items()), key=lambda x: -x[1])
        top_ans_pred = sorted_items[0][0]
        return top_ans_pred

    def evaluate(self, instance_id2ans: dict, return_full = False):
        true_labels, predicted_labels, predicted_scores = [], [], []
        for cid, pred in instance_id2ans.items():
            datum = self.dataset.id2datum[cid]
            true_labels.append(self.convert_dict_to_hard_label(datum['label']))
            predicted_labels.append(pred['label'])
            if 'scores' in pred:
                predicted_scores.append(pred['scores'])

        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        
        n_labels = self.dataset.num_answers
                
        if len(predicted_scores) == 0:
            if n_labels == 2:
                predicted_scores = predicted_labels
            else:
                predicted_scores = sklearn.preprocessing.label_binarize(
                    predicted_labels, classes=np.arange(n_labels))
        else:
            predicted_scores = np.array(predicted_scores)

        if n_labels == 2:
            res = eval_utils.get_metrics_binary(
                predicted_scores, predicted_labels, true_labels)
        else:
            res = eval_utils.get_metrics_multiclass(
                predicted_scores, predicted_labels, true_labels)

        print(res)
        
        if return_full:
            return res
        else:
            return res[args.optimize_metric]

    def dump_result(self, instance_id2ans: dict, path):
        """
        Dump the result to a prediction json of the following form:
            results = {'per_instance': [result], 'metrics': metrics, 'args':args}
            result = {
                "instance_id": str,
                "predicted_answer": str,
                "predicted_label": str,
                "answer": ground truth answer,
                "label": ground truth label,
                "predicted_scores": list of floats representing the logits for each class,
            }
            metric is a dictionary of evaluation metrics.
            
        """
        with open(path, 'w') as f:

            metrics = self.evaluate(instance_id2ans, return_full=True)
            result = []

            for k, v in metrics.items():
                metrics[k] = float(v)
                
            for cid, ans in instance_id2ans.items():
                datum = self.dataset.id2datum[cid]
                result.append({
                    'instance_id': cid,
                    'predicted_answer': str(ans['answer']),
                    'predicted_label': int(self.dataset.ans2label[ans['answer']]),
                    'predicted_scores': list([float(x) for x in ans['scores']]),
                    'label': int(self.convert_dict_to_hard_label(datum['label'])),
                    'answer': str(self.convert_dict_to_hard_answer(datum['label'])),
                    'input': datum,
                })
                
            json.dump({'result': result, 'metrics': metrics, 'args':vars(args)}, f, indent=4, sort_keys=True)


