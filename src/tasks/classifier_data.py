# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from finetune_param import args
from utils import load_obj_tsv

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
        "sent": "What is on the white wall?"
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
            self.data.extend(json.load(split))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['instance_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open(args.ans2label))
        self.label2ans = {v: k for k, v in self.ans2label}

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

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.

        # This could be more memory efficient. For example, the buffer could
        # load only image data that has a corresponding datapoint, rather
        # than the other way around.
        img_data = []

        for path in args.image_feat_tsv.split(','):
            img_data.extend(classifier_buffer_loader.load_data(path, -1))
        
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

        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return instance_id, feats, boxes, sent, target
        else:
            return instance_id, feats, boxes, sent


class ClassifierEvaluator:
    def __init__(self, dataset: ClassifierDataset):
        self.dataset = dataset

    def evaluate(self, instance_id2ans: dict):
        score = 0.
        for instance_id, ans in instance_id2ans.items():
            datum = self.dataset.id2datum[instance_id]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(instance_id2ans)

    def dump_result(self, instance_id2ans: dict, path):
        """
        Dump the result to a prediction json of the following form:
            results = [result]
            result = {
                "questionId": str,
                "prediction": str
            }
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


