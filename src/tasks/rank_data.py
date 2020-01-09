# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
from torch.utils.data import Dataset

from finetune_param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


def convert_example(ex):
    ex['img_id_0'] = ex['image0']
    ex['img_id_1'] = ex['image1']
    ex['instance_id'] = ex['pair_identifier']
    ex['label'] = ex['label'] * 2 - 1
    del ex['image0']
    del ex['image1']
    del ex['pair_identifier']
    return ex

class RankDataset:
    """
    A rank data example in the json file:
    {
      'pair_identifier': unique string identifying this pair,
      'image0': image identifier 0,
      'image1': image identifier 0,
      'sent0': text of sentence 0,
      'sent1': text of sentence 1,
      'label': 1 if (image0, sent0) better than (image1, sent1) image else 0
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')
        
        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        self.data = [convert_example(d) for d in self.data]

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['instance_id']: datum
            for datum in self.data
        }

    def __len__(self):
        return len(self.data)


class RankBufferLoader():
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

rank_buffer_loader = RankBufferLoader()
    
"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class RankTorchDataset(Dataset):
    def __init__(self, dataset: RankDataset):
        super().__init__()
        self.raw_dataset = dataset

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
            img_data.extend(rank_buffer_loader.load_data(path, topk))
        
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id_0'] in self.imgid2img:
                if datum['img_id_1'] in self.imgid2img:
                    self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        instance_id = datum['instance_id']
        sent0, sent1 = datum['sent0'], datum['sent1']
        img_id_0, img_id_1 = datum['img_id_0'], datum['img_id_1']

        all_feats, all_boxes = [], []
        # Get image info
        for img_id in [img_id_0, img_id_1]:
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
            all_feats.append(feats)
            all_boxes.append(boxes)

        f0, f1 = all_feats
        b0, b1 = all_boxes
            
        # Create target
        if 'label' in datum:
            label = datum['label']
            return instance_id, f0, b0, f1, b1, sent0, sent1, label
        else:
            return instance_id, f0, b0, f1, b1, sent0, sent1


class RankEvaluator:
    def __init__(self, dataset: RankDataset):
        self.dataset = dataset

    def evaluate(self, instance_id2ans: dict):
        score = 0.
        for instance_id, ans in instance_id2ans.items():
            datum = self.dataset.id2datum[instance_id]
            label = datum['label']
            if ans == label:
                score += 1
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
            for instance_id, ans in instance_id2ans.items():
                result.append({
                    'instanceId': instance_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)
