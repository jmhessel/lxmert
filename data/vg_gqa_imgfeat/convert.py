'''
Convert the output object detection tsv to python arrays/jsons
'''
import argparse
import csv
import base64
import time
import sys
import numpy as np
import tqdm
import json

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def item_to_feature_tensors(item):
    # 7 location features from
    # https://arxiv.org/pdf/1909.11740.pdf
    # assume the image has width, height = w, h
    # assume the bbox has width, height = wbox, hbox
    # [x1 / w, y1 / h, x2 / w, y2 / h,  --- normalized coordinates
    #  wbox/w, hbox/h, --- normalized fractional width/height
    #  wbox * hbox / (w * h)] --- normalized area
    # global w,h
    w, h = float(item['img_w']), float(item['img_h'])

    # coordinates for lower left/upper right for boxes
    x1, y1, x2, y2 = item['boxes'].transpose()

    wbox = x2 - x1
    hbox = y2 - y1

    location_features = [x1 / w, y1 / h, x2 / w,
                         x2 / h, wbox / w, hbox / h,
                         wbox * hbox / (w * h)]
    location_features = np.vstack(location_features).transpose()

    ## And, of course, the content features
    content_features = item['features']
    return (item['img_id'], content_features, location_features)


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.
    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.

    via
    https://github.com/airsplay/lxmert/blob/master/src/utils.py
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in tqdm.tqdm(enumerate(reader)):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            if boxes != 36:
                print('WHAT')
                print(i)
                print(item['img_id'])
                continue
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item_to_feature_tensors(item))

            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))

    content_features = np.vstack([np.expand_dims(d[1], 0) for d in data])
    print("{} GB for content features".format(
        content_features.nbytes / 1e9 ))
    location_features = np.vstack([np.expand_dims(d[2], 0) for d in data])
    id2row = dict([(k, v) for v, k in enumerate([d[0] for d in data])])
    return id2row, content_features, location_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('obj_tsv')
    parser.add_argument('dataset_name')
    return parser.parse_args()


def main():
    args = parse_args()
    id2row, content_features, location_features = load_obj_tsv(args.obj_tsv)
    with open('{}_id2row.json'.format(args.dataset_name),
              'w'.format(args.dataset_name)) as f:
        f.write(json.dumps(id2row))

    np.savez('{}_bbox_features.npz'.format(args.dataset_name),
             content_features=content_features,
             location_features=location_features)

    
if __name__ == '__main__':
    main()
