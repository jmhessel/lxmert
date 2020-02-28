'''
Reports normal and additive accuracies:

python parse_logit_results.py repeat_10_testdev_predict.json ../../../data/gqa/trainval_ans2label.json
'''
import argparse
import json
import collections
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_preds')
    parser.add_argument('ans2label')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.ans2label) as f:
        ans2label = json.load(f)
    label2ans = {v:k for k, v in ans2label.items()}
    print('Loaded {} labels'.format(len(label2ans)))
    
    with open(args.input_preds) as f:
        preds = json.load(f)
    print('Loaded {} preds'.format(len(preds)))

    v_idx2t_idx2logits = collections.defaultdict(dict)
    idx2data = {}
    for p in preds:
        assert(p['answer'] == label2ans[np.argmax(p['logits'])])
        v_idx2t_idx2logits[p['input']['image_idx']][p['input']['text_idx']] = np.array(p['logits'])
        if p['input']['image_idx'] == p['input']['text_idx']:
            idx2data[p['input']['image_idx']] = p['input']
        
    # compute means
    n_points = len(v_idx2t_idx2logits)
    v_idx2mean, t_idx2mean = collections.defaultdict(list), collections.defaultdict(list)
    mean = []
    for v_idx in range(n_points):
        for t_idx in range(n_points):
            v_idx2mean[v_idx].append(v_idx2t_idx2logits[v_idx][t_idx])
            t_idx2mean[t_idx].append(v_idx2t_idx2logits[v_idx][t_idx])
            mean.append(v_idx2t_idx2logits[v_idx][t_idx])
    v_idx2mean = {k: np.mean(np.vstack(v), axis=0) for k, v in v_idx2mean.items()}
    t_idx2mean = {k: np.mean(np.vstack(v), axis=0) for k, v in t_idx2mean.items()}
    mean = np.mean(np.vstack(mean), axis=0)

    # data:
    # {'answer_type': 'other', 'image_idx': 0, 'img_id':
    # 'COCO_val2014_000000229782', 'label': {'rectangle': 1, 'square':
    # 0.3}, 'question_id': 0, 'question_type': 'what', 'sent': 'What
    # shape is this sign in?', 'text_idx': 0}

    def get_pred(logits):
        return label2ans[np.argmax(logits)]

    orig_tot, add_tot, lang_tot, image_tot, mean_tot = 0, 0, 0, 0, 0
    for idx in range(n_points):
        data = idx2data[idx]
        pred_orig = get_pred(v_idx2t_idx2logits[idx][idx])
        pred_additive = get_pred(v_idx2mean[idx] + t_idx2mean[idx] - mean)
        pred_lang = get_pred(t_idx2mean[idx])
        pred_image = get_pred(v_idx2mean[idx])
        pred_mean = get_pred(mean)

        if pred_orig in data['label']:
            orig_tot += data['label'][pred_orig]

        if pred_additive in data['label']:
            add_tot += data['label'][pred_additive]

        if pred_lang in data['label']:
            lang_tot += data['label'][pred_lang]

        if pred_image in data['label']:
            image_tot += data['label'][pred_image]
            
        if pred_mean in data['label']:
            mean_tot += data['label'][pred_mean]
            

    print('orig/add/image/text/mean, n points={}: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(
        n_points,
        orig_tot / n_points * 100,
        add_tot / n_points * 100,
        image_tot / n_points * 100,
        lang_tot / n_points * 100,
        mean_tot / n_points * 100))
        
        

    
if __name__ == '__main__':
    main()
