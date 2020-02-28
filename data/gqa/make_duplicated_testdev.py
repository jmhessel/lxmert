'''
Generates json file
'''
import argparse
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_json',
        default='testdev.json',
        type=str)
    parser.add_argument(
        '--n_questions',
        default=100,
        type=int)
    parser.add_argument(
        '--random_seed',
        default=1,
        type=int)
    return parser.parse_args()


def main():    
    args = parse_args()
    np.random.seed(args.random_seed)
    
    with open(args.input_json) as f:
        data = json.load(f)

    np.random.shuffle(data)
    data = data[:args.n_questions]

    v_idx2img = {}
    t_idx2text = {}
    idx2other = {}

    oth_key = ['label']
    
    # {'answer_type': 'other', 'img_id': 'COCO_val2014_000000229782', 'label': {'rectangle': 1, 'square': 0.3},
    # 'question_id': 229782003, 'question_type': 'what', 'sent': 'What shape is this sign in?'}
    for idx, d in enumerate(data):
        v_idx2img[idx] = d['img_id']
        t_idx2text[idx] = d['sent']
        idx2other[idx] = {k: d[k] for k in oth_key}

    question_id = 0
    new_data = []
    for v_idx in range(args.n_questions):
        for t_idx in range(args.n_questions):
            cur_dict = idx2other[v_idx].copy()
            cur_dict['image_idx'] = v_idx
            cur_dict['text_idx'] = t_idx
            cur_dict['sent'] = t_idx2text[t_idx]
            cur_dict['img_id'] = v_idx2img[v_idx]
            cur_dict['question_id'] = question_id
            new_data.append(cur_dict)
            question_id += 1

    print('generated {} new questions'.format(question_id))
    with open('repeat_{}_'.format(args.n_questions) + args.input_json, 'w') as f:
        f.write(json.dumps(new_data))

        
if __name__ == '__main__':
    main()
