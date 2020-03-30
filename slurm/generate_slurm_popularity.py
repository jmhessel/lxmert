'''
/home/jmh563/compiled_python3/bin/python3
'''
import argparse


header = '''#!/bin/bash
#SBATCH -J {NAME}                         # Job name
#SBATCH -o {NAME}.out                     # Name of stdout output log file (%j expands to jobID)
#SBATCH -e {NAME}.err                     # Name of stderr output log file (%j expands to jobID)
#SBATCH -N 1                             # Total number of nodes requested
#SBATCH -n 1                             # Total number of cores requested
#SBATCH --mem=32000                      # Total amount of (real) memory requested (per node)
#SBATCH -t 168:00:00                     # Time limit (hh:mm:ss)
#SBATCH --partition=default_gpu          # Request partition for resource allocation
#SBATCH --gres=gpu:1                     # Specify a list of generic consumable resources (per node)
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--python_executable',
        default='python')
    return parser.parse_args()


def main():
    args = parse_args()

    for model in ['full']:
        for idx in range(15):
            for lr in [.000001,.000005,.00001,.00005,.0001]:
                fname = '{}_{}_{}.sub'.format(model, idx, lr)
                cur_header = header.format(NAME='{}_{}_{}'.format(model, idx, lr))
                cmd = '{PYTHON} src/tasks/rank.py data/reddit_data/pics_split_{IDX}.json.lxmert_train_json data/reddit_data/pics_split_{IDX}.json.lxmert_val_json data/reddit_data/pics_split_{IDX}.json.lxmert_test_json data/reddit_data/pics_bbox_features.tsv reddit_{MODEL}_{IDX}_{LR} --loadLXMERT snap/pretrained/model_LXRT.pth --model_type {MODEL} --lr {LR} --epochs 10 --optimize_metric acc'.format(
                    PYTHON=args.python_executable,
                    IDX=idx,
                    MODEL=model,
                    LR='{:.9f}'.format(lr))
                rm_cmd = 'python src/clean_all_but_best.py reddit_{MODEL}_{IDX}_{LR}/'.format(MODEL=model, IDX=idx, LR='{:.9f}'.format(lr))
                with open(fname, 'w') as f:
                    f.write(cur_header + '\n\n')
                    f.write('cd ..; ')
                    f.write(cmd + '; ' )
                    f.write(rm_cmd + '\n')

    for model in ['full']:
        for idx in range(15):
            for lr in [.000001,.000005,.00001,.00005,.0001]:
                fname = '{}logits_{}_{}.sub'.format(model, idx, lr)
                cur_header = header.format(NAME='{}logits_{}_{}'.format(model, idx, lr))
                cmd = '{PYTHON} src/tasks/rank.py data/reddit_data/pics_split_{IDX}.json.lxmert_train_json data/reddit_data/pics_split_{IDX}.json.lxmert_val_json data/reddit_data/pics_split_{IDX}.json.lxmert_test_json data/reddit_data/pics_bbox_features.tsv reddit_{MODEL}logits_{IDX}_{LR} --loadLXMERT snap/pretrained/model_LXRT.pth --model_type {MODEL} --lr {LR} --epochs 10 --optimize_metric acc --use_logits 1'.format(
                    PYTHON=args.python_executable,
                    IDX=idx,
                    MODEL=model,
                    LR='{:.9f}'.format(lr))
                rm_cmd = 'python src/clean_all_but_best.py reddit_{MODEL}logits_{IDX}_{LR}/'.format(MODEL=model, IDX=idx, LR='{:.9f}'.format(lr))
                with open(fname, 'w') as f:
                    f.write(cur_header + '\n\n')
                    f.write('cd ..; ')
                    f.write(cmd + '; ' )
                    f.write(rm_cmd + '\n')
    
if __name__ == '__main__':
    main()
