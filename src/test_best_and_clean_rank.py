'''
Given a folder full of models and a test file:
1) finds the best model checkpoint according to the validation metric
2) runs the model on the given test_file
3) outputs predictions, cleans up, etc.

'''
import argparse
import os
import subprocess


def call(x, just_print=True):
    if just_print:
        print(x)
        return
    subprocess.call(x, shell=True)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'checkpoint_dir',
        help='directory containing checkpoints')
    parser.add_argument(
        'test_file',
        help='file to produce test predictions on')
    parser.add_argument(
        'image_features',
        help='path to image features')
    parser.add_argument(
        '--script_name',
        default='rank.py')
    parser.add_argument(
        '--use_logits',
        default=0)
    parser.add_argument(
        '--model_type',
        default='full')
    
    return parser.parse_args()


def name2score(name):
    #blah/epoch_0_valscore_65.22267_argshash_XXXX.pth
    return float(name.split('/')[-1].split('_')[-3])
    

def main():
    args = parse_args()
    f2score = {x:name2score(x) for x in os.listdir(args.checkpoint_dir) if '.pth' in x and 'LAST' not in x}

    best_model = args.checkpoint_dir + '/' + list(sorted(f2score.items(), key=lambda x: -x[1]))[0][0]

    # we did this already in clean_all_but_best.py
    #new_best_model = args.checkpoint_dir + '/' + 'BEST_' + best_model.split('/')[-1]
    #call('cp {} {}'.format(best_model, new_best_model))
    #best_model = new_best_model
    
    test_cmd = '/usr/local/bin/python3 src/tasks/{} -1 -1 {} {} {} --load_finetune {} --use_logits {} --model_type {} --batchSize 512'.format(
        args.script_name, args.test_file, args.image_features, args.checkpoint_dir, best_model, args.use_logits, args.model_type)

    call(test_cmd)
    
    
if __name__ == '__main__':
    main()
