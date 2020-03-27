'''
Given a folder full of models and a test file:
1) finds the best model checkpoint according to the validation metric
2) runs the model on the given test_file
3) outputs predictions, cleans up, etc.

'''
import argparse
import os
import subprocess


def call(x, just_print=False):
    if just_print:
        print(x)
        return
    subprocess.call(x, shell=True)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'checkpoint_dir',
        help='directory containing checkpoints')
    return parser.parse_args()


def name2score(name):
    #blah/epoch_0_valscore_65.22267_argshash_XXXX.pth
    return float(name.split('/')[-1].split('_')[-3])
    

def main():
    args = parse_args()
    f2score = {x:name2score(x) for x in os.listdir(args.checkpoint_dir) if '.pth' in x and 'LAST' not in x}

    best_model = args.checkpoint_dir + '/' + list(sorted(f2score.items(), key=lambda x: -x[1]))[0][0]
    new_best_model = args.checkpoint_dir + '/' + 'BEST_' + best_model.split('/')[-1]
    
    call('mv {} {}'.format(best_model, new_best_model))
    best_model = new_best_model

    for x in os.listdir(args.checkpoint_dir):
        if '.pth' in x:
            full_path = '{}/{}'.format(args.checkpoint_dir, x)
            if full_path != new_best_model:
                remove_cmd = 'rm {}'.format(full_path)
                call(remove_cmd)
    
    
if __name__ == '__main__':
    main()
