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
#SBATCH --mem=64000                      # Total amount of (real) memory requested (per node)
#SBATCH -t 168:00:00                     # Time limit (hh:mm:ss)
#SBATCH --partition=default_gpu          # Request partition for resource allocation
#SBATCH --gres=gpu:1                     # Specify a list of generic consumable resources (per node)
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--python_executable',
        default='/home/jmh563/compiled_python3/bin/python3')
    parser.add_argument(
        '--commands_to_run',
        default='test_time_commands_to_run.txt')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.commands_to_run) as f:
        all_commands = [x.strip() for x in f.readlines()]

    for idx in range(len(all_commands)):
        fname = 'test_{}.sub'.format(idx)
        cur_header = header.format(NAME='test_{}'.format(idx))
        
        with open(fname, 'w') as f:
            f.write(cur_header + '\n\n')
            f.write('cd ..; ')
            f.write(all_commands[idx] + ';\n')

    
if __name__ == '__main__':
    main()
