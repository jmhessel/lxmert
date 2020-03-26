'''
Plots accuracy orig vs. accuracy projected
'''
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results',
        default='results.txt')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.results) as f:
        results = [x.strip().split()[-1].split('/') for x in f.readlines()]
    results = [(float(x[0]), float(x[1]), float(x[-1])) for x in results]
    orig_acc = np.array([r[0] for r in results])
    proj_acc = np.array([r[1] for r in results])
    mean_acc = np.array([r[2] for r in results])
    print('Orig acc: {:.2f}, proj acc: {:.2f}, const acc: {:.2f}'.format(
        np.mean(orig_acc), np.mean(proj_acc), np.mean(mean_acc)))    
    orig_better = np.sum(orig_acc > proj_acc)
    plt.scatter(orig_acc, proj_acc)
    plt.plot([0, 100], [0, 100], linestyle='--', linewidth=3, color='r')
    start = min(np.min(orig_acc) - 1, np.min(proj_acc) - 1)
    end = max(np.max(orig_acc) + 1, np.max(orig_acc) + 1)
    plt.xlim(start, end)
    plt.ylim(start, end)
    plt.xlabel('Accuracy Original (win rate={:.0f}%)'.format(orig_better / len(orig_acc) * 100))
    plt.ylabel('Accuracy Projected')
    plt.tight_layout()
    plt.savefig('gqa_results.pdf')
    
if __name__ == '__main__':
    main()
