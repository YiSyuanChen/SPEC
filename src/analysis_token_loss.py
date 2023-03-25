import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
pal = sns.color_palette('Set1')

def main(args):
    data_gold = np.load(args.token_loss_file_1)
    data_cluster = np.load(args.token_loss_file_2)

    data_gold[data_gold==np.inf] = 31
    data_cluster[data_cluster==np.inf] = 31

    sns.histplot(data=data_gold, alpha=.1, color=pal[0],
                 label="Gold", kde=True, bins=range(0,32,1), stat="percent")
    sns.histplot(data=data_cluster, alpha=.1, color=pal[1],
                 label="Cluster", kde=True, bins=range(0,32,1), stat="percent")

    plt.xlim(0, 25)
    plt.legend(prop={'size':20})

    # Change font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Loss", fontsize=20)
    plt.ylabel("Percentage", fontsize=20)

    plt.savefig(os.path.join(args.output_dir, "token_loss_xsum.png"),
                bbox_inches='tight',
                dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis the generation results.')
    parser.add_argument("--token_loss_file_1", type=str, required=True)
    parser.add_argument("--token_loss_file_2", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
