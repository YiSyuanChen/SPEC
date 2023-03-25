import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import pdb

pal = sns.color_palette('tab20c')

fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 10))
ylimits = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

#datasets = ['xsum', 'reddit_tifu', 'amazon_reviews_multi', 'scitldr', 'rottentomatoes', 'samsum']
datasets = ['amazon_reviews_multi', 'reddit_tifu', 'rottentomatoes', 'samsum', 'scitldr', 'xsum']
#datasets = ['xsum']
true_names = {'xsum':'XSum', 'reddit_tifu':'Reddit-TIFU', 'amazon_reviews_multi':'Amazon',
              'scitldr':'Scitldr', 'rottentomatoes':'Movie', 'samsum':'SAMSum'}
for dataset, ax, ylimit in zip(datasets, axs, ylimits):
    ax.set_ylim(ylimit)
    #ax.set_title(true_names[dataset])
    ax.text(0.05,0.7, true_names[dataset], fontsize=18)
    ax.tick_params(axis='x', labelsize=18)

    # Read centroids
    df_cent = pd.DataFrame(np.load(
        f"../../ConditionDataset/analysis/{dataset}/means_iter_5000_perp_30_cluster_8_examples_100.npy"
    ),
                           columns=[
                               'rouge_1', 'rouge_2', 'rouge_l',
                               'compress_ratio', 'ext_coverage', 'ext_density',
                               'novel_word_ratio'
                           ])
    df_cent = df_cent[[
        'rouge_1', 'rouge_2', 'rouge_l', 'compress_ratio', 'ext_coverage',
        'novel_word_ratio'
    ]]
    df_cent['dataset'] = "Centroids"
    df_cent = df_cent.rename(
        columns={
            'rouge_1': 'R-1',
            'rouge_2': 'R-2',
            'rouge_l': 'R-L',
            'compress_ratio': 'CR',
            'ext_coverage': 'EC',
            'ext_density': 'ED',
            'novel_word_ratio': 'NR'
        })

    # Read data
    df = pd.read_csv(
        f"../../ConditionDataset/datasets/supervised/{dataset}/test.csv")
    df = df[[
        'rouge_1', 'rouge_2', 'rouge_l', 'compress_ratio', 'ext_coverage',
        'novel_word_ratio'
    ]]
    df['dataset'] = dataset
    df = df.rename(
        columns={
            'rouge_1': 'R-1',
            'rouge_2': 'R-2',
            'rouge_l': 'R-L',
            'compress_ratio': 'CR',
            'ext_coverage': 'EC',
            'ext_density': 'ED',
            'novel_word_ratio': 'NR'
        })

    # Read unsupervised data
    df_unsup = pd.read_csv(
        f"../../ConditionDataset/datasets/unsupervised/{dataset}/test.csv")

    # Shuffle and take part data
    df_unsup = df_unsup.sample(frac=1).reset_index(drop=True)
    df_unsup = df_unsup[:len(df)]

    df_unsup = df_unsup[[
        'rouge_1', 'rouge_2', 'rouge_l', 'compress_ratio', 'ext_coverage',
        'novel_word_ratio'
    ]]
    df_unsup['dataset'] = dataset
    df_unsup = df_unsup.rename(
        columns={
            'rouge_1': 'R-1',
            'rouge_2': 'R-2',
            'rouge_l': 'R-L',
            'compress_ratio': 'CR',
            'ext_coverage': 'EC',
            'ext_density': 'ED',
            'novel_word_ratio': 'NR'
        })

    # Make the plot
    parallel_coordinates(df_unsup,
                         'dataset',
                         color=matplotlib.colors.rgb2hex(pal[1]),
                         alpha=0.01,
                         ax=ax)
    parallel_coordinates(df,
                         'dataset',
                         color=matplotlib.colors.rgb2hex(pal[-1]),
                         alpha=0.01,
                         ax=ax)
    #parallel_coordinates(df_cent,
    #                     'dataset',
    #                     color=matplotlib.colors.rgb2hex(pal[0]),
    #                     alpha=0.5,
    #                     ax=ax)
    ax.get_legend().remove()

plt.savefig("../analysis/preference_parallel_plot.png")
#plt.show()
