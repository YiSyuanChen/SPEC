import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pdb


#datasets = ['samsum', 'samsum', 'samsum', 'samsum']
#datasets = ['samsum', 'samsum', 'samsum', 'samsum']
datasets = ['xsum', 'reddit_tifu', 'rottentomatoes', 'samsum']
true_names = {'xsum':'XSum', 'reddit_tifu':'Reddit-TIFU', 'amazon_reviews_multi':'Amazon',
              'scitldr':'Scitldr', 'rottentomatoes':'Movie', 'samsum':'SAMSum'}
fig = plt.figure(figsize=(8,8))

for idx, dataset in enumerate(datasets):
    # Read data
    df = pd.read_csv(
        f"../../ConditionDataset/datasets/supervised/{dataset}/test.csv")
    df = df[[
        'rouge_1', 'rouge_2', 'rouge_l', 'compress_ratio', 'ext_coverage',
        'novel_word_ratio'
    ]]
    df['dataset'] = 0 #"Supervised"
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
    df_unsup['dataset'] = 1 #"Self-Supervised"
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
    df = pd.concat([df, df_unsup], ignore_index=True)

    # Whitening
    feat_cols = ['R-1','R-2','R-L','CR','EC','NR']
    df.loc[:, feat_cols] = df[feat_cols].apply(lambda col: col / col.std())

    # For reproducing
    np.random.seed(0)
    rndperm = np.random.permutation(df.shape[0])

    # PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]

    # Remove Outlier
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)].reset_index(drop=True)


    """
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="dataset",
        palette=sns.color_palette("hls", 2),
        data=df.loc[rndperm,:],
        legend="full",
        alpha=0.3
    )
    plt.show()
    """

    #ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax = fig.add_subplot(2, 2, idx+1, projection='3d')
    ax.scatter(
        xs=df["pca-one"],#xs=df.loc[rndperm,:]["pca-one"],
        ys=df["pca-two"],#ys=df.loc[rndperm,:]["pca-two"],
        zs=df["pca-three"],#zs=df.loc[rndperm,:]["pca-three"],
        c=df["dataset"],
        cmap='tab10'
    )
    ax.view_init(elev=15, azim=74)
    #ax.set_xlabel('PCA-1')
    #ax.set_ylabel('PCA-2')
    #ax.set_zlabel('PCA-3')
    ax.set_title(true_names[dataset], y=0.01)

# Show the graph
fig.tight_layout(pad=-0.5)
os.makedirs("../analysis", exist_ok=True)
#plt.savefig(
#    "../analysis/preference_pca_plot.png",
#    bbox_inches='tight', dpi=300)

plt.show() 
