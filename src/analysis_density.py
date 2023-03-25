import os
import numpy as np
import pandas as pd

from scipy.stats import kde
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns


datasets = ['xsum', 'reddit_tifu', 'rottentomatoes', 'samsum']
true_names = {'xsum':'XSum', 'reddit_tifu':'Reddit-TIFU', 'amazon_reviews_multi':'Amazon',
              'scitldr':'Scitldr', 'rottentomatoes':'Movie', 'samsum':'SAMSum'}
fig = plt.figure(figsize=(8,8))
n_clusters = 8
feat_cols = ['R-1','R-2','R-L','CR','EC','NR']

for idx, dataset in enumerate(datasets):

    # Read training examples
    df_train = pd.read_csv(f"../../ConditionDataset/datasets/supervised/{dataset}/train.csv")
    df_train = df_train.rename(
        columns={
            'rouge_1': 'R-1',
            'rouge_2': 'R-2',
            'rouge_l': 'R-L',
            'compress_ratio': 'CR',
            'ext_coverage': 'EC',
            'ext_density': 'ED',
            'novel_word_ratio': 'NR'
        })
    df_train = df_train[feat_cols]
    df_train["type"] = "train"

    # Randomly pick training examples
    np.random.seed(6)
    rndperm = np.random.permutation(df_train.shape[0])[:100]
    df_train = df_train.iloc[rndperm]

    # Read testing examples
    df_test = pd.read_csv(f"../../ConditionDataset/datasets/supervised/{dataset}/test.csv")
    df_test = df_test.rename(
        columns={
            'rouge_1': 'R-1',
            'rouge_2': 'R-2',
            'rouge_l': 'R-L',
            'compress_ratio': 'CR',
            'ext_coverage': 'EC',
            'ext_density': 'ED',
            'novel_word_ratio': 'NR'
        })
    df_test = df_test[feat_cols]
    df_test["type"] = "test"

    # NOTE: pick data
    q = int(len(df_test)*0.9)
    topk_idx = np.argsort(
        np.dot(df_test[feat_cols].to_numpy(), df_train[feat_cols].to_numpy().T).mean(-1)
    )[q:q+n_clusters]
    df_test = df_test.iloc[topk_idx]

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_train[feat_cols].to_numpy())
    df_rep = pd.DataFrame(kmeans.cluster_centers_, columns=feat_cols)
    df_rep["type"] = "cluster"

    # Concat
    df = pd.concat([df_train, df_test, df_rep], ignore_index=True)

    # Whitening
    df.loc[:, feat_cols] = df[feat_cols].apply(lambda col: col / col.std())

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feat_cols].values)

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    x = pca_result[:len(df_train), 0]
    y = pca_result[:len(df_train), 1]

    nbins=600
    k = kde.gaussian_kde([x,y], 0.1)
    xi, yi = np.mgrid[
        pca_result[:,0].min():pca_result[:,0].max():nbins*1j,
        pca_result[:,1].min():pca_result[:,1].max():nbins*1j
    ]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Make the plot
    ax = fig.add_subplot(2, 2, idx+1)
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.magma)
    ax.scatter(
        x=pca_result[len(df_train):len(df_train)+len(df_test):, 0],
        y=pca_result[len(df_train):len(df_train)+len(df_test):, 1],
        #c="#ef476f",
        c="#1f77b4",
        marker="o",
        s=50,
        alpha=0.9,
        label="Gold Preferences",
    )
    ax.scatter(
        x=pca_result[-n_clusters:, 0],
        y=pca_result[-n_clusters:, 1],
        #c="#118ab2",
        c="#2ca02c",
        marker="P",
        s=60,
        alpha=0.9,
        label="Cluster Preferences",
    )
    ax.set_title(true_names[dataset])

# Shared legend
handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5,1.04))
fig.legend(handles, labels, loc='upper center', ncol=1, bbox_to_anchor=(0.185,0.95))
fig.tight_layout()

# Show the graph
os.makedirs("../analysis", exist_ok=True)
plt.savefig(
    "../analysis/preference_density_plot.png",
    bbox_inches='tight', dpi=300)
plt.show()
