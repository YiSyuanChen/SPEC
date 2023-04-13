""" Visualize Conditions"""
import os
from time import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from datasets import Dataset

from gap_statistic import OptimalK
import pdb

condition_list = [
    'rouge_1', 'rouge_2', 'rouge_l', 'compress_ratio', 'ext_coverage',
    'ext_density', 'novel_word_ratio'
]


def analysis_with_kmeans(file_path,
                         output_dir,
                         num_cluster,
                         perplexity,
                         n_iter,
                         max_samples=None,
                         bias_mode=None):

    file_dir, file_name = os.path.split(file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(file_path)

    # Shuffle
    #df = df.sample(frac=1, random_state=0)
    df = Dataset.from_pandas(df).shuffle(seed=0).to_pandas() # NOTE: follow Huggingface

    if bias_mode:
        if bias_mode == 'lowest_rouge':
            df = df.sort_values(by=["rouge_1"])[df["rouge_1"]!=0]
        elif bias_mode == 'highest_rouge':
            df = df.sort_values(by=["rouge_1"], ascending=False)[df["rouge_1"]!=0]
        else:
            raise ValueError("Invalid bias mode.")

    if max_samples:
        df = df.iloc[:max_samples]

    if num_cluster == None:
        optimalK = OptimalK(parallel_backend='rust')
        num_cluster = optimalK(df[condition_list].values,
                               cluster_array=np.arange(1, 15))
        print(f"Estimated optimal K : {num_cluster}")

    print("KMeans Clustering...")
    kmeans_result = KMeans(n_clusters=num_cluster,
                           random_state=0).fit(df[condition_list].values)

    if max_samples is not None:
        if bias_mode is not None:
            np.save(
                os.path.join(
                    output_dir,
                    f"means_iter_{n_iter}_perp_{perplexity}_cluster_{num_cluster}_examples_{max_samples}_bias_{bias_mode}.npy"
                ), kmeans_result.cluster_centers_)
        else:
            np.save(
                os.path.join(
                    output_dir,
                    f"means_iter_{n_iter}_perp_{perplexity}_cluster_{num_cluster}_examples_{max_samples}.npy"
                ), kmeans_result.cluster_centers_)
    else:
        np.save(
            os.path.join(
                output_dir,
                f"means_iter_{n_iter}_perp_{perplexity}_cluster_{num_cluster}_examples_full.npy"
            ), kmeans_result.cluster_centers_)

    df['kmeans'] = kmeans_result.labels_

    '''
    print("PCA reduction...")
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[condition_list].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    '''

    print("t-SNE computing...")
    time_start = time()
    tsne = TSNE(n_components=2,
                n_jobs=-1,
                verbose=2,
                perplexity=perplexity,
                n_iter=n_iter)
    tsne_results = tsne.fit_transform(df[condition_list])
    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]
    print('t-SNE done! Time elapsed: {} seconds'.format(time() - time_start))

    print("t-SNE computing...")
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x="tsne-one",
                    y="tsne-two",
                    hue="kmeans",
                    palette=sns.color_palette("hls", num_cluster),
                    data=df,
                    legend="full",
                    alpha=0.3)
    if max_samples is not None:
        if bias_mode is not None:
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"tsne_iter_{n_iter}_perp_{perplexity}_cluster_{num_cluster}_examples_{max_samples}_bias_{bias_mode}.png"
                ))
        else:
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"tsne_iter_{n_iter}_perp_{perplexity}_cluster_{num_cluster}_examples_{max_samples}.png"
                ))
    else:
        plt.savefig(
            os.path.join(
                output_dir,
                f"tsne_iter_{n_iter}_perp_{perplexity}_cluster_{num_cluster}_examples_full.png"
            ))
    #plt.show()


def analysis_with_label(file_paths, output_dir, max_samples=None):

    file_dirs = []
    file_names = []
    for file_path in file_paths:
        file_dir, file_name = os.path.split(file_path)
        file_dirs.append(file_dir)
        file_names.append(file_name)

    # Read CSV
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        # Shuffle
        df = df.sample(frac=1, random_state=0)

        if max_samples:
            df = df.iloc[:max_samples]
        dfs.append(df)

    print("Create labels...")
    for idx, df in enumerate(dfs):
        df['class'] = idx
    df = pd.concat(dfs)

    '''
    print("PCA reduction...")
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[condition_list].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    '''

    print("t-SNE computing...")
    time_start = time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=30)
    tsne_results = tsne.fit_transform(df[condition_list])
    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]
    print('t-SNE done! Time elapsed: {} seconds'.format(time() - time_start))

    plt.figure(figsize=(16, 10))
    sns.scatterplot(x="tsne-one",
                    y="tsne-two",
                    hue="class",
                    palette=sns.color_palette("hls", len(dfs)),
                    data=df,
                    legend="full",
                    alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    if max_samples != None:
        plt.savefig(os.path.join(output_dir, f"tsne_{max_samples}_samples.png"))
    else:
        plt.savefig(os.path.join(output_dir, f"tsne_full_samples.png"))
    #plt.show()


def main(args):
    if args.dataset_file_2 == None:
        analysis_with_kmeans(args.dataset_file_1, args.output_dir,
                             args.num_cluster, args.perplexity, args.n_iter,
                             args.max_samples, args.bias_mode)
    else:
        files = [args.dataset_file_1, args.dataset_file_2]
        analysis_with_label(files, args.output_dir, args.max_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract Extractive Oracle according to ROUGE')
    parser.add_argument("--dataset_file_1", type=str, required=True)
    parser.add_argument("--dataset_file_2", type=str, default=None)
    parser.add_argument("--name_1", type=str, required=True)
    parser.add_argument("--name_2", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_cluster", type=int, default=None)
    parser.add_argument("--perplexity", type=int, default=None)
    parser.add_argument("--n_iter", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bias_mode", type=str, default=None)
    args = parser.parse_args()

    main(args)
