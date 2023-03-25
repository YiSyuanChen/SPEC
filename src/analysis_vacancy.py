import os
import argparse
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets, random_projection
from sklearn.preprocessing import normalize, StandardScaler
import matplotlib.pyplot as plt
import networkx as nx
import kmapper as km

np.random.seed(0)

def main(args):
    zs = pd.read_csv(args.csv_file)[
        ["rouge_1", "rouge_2", "rouge_l", "compress_ratio", "ext_coverage", "ext_density", "novel_word_ratio"]
    ].to_numpy()
    zs = zs[np.random.permutation(len(zs))][:1000]

    if args.add_noise:
        zs = np.concatenate([zs+np.random.normal(0, 1, zs.shape) for i in range(args.repetition)])
    else:
        zs = np.concatenate([zs for i in range(args.repetition)])

    zs = normalize(zs, axis=1, norm='l1')
    scaler = StandardScaler().fit(zs)
    zs = scaler.transform(zs)

    mapper = km.KeplerMapper(verbose=1)
    z_embed = mapper.fit_transform(zs, projection='sum')
    graph = mapper.map(z_embed, zs,
            clusterer=sklearn.cluster.DBSCAN(eps=0.08, min_samples=3, metric='cosine'),
            cover=km.Cover(n_cubes=20, perc_overlap=0.4))
    mapper.visualize(graph, path_html=os.path.join(args.output_dir,f'tda_{args.repetition}{"_noise" if args.add_noise else ""}.html'), title='tda')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis the preference vacancy.')
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument('--add_noise', action="store_true")
    parser.add_argument('--repetition', type=int)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
