import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

tqdm.pandas()

from datasets import Dataset

from data.build_datasets import summarization_name_mapping

from rouge import Rouge
import pdb


def main(args):
    art_column, summ_column = summarization_name_mapping.get(
        args.dataset_name, None)

    # Load data
    test_df = pd.read_csv(args.test_file)

    # NOTE: make order consistent
    test_df = Dataset.from_pandas(test_df).shuffle(seed=0).to_pandas()

    # Load all generation file
    folders = sorted(os.listdir(args.gen_folder))
    for folder in folders:
        with open(os.path.join(args.gen_folder, folder, 'test_generations.txt'), "r") as f:
            gen_summ = f.readlines()
        test_df[folder] = gen_summ

    # Write outputs
    pdb.set_trace()
    test_df = test_df[[art_column, summ_column]+folders]
    #test_df = test_df.reindex(sorted(test_df.columns), axis=1)

    # Compute ROUGE
    rouge = Rouge()
    def _compute_rouges(example):
        for folder in folders:
            example[folder+'_score'] = rouge.get_scores(example[folder],
                                                        example[summ_column])[0]['rouge-2']['f']
        return example
    test_df = test_df.progress_apply(lambda d: _compute_rouges(d), axis=1)
    test_df = test_df.sort_values(folders[0]+'_score', ascending=False)
    test_df.to_excel(os.path.join(args.output_dir,"examples_"+args.dataset_name+'.xlsx'))

    # NOTE: choose 49

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis the generation results.')
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--gen_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
