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

    gold_df = pd.read_csv(args.gold_file)

    # NOTE: make order consistent
    gold_df = Dataset.from_pandas(gold_df).shuffle(seed=0).to_pandas()

    with open(args.gen_file, "r") as f:
        gen_summ = f.readlines()
    gold_df['gen_summ'] = gen_summ

    rouge = Rouge()

    def _compute_rouges(example):
        example["gen_summ_score"] = rouge.get_scores(
            example['gen_summ'], example[summ_column])[0]['rouge-2']['f']
        return example

    gold_df = gold_df.progress_apply(lambda d: _compute_rouges(d), axis=1)
    gold_df = gold_df.sort_values("gen_summ_score", ascending=False)

    top_num = 100
    os.makedirs(os.path.join(args.output_dir, args.dataset_name + "_examples"),
                exist_ok=True)
    gold_df.iloc[:top_num].to_csv(
        os.path.join(args.output_dir,"examples", args.dataset_name,
                     "test.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis the generation results.')
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--gen_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
