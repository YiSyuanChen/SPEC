import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rouge import Rouge
import pdb


def main(args):
    gold_df = pd.read_csv(args.gold_file)

    with open(args.gen_file_1, "r") as f:
        mid_rouge_summ = f.readlines()
    with open(args.gen_file_2, "r") as f:
        high_rouge_summ = f.readlines()

    gold_df['mid_rouge_summ'] = mid_rouge_summ
    gold_df['high_rouge_summ'] = high_rouge_summ

    rouge = Rouge()
    def _compute_rouges(example):
        example["mid_rouge_summ_score"] = rouge.get_scores(example['mid_rouge_summ'], example['summary'])[0]['rouge-1']['f']
        example["high_rouge_summ_score"] = rouge.get_scores(example['high_rouge_summ'], example['summary'])[0]['rouge-1']['f']
        return example

    gold_df = gold_df.apply(lambda d: _compute_rouges(d), axis=1)
    gold_df = gold_df.sort_values("high_rouge_summ_score", ascending=False)
    # NOTE: use gold_df.iloc[1]
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis the generation results.')
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--gen_file_1", type=str, required=True)
    parser.add_argument("--gen_file_2", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
