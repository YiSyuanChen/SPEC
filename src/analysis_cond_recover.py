import os
import re
from copy import deepcopy
from math import pi
import argparse
    
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

tqdm.pandas()
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

from datasets import load_dataset

from data.build_datasets import summarization_name_mapping

import pdb

##### From PreSumm (EMNLP 2019) ####
# NOTE: this is way more faster than huggingface metrics


def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
        n (int): which n-grams to calculate
        text (list[str]): a list of words
    Returns:
        A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates n-grams for multiple sentences.

    Args:
        n (int): which n-grams to calculate
        sentences (list[list[str]]): a list of list of words
    Returns:
        A set of n-grams
    """
    assert len(sentences) > 0
    assert n > 0
    words = sum(sentences, [])
    return _get_ngrams(n, words)


def cal_rouge_n(evaluated_ngrams, reference_ngrams):
    """ Compute ROUGE-N scores.

    Args:
        evaluated_ngrams (set(tuple))
        reference_ngrams (set(tuple))
    Returns:
        A dict of ROUGE-N scores.
    """
    evaluated_count = len(evaluated_ngrams)
    reference_count = len(reference_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


##### Other Metrics #####


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]
    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def cal_rouge_l(evaluated_grams, reference_grams):
    """ Compute ROUGE-L scores.

    Args:
        evaluated_ngrams (list[str])
        reference_ngrams (list[str])
    Returns:
        A dict of ROUGE-L scores.
    """
    evaluated_count = len(evaluated_grams)
    reference_count = len(reference_grams)

    lcs_count = lcs(evaluated_grams, reference_grams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = lcs_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = lcs_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def cal_compression_ratio(art_tokens, summ_tokens):
    if len(art_tokens) != 0:
        compress_ratio = len(summ_tokens) / len(art_tokens)
    else:
        compress_ratio = 0.0
    return compress_ratio


def cal_ext_degree(art_tokens, summ_tokens):
    """ Compute extractive diversity and density.

    Args:
        art_tokens (list[str])
        summ_tokens (list[str])
    """
    art_token_num = len(art_tokens)
    summ_token_num = len(summ_tokens)

    # Extract fragments
    fragments = []
    i = 0
    j = 0
    while i < summ_token_num:
        best_fragment = []
        while j < art_token_num:
            if summ_tokens[i] == art_tokens[j]:
                i_ = i
                j_ = j
                cand_fragment = []
                while summ_tokens[i_] == art_tokens[j_]:
                    cand_fragment.append(summ_tokens[i_])
                    i_ += 1
                    j_ += 1
                    if i_ == summ_token_num or j_ == art_token_num:
                        break
                if len(best_fragment) < i_ - i:
                    best_fragment = cand_fragment
                j = j_
            else:
                j += 1

        i = i + max(len(best_fragment), 1)
        j = 1
        fragments.append(best_fragment)

    # Compute etractive fragment coverage / density
    if summ_token_num != 0:
        coverage = (1 / summ_token_num) * sum([len(f) for f in fragments])
        density = (1 / summ_token_num) * sum([len(f)**2 for f in fragments])
    else:
        coverage = 0.0
        density = 0.0

    return coverage, density


def cal_novel_word_ratio(art_tokens, summ_tokens):
    """ Compute novel word ratio in summary.

    Args:
        art_tokens (list[str])
        summ_tokens (list[str])
    """
    stops = set(stopwords.words('english'))
    novel_word_count = 0
    for t in summ_tokens:
        if t not in art_tokens and t not in stops:
            novel_word_count += 1

    if len(summ_tokens) != 0:
        novel_word_ratio = novel_word_count / len(summ_tokens)
    else:
        novel_word_ratio = 0.0

    return novel_word_ratio


def analysis(args, gen_file):

    art_column, summ_column = summarization_name_mapping.get(
        args.dataset_name, None)
    summ_column = summ_column + "_gen"

    # Read data
    extension = args.test_file.split(".")[-1]
    datasets = load_dataset(extension, data_files={"test": args.test_file})
    datasets = datasets.shuffle(seed=0)
    with open(gen_file, "r") as f:
        gen_results = f.readlines()
    df = datasets['test'].to_pandas()
    df[summ_column] = gen_results

    if args.max_samples:
        df = df.iloc[:args.max_samples]

    def _analysis(data):

        # NOTE: handle NaN input
        if not isinstance(data[art_column], str):
            data[art_column] = " "
        if not isinstance(data[summ_column], str):
            data[summ_column] = " "

        # Simple text cleaning
        def _rouge_clean(s):
            return re.sub(r'[^a-zA-Z0-9 ]', '', s)

        art = _rouge_clean(data[art_column])
        summ = _rouge_clean(data[summ_column])

        # Tokenize
        art_tokens = nltk.word_tokenize(art)
        summ_tokens = nltk.word_tokenize(summ)

        # Create ngrams
        evaluated_1grams = _get_word_ngrams(1, [art_tokens])
        evaluated_2grams = _get_word_ngrams(2, [art_tokens])
        reference_1grams = _get_word_ngrams(1, [summ_tokens])
        reference_2grams = _get_word_ngrams(2, [summ_tokens])

        # Compute properties
        rouge_1 = cal_rouge_n(evaluated_1grams, reference_1grams)['f']
        rouge_2 = cal_rouge_n(evaluated_2grams, reference_2grams)['f']
        rouge_l = cal_rouge_l(art_tokens, summ_tokens)['f']
        compress_ratio = cal_compression_ratio(art_tokens, summ_tokens)
        ext_coverage, ext_density = cal_ext_degree(art_tokens, summ_tokens)
        novel_word_ratio = cal_novel_word_ratio(art_tokens, summ_tokens)

        data['rouge_1'] = abs(rouge_1 - data['rouge_1'])
        data['rouge_2'] = abs(rouge_2 - data['rouge_2'])
        data['rouge_l'] = abs(rouge_l - data['rouge_l'])
        data['compress_ratio'] = abs(compress_ratio - data['compress_ratio'])
        data['ext_coverage'] = abs(ext_coverage - data['ext_coverage'])
        data['ext_density'] = abs(ext_density - data['ext_density'])
        data['novel_word_ratio'] = abs(novel_word_ratio -
                                       data['novel_word_ratio'])
        data = data[[
            'rouge_1', 'rouge_2', 'rouge_l', 'compress_ratio', 'ext_coverage',
            'ext_density', 'novel_word_ratio'
        ]]

        return data

    if args.debug:
        df = df.progress_apply(lambda d: _analysis(d), axis=1)
    else:
        df = df.parallel_apply(lambda d: _analysis(d), axis=1)

    avg_error = df.mean(axis=0).tolist()
    std_error = df.std(axis=0).tolist()
    return df, avg_error, std_error


def main(args):

    # Analysis
    df_1, avg_error_1, std_error_1 = analysis(args, args.gen_file_1)
    df_2, avg_error_2, std_error_2 = analysis(args, args.gen_file_2)


    df_1 = df_1.rename(columns={'rouge_1':'R-1', 'rouge_2':'R-2', 'rouge_l':'R-L',
                                'compress_ratio':'CR', 'ext_coverage':'EC',
                                'ext_density':'ED', 'novel_word_ratio':'NR'})
    df_2 = df_2.rename(columns={'rouge_1':'R-1', 'rouge_2':'R-2', 'rouge_l':'R-L',
                                'compress_ratio':'CR', 'ext_coverage':'EC',
                                'ext_density':'ED', 'novel_word_ratio':'NR'})

    df_1 = df_1.drop(columns=['ED'])
    df_2 = df_2.drop(columns=['ED'])
    df_2 = df_2.stack().reset_index().sort_values(['level_1'])
    df_1 = df_1.stack().reset_index().sort_values(['level_1'])
    df_1["type"] = "Gold"
    df_2["type"] = "Cluster"
    df = pd.concat([df_1, df_2])
    df = df.rename(columns={0: "Error", "level_1": "Preference"})
    #sns.set_context("paper", rc={"font.size":14,"axes.titlesize":20,"axes.labelsize":20})
    sns.boxplot(x="Preference", y="Error", hue="type", data=df, palette="Set1", width=0.5, showfliers = False)

    # Add legend
    plt.legend(loc='upper right', prop={'size':20})

    # Change font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Preference", fontsize=20)
    plt.ylabel("Error", fontsize=20)

    # Show the graph
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(args.output_dir,
                     "condition_error_" + args.dataset_name + ".png"),
        bbox_inches='tight', dpi=300)

    plt.show()

    return

    """
    # Create background

    # number of variable
    categories = [
        'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Compress Ratio', 'Extractive\nDivergence',
        'Extractive\nDensity', 'Novel Word Ratio'
    ]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.2, 0.3], ["0.1", "0.2", "0.3"], color="grey", size=7)
    plt.ylim(0, 0.7)

    # Add plots
    # First
    values = avg_error_1
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Gold")

    std_top = [e+s for e, s in zip(avg_error_1, std_error_1)]
    std_bot = [e-s for e, s in zip(avg_error_1, std_error_1)]
    pdb.set_trace()
    ax.fill(angles, values, 'b', alpha=0.1)

    # Second
    values = avg_error_2
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Cluster")
    ax.fill(angles, values, 'r', alpha=0.1)

    # NOTE: adjust the position of xtick
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right') 

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.0, 0.1))

    # Show the graph
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(args.output_dir,
                     "condition_error_" + args.dataset_name + ".png"))
    plt.show()
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis the generation results.')
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--gen_file_1", type=str, required=True)
    parser.add_argument("--gen_file_2", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    main(args)
