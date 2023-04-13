""" Create condition datasets. """
import os
import re
import math
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import nltk
import gensim.downloader as downloader
from nltk.corpus import stopwords
from tqdm import tqdm

tqdm.pandas()
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
import pdb
#from third_parties.doc_sim.api import topic_similarity

summarization_name_mapping = {
    "xsum": ("document", "summary"),
    "cnn_dailymail": ("article", "highlights"),
    "newsroom": ("text", "summary"),  # Manual 
    "multi_news": ("document", "summary"),
    "gigaword": ("document", "summary"),
    "wikihow": ("text", "headline"),  # Manual
    "reddit": ("content", "summary"),
    "reddit_tifu": ("documents", "tldr"),
    "big_patent": ("description", "abstract"),
    "scientific_papers": ("article", "abstract"),
    "aeslc": ("email_body", "subject_line"),
    "billsum": ("text", "summary"),
    "wiki_xsum": ("article", "summary"),
    "xlsum": ("text", "summary"),
    "scitldr": ("source", "target"),
    "samsum": ("dialogue", "summary"),
    "amazon_reviews_multi": ("review_body", "review_title"),
    "msr_text_compression": ("source_text", "targets"),
    "rottentomatoes": ("_critics", "_critic_consensus"),
    "idebate": ("_argument_sentences", "_claim"),
    "manual": ("document", "summary"), # NEW: for manual data
}

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


def compute_conditions(art_sents, summ_sent_num, max_sent_len):
    """
    Args:
        art_sents (list[list[str]]) : article sentences
        summ_sent_num (int) : number of sentences to be a summary
        max_sent_len (int) : max number of words in a summary for calculation

    Returns:
        conditions (dict{list}) : conditions for leave-one-out dataset
    """

    # Clean text
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    art_sents = [_rouge_clean(sent) for sent in art_sents]

    # Tokenize each sentence
    art_sents_tokens = [nltk.word_tokenize(sent) for sent in art_sents]
    art_sents_tokens = [s[:max_sent_len] for s in art_sents_tokens]

    # Get ngrams for each sentence
    evaluated_1grams = [
        _get_word_ngrams(1, [sent]) for sent in art_sents_tokens
    ]
    evaluated_2grams = [
        _get_word_ngrams(2, [sent]) for sent in art_sents_tokens
    ]

    # Compute conditions
    art_sent_num = len(art_sents_tokens)
    conditions = defaultdict(list)

    if art_sent_num > summ_sent_num:
        max_out_id_start = art_sent_num - summ_sent_num
        for out_id_start in range(0, max_out_id_start + 1):

            out_ids = set(range(out_id_start, out_id_start + summ_sent_num))
            out_ids = sorted(out_ids)
            in_ids = set(range(art_sent_num)).difference(out_ids)
            in_ids = sorted(in_ids)

            # Compute ROUGE-N
            candidates_1 = [evaluated_1grams[i] for i in in_ids]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[i] for i in in_ids]
            candidates_2 = set.union(*map(set, candidates_2))
            reference_1 = [evaluated_1grams[i] for i in out_ids]
            reference_1 = set.union(*map(set, reference_1))
            reference_2 = [evaluated_2grams[i] for i in out_ids]
            reference_2 = set.union(*map(set, reference_2))
            rouge_1 = cal_rouge_n(candidates_1, reference_1)['f']
            rouge_2 = cal_rouge_n(candidates_2, reference_2)['f']

            # Compute ROUGE-L
            candidates_l = sum([art_sents_tokens[i] for i in in_ids], [])
            reference_l = sum([art_sents_tokens[i] for i in out_ids], [])
            rouge_l = cal_rouge_l(candidates_l, reference_l)['f']

            # Compute compress ratio
            compress_ratio = cal_compression_ratio(candidates_l, reference_l)

            # Compute extractive degree
            ext_coverage, ext_density = cal_ext_degree(candidates_l,
                                                       reference_l)

            # Compute noval word ratio
            novel_word_ratio = cal_novel_word_ratio(candidates_l, reference_l)

            # Record
            conditions['rouge_1'].append(rouge_1)
            conditions['rouge_2'].append(rouge_2)
            conditions['rouge_l'].append(rouge_l)
            conditions['compress_ratio'].append(compress_ratio)
            conditions['ext_coverage'].append(ext_coverage)
            conditions['ext_density'].append(ext_density)
            conditions['novel_word_ratio'].append(novel_word_ratio)
    else:
        # NOTE: use empty summary if no enough sentences to be extracted
        conditions['rouge_1'].append(0.0)
        conditions['rouge_2'].append(0.0)
        conditions['rouge_l'].append(0.0)
        conditions['compress_ratio'].append(0.0)
        conditions['ext_coverage'].append(0.0)
        conditions['ext_density'].append(0.0)
        conditions['novel_word_ratio'].append(0.0)

    return conditions


def leave_one_out(art_sents, summ_sent_num):
    """ Create leave-one-out articles and summaries.

    Args:
        art_sents (list[str]) : article sentences
        summ_sent_num (int) : number of sentences to be summary

    Returns:
        new_arts (list[str]) : a list of new articles
        new_summs (list[str]) : a list of new summaries
    """
    art_sent_num = len(art_sents)
    max_out_id_start = art_sent_num - summ_sent_num

    new_arts = []
    new_summs = []
    if art_sent_num > summ_sent_num:
        for out_id_start in range(0, max_out_id_start + 1):
            out_ids = set(range(out_id_start, out_id_start + summ_sent_num))
            in_ids = set(range(art_sent_num)).difference(out_ids)
            new_arts.append(" ".join([art_sents[i] for i in sorted(in_ids)]))
            new_summs.append(" ".join([art_sents[i] for i in sorted(out_ids)]))
    else:
        # NOTE: use empty summary if no enough sentences to be extracted
        new_arts.append(" ".join(art_sents))
        new_summs.append(" ")

    return new_arts, new_summs


def lead_bias(art_sents, summ_sent_num, max_sent_len):
    """ Create leave-one-out articles and summaries.

    Args:
        art_sents (list[str]) : article sentences
        summ_sent_num (int) : number of sentences to be summary

    Returns:
        new_art (str) : new article
        new_summ (str) : new summary
    """
    art_sent_num = len(art_sents)

    if art_sent_num > summ_sent_num:

        # Tokenize each sentence
        art_sents_tokens = [nltk.word_tokenize(sent) for sent in art_sents]
        art_sents_tokens = [s[:max_sent_len] for s in art_sents_tokens]

        # Get ngrams for each sentence
        evaluated_1grams = [
            _get_word_ngrams(1, [sent]) for sent in art_sents_tokens
        ]
        evaluated_2grams = [
            _get_word_ngrams(2, [sent]) for sent in art_sents_tokens
        ]

        # Compute ROUGE-N for lead sentences and each article sentence
        lead_ids = set(range(summ_sent_num))
        lead_ids = sorted(lead_ids)
        reference_1 = [evaluated_1grams[i] for i in lead_ids]
        reference_1 = set.union(*map(set, reference_1))
        reference_2 = [evaluated_2grams[i] for i in lead_ids]
        reference_2 = set.union(*map(set, reference_2))

        out_ids = set(range(art_sent_num)).difference(lead_ids)
        out_ids = sorted(out_ids)
        scores = []
        for out_id in out_ids:
            candidates_1 = evaluated_1grams[out_id]
            candidates_2 = evaluated_2grams[out_id]
            rouge_1 = cal_rouge_n(candidates_1, reference_1)['f']
            rouge_2 = cal_rouge_n(candidates_2, reference_2)['f']
            scores.append((rouge_1 + rouge_2) / 2)

        # Use the top-k sentences as summary
        topk_ids = sorted(range(len(scores)),
                          key=lambda i: scores[i],
                          reverse=True)[:summ_sent_num]
        in_ids = set(range(art_sent_num)).difference(set(topk_ids))
        in_ids = sorted(in_ids)
        new_art = " ".join([art_sents[i] for i in in_ids])
        new_summ = " ".join([art_sents[i] for i in topk_ids])

    else:
        # NOTE: use empty summary if no enough sentences to be extracted
        new_art = " ".join(art_sents)
        new_summ = " "

    return new_art, new_summ


def unsupervised_dataset(file_path,
                         art_column,
                         summ_column,
                         max_samples=None,
                         max_sent_len=None,
                         summ_sent_num=1,
                         debug=False):
    file_dir, file_name = os.path.split(file_path)
    assert args.output_dir != file_dir

    # Read CSV
    df = pd.read_csv(file_path)
    if max_samples:
        df = df.iloc[:max_samples]

    # Create extractive oracle and add back to dataset
    print(f"Creating unsupervised dataset with conditions for {file_path}...")

    def _unsupervised_dataset(data):

        # NOTE: handle NaN input
        if not isinstance(data[art_column], str):
            data[art_column] = " "
        if not isinstance(data[summ_column], str):
            data[summ_column] = " "

        # Sentence tokenize
        art_sents = nltk.sent_tokenize(data[art_column])

        # Create new article and summary
        new_arts, new_summs = leave_one_out(art_sents, summ_sent_num)

        # Compute properties
        conditions = compute_conditions(art_sents, summ_sent_num, max_sent_len)

        # Create new data
        data[art_column] = new_arts
        data[summ_column] = new_summs
        data = pd.concat([data, pd.Series(conditions)])

        return data

    if debug:
        df = df.progress_apply(lambda d: _unsupervised_dataset(d), axis=1)
    else:
        df = df.parallel_apply(lambda d: _unsupervised_dataset(d), axis=1)
    df = df.apply(pd.Series.explode).reset_index(drop=True)

    # Output results as CSV
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, file_name), index=False)


def supervised_dataset(file_path,
                       art_column,
                       summ_column,
                       max_samples=None,
                       debug=False):
    file_dir, file_name = os.path.split(file_path)
    assert args.output_dir != file_dir

    # Read CSV
    df = pd.read_csv(file_path)
    if max_samples:
        df = df.iloc[:max_samples]

    # Create extractive oracle and add back to dataset
    print(f"Creating supervised dataset with conditions for {file_path}...")

    def _supervised_dataset(data):

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

        data['rouge_1'] = rouge_1
        data['rouge_2'] = rouge_2
        data['rouge_l'] = rouge_l
        data['compress_ratio'] = compress_ratio
        data['ext_coverage'] = ext_coverage
        data['ext_density'] = ext_density
        data['novel_word_ratio'] = novel_word_ratio

        return data

    if debug:
        df = df.progress_apply(lambda d: _supervised_dataset(d), axis=1)
    else:
        df = df.parallel_apply(lambda d: _supervised_dataset(d), axis=1)

    # Output results as CSV
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, file_name), index=False)


def lead_bias_dataset(file_path,
                      art_column,
                      summ_column,
                      max_samples=None,
                      max_sent_len=None,
                      summ_sent_num=1,
                      debug=False):
    file_dir, file_name = os.path.split(file_path)
    assert args.output_dir != file_dir

    # Read CSV
    df = pd.read_csv(file_path)
    if max_samples:
        df = df.iloc[:max_samples]

    # Create extractive oracle and add back to dataset
    print(f"Creating lead bias dataset with conditions for {file_path}...")

    def _lead_bias_dataset(data):

        # NOTE: handle NaN input
        if not isinstance(data[art_column], str):
            data[art_column] = " "
        if not isinstance(data[summ_column], str):
            data[summ_column] = " "

        # Sentence tokenize
        art_sents = nltk.sent_tokenize(data[art_column])

        # Create new article and summary
        new_art, new_summ = lead_bias(art_sents, summ_sent_num, max_sent_len)

        # Tokenize
        art_tokens = nltk.word_tokenize(new_art)
        summ_tokens = nltk.word_tokenize(new_summ)

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

        data[art_column] = new_art
        data[summ_column] = new_summ
        data['rouge_1'] = rouge_1
        data['rouge_2'] = rouge_2
        data['rouge_l'] = rouge_l
        data['compress_ratio'] = compress_ratio
        data['ext_coverage'] = ext_coverage
        data['ext_density'] = ext_density
        data['novel_word_ratio'] = novel_word_ratio

        return data

    if debug:
        df = df.progress_apply(lambda d: _lead_bias_dataset(d), axis=1)
    else:
        df = df.parallel_apply(lambda d: _lead_bias_dataset(d), axis=1)

    # Output results as CSV
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, file_name), index=False)


def main(args, art_column, summ_column):
    if args.use_ground_truth:
        supervised_dataset(args.train_file, art_column, summ_column,
                           args.max_train_samples, args.debug)
        #supervised_dataset(args.validation_file, art_column, summ_column,
        #                   args.max_val_samples, args.debug)
        #supervised_dataset(args.test_file, art_column, summ_column,
        #                   args.max_test_samples, args.debug)
        pass
    elif args.use_lead_bias:
        #lead_bias_dataset(args.train_file, art_column, summ_column,
        #                  args.max_train_samples, args.max_sent_len,
        #                  args.summ_sent_num, args.debug)
        #lead_bias_dataset(args.validation_file, art_column, summ_column,
        #                  args.max_val_samples, args.max_sent_len,
        #                  args.summ_sent_num, args.debug)
        lead_bias_dataset(args.test_file, art_column, summ_column,
                          args.max_test_samples, args.max_sent_len,
                          args.summ_sent_num, args.debug)
        pass
    else:
        unsupervised_dataset(args.train_file, art_column, summ_column,
                             args.max_train_samples, args.max_sent_len,
                             args.summ_sent_num, args.debug)
        #unsupervised_dataset(args.validation_file, art_column, summ_column,
        #                     args.max_val_samples, args.max_sent_len,
        #                     args.summ_sent_num, args.debug)
        #unsupervised_dataset(args.test_file, art_column, summ_column,
        #                     args.max_test_samples, args.max_sent_len,
        #                     args.summ_sent_num, args.debug)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract Extractive Oracle according to ROUGE')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--max_train_samples", type=int, default=1000000)
    parser.add_argument("--max_val_samples", type=int, default=1000000)
    parser.add_argument("--max_test_samples", type=int, default=1000000)
    parser.add_argument("--max_sent_len", type=int, default=128)
    parser.add_argument("--summ_sent_num", type=int, default=1)
    parser.add_argument("--use_ground_truth", action='store_true')
    parser.add_argument("--use_lead_bias", action='store_true')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    art_column, summ_column = summarization_name_mapping.get(args.dataset, None)

    main(args, art_column, summ_column)
