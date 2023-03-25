""" Set dataset mappings. """
import os

summarization_name_mapping = {
    "aeslc": ("email_body", "subject_line"),
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "billsum": ("text", "summary"),
    "cnn_dailymail": ("article", "highlights"),
    "csebuetnlp/xlsum": ("text", "summary"),
    "gigaword": ("document", "summary"),
    "idebate_own": ("_argument_sentences", "_claim"),  # own files
    "msr_text_compression": ("source_text", "targets"),
    "multi_news": ("document", "summary"),
    "newsroom": ("text", "summary"),  # require manual files 
    "reddit": ("content", "summary"),
    "reddit_tifu": ("documents", "tldr"),
    "rottentomatoes_own": ("_critics", "_critic_consensus"),  # own files
    "samsum": ("dialogue", "summary"),
    "scientific_papers": ("article", "abstract"),
    "scitldr": ("source", "target"),
    "wikihow": ("text", "headline"),  # require manual files
    "xsum": ("document", "summary"),
    "xsum_wiki_own": ("article", "summary"),  # own files
    "cnn_dailymail_wiki_own": ("article", "summary"),  # own files
    "reddit_tifu_wiki_own": ("article", "summary"),  # own files
    "scitldr_except_first_targets_own": ("source", "target"),
    "xsum_example_own": ("document", "summary"),
}

summarization_length_mapping = {
    "aeslc": (1024, 32),
    "amazon_reviews_multi": (1024, 64),
    "big_patent": (1024, 256),
    "billsum": (1024, 256),
    "cnn_dailymail": (1024, 128),
    "csebuetnlp/xlsum": (1024, 64),
    "gigaword": (1024, 32),
    "idebate_own": (1024, 64),
    "msr_text_compression": (1024, 64),
    "multi_news": (1024, 256),
    "newsroom": (1024, 128),
    "reddit": (1024, 128),
    "reddit_tifu": (1024, 128),
    "rottentomatoes_own": (1024, 64),
    "samsum": (1024, 64),
    "scientific_papers": (1024, 256),
    "scitldr": (1024, 64),
    "wikihow": (1024, 256),
    "xsum": (1024, 64),
    "xsum_wiki_own": (1024, 64),
    "cnn_dailymail_wiki_own": (1024, 128),
    "reddit_tifu_wiki_own": (1024, 128),
    "scitldr_except_first_targets_own": (1024, 64),
    "xsum_example_own": (1024, 64),
}


def register_datasets():
    assert list(summarization_name_mapping.keys()) == list(
        summarization_length_mapping.keys())
    dataset_name_list = list(summarization_name_mapping.keys())

    summarization_own_file_mapping = {}
    for dataset_name in dataset_name_list:
        # Find the original name of dataset
        true_name = dataset_name
        if '/' in dataset_name:
            true_name = dataset_name.split("/")[1]
        if '_own' in dataset_name:
            true_name = dataset_name.replace("_own", "")

        # Add cond and unsup version for the dataset
        summarization_name_mapping[
            true_name + '_cond_own'] = summarization_name_mapping[dataset_name]
        summarization_name_mapping[
            true_name +
            '_unsup_own'] = summarization_name_mapping[dataset_name]
        summarization_length_mapping[
            true_name +
            '_cond_own'] = summarization_length_mapping[dataset_name]
        summarization_length_mapping[
            true_name +
            '_unsup_own'] = summarization_length_mapping[dataset_name]
        summarization_own_file_mapping[true_name + '_cond_own'] = os.path.join(
            "../../ConditionDataset/datasets/supervised", true_name)
        summarization_own_file_mapping[
            true_name + '_unsup_own'] = os.path.join(
                "../../ConditionDataset/datasets/unsupervised", true_name)

        if '_own' in dataset_name:
            if true_name == 'rottentomatoes' or true_name == 'idebate':
                summarization_own_file_mapping[dataset_name] = os.path.join(
                    "../../../OpinionAbstracts/datasets", true_name)
            if '_wiki' in dataset_name:
                summarization_own_file_mapping[dataset_name] = os.path.join(
                    "../../../WikiSumm/datasets/reformated",
                    true_name.replace("_wiki", ""))
            if '_example' in dataset_name:
                summarization_own_file_mapping[dataset_name] = os.path.join(
                    "../analysis/examples",
                    true_name.replace("_example", ""))


    return summarization_name_mapping, summarization_length_mapping, \
           summarization_own_file_mapping


summarization_info = register_datasets()
summarization_name_mapping = summarization_info[0]
summarization_length_mapping = summarization_info[1]
summarization_own_file_mapping = summarization_info[2]
