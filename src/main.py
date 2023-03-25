""" Self-Supervised / Meta-Transfer Learning for Low Resource Abstractive Summarization.

The code is developed based on :
    https://github.com/huggingface/transformers/tree/v4.4.2/examples/seq2seq
"""

import logging
import os
import sys

import torch
import transformers
from transformers import (CONFIG_MAPPING, AutoConfig, AutoModelForSeq2SeqLM,
                          AutoTokenizer, HfArgumentParser, set_seed)
from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version

from data.build_datasets import build_datasets
from pipelines.build_trainer import build_trainer
from others.arguments import ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments
from processes import train_process, eval_process, predict_process

from models.conditions import insert_conditional_adapters

import pdb

check_min_version("4.4.2")
logger = logging.getLogger(__name__)


def main():

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,
                               CustomSeq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set the verbosity to warning of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_warning()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.WARN if is_main_process(training_args.local_rank
                                                    ) else logging.WARN)
    logger.info(f"Training/evaluation parameters {training_args}")

    # Log on each process the small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, " +
        f"device: {training_args.device}, " +
        f"n_gpu: {training_args.n_gpu}, " +
        f"distributed training: {bool(training_args.local_rank != -1)}, " +
        f"16-bits training: {training_args.fp16}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # NOTE: [For T5] need to assign prefix for summarization
    if data_args.source_prefix is None and model_args.model_name_or_path in [
            "t5-small",
            "t5-base",
            "t5-large",
            "t5-3b",
            "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, "
            "which is the expected, e.g. with `--source_prefix 'summarize: ' `"
        )

    # Load config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,
                                            **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    # NOTE: [For BART-large] not sure why
    if 'bart-large' in model_args.model_name_or_path:
        config.forced_bos_token_id = config.bos_token_id

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name,
                                                  **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. "
            "This is not supported by this scripts. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load model
    if model_args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.warning("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    if training_args.label_smoothing_factor > 0 and not hasattr(
            model, "prepare_decoder_input_ids_from_labels"):
        logger.warn(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    # NEW: Load trained model before or after architecture modification
    if model_args.load_model_before_modification:
        if model_args.load_trained_model_from:
            logger.warning(
                f"Load trained model from {model_args.load_trained_model_from}..."
            )
            ckpt_path = os.path.join(model_args.load_trained_model_from,
                                     "pytorch_model.bin")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)

        if training_args.insert_conditional_adapters:
            logger.warning("Insert conditional adapters...")
            model = insert_conditional_adapters(model_args, data_args, model)
    else:
        if training_args.insert_conditional_adapters:
            logger.warning("Insert conditional adapters...")
            model = insert_conditional_adapters(model_args, data_args, model)

        if model_args.load_trained_model_from:
            logger.warning(
                f"Load trained model from {model_args.load_trained_model_from}..."
            )
            ckpt_path = os.path.join(model_args.load_trained_model_from,
                                     "pytorch_model.bin")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)

    # Build datasets
    train_dataset, eval_dataset, test_dataset = build_datasets(
        data_args, training_args, tokenizer)

    # Build trainer
    trainer = build_trainer(model_args, data_args, training_args, model,
                            tokenizer, train_dataset, eval_dataset)

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        train_process(model_args, data_args, training_args, trainer,
                      train_dataset)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluation ***")
        eval_process(data_args, trainer, eval_dataset)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Prediction ***")
        predict_process(data_args, training_args, trainer, test_dataset,
                        tokenizer)


if __name__ == "__main__":
    main()
