""" Builds trainer. """
import nltk
import numpy as np

from datasets import load_metric
from transformers import DataCollatorForSeq2Seq

from .self_sup_trainer import SelfSupSeq2SeqTrainer
from .meta_trainer import MetaSeq2SeqTrainer
import pdb


def build_trainer(model_args, data_args, training_args, model, tokenizer,
                  train_dataset, eval_dataset):
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels,
                                use_stemmer=True)

        # Extract a few results from ROUGE
        result = {
            key: value.mid.fmeasure * 100
            for key, value in result.items()
        }

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    if training_args.meta_mode:
        trainer = MetaSeq2SeqTrainer(
            model=model,
            args=training_args,
            model_args=model_args,  # NEW: for record
            data_args=data_args,  #NEW: for record
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
            if training_args.predict_with_generate else None,
        )
    else:
        trainer = SelfSupSeq2SeqTrainer(
            model=model,
            args=training_args,
            model_args=model_args,  # NEW: for record
            data_args=data_args,  #NEW: for record
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
            if training_args.predict_with_generate else None,
        )

    return trainer
