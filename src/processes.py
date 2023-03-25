""" Defines different operation processes. """
import os
import numpy as np
import pdb


def train_process(model_args, data_args, training_args, trainer,
                  train_dataset):
    # Detect last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir
    ) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(
                training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    max_train_samples = (data_args.max_train_samples
                         if data_args.max_train_samples is not None else
                         len(train_dataset))
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def eval_process(data_args, trainer, eval_dataset):
    metrics = trainer.evaluate(max_length=data_args.val_max_target_length,
                               num_beams=data_args.num_beams,
                               metric_key_prefix="eval")
    max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
        eval_dataset)
    metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict_process(data_args, training_args, trainer, test_dataset,
                    tokenizer):

    test_results = trainer.predict(
        test_dataset,
        metric_key_prefix="test",
        max_length=data_args.val_max_target_length,
        num_beams=data_args.num_beams,
    )

    # NEW
    if training_args.predict_with_loss_per_token:
        np.save(
            os.path.join(training_args.output_dir, "test_loss_per_token.npy"),
            np.array(trainer.token_losses))

    metrics = test_results.metrics
    max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(
        test_dataset)
    metrics["test_samples"] = min(max_test_samples, len(test_dataset))

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate and not training_args.prediction_loss_only:
            # Output generation results
            test_preds = tokenizer.batch_decode(
                test_results.predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True)
            test_preds = [
                pred.strip().replace("\n", " ") for pred in test_preds
            ]
            output_test_preds_file = os.path.join(training_args.output_dir,
                                                  "test_generations.txt")
            with open(output_test_preds_file, "w") as writer:
                writer.write("\n".join(test_preds))

            # Also output the ground truth
            test_golds = tokenizer.batch_decode(
                test_dataset['labels'],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True)
            test_golds = [
                gold.strip().replace("\n", " ") for gold in test_golds
            ]
            output_test_golds_file = os.path.join(training_args.output_dir,
                                                  "test_golds.txt")
            with open(output_test_golds_file, "w") as writer:
                writer.write("\n".join(test_golds))
