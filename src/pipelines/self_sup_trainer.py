""" Trainer for self-supervised transfer learning

The code is developed based on:
    https://github.com/huggingface/transformers/blob/v4.4.2/src/transformers/trainer.py
"""
import os
import time
import string
import shutil
from typing import Optional, Union, Dict, Any, Callable, Tuple, List
import numpy as np

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from transformers import Seq2SeqTrainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

from pipelines.base_trainer import BaseSeq2SeqTrainer
from models.conditions import ConditionalAdapter, ConditionalLN, ConditionalSwitchFF
from others.arguments import ModelArguments, DataTrainingArguments

from rouge import Rouge
import pdb

logger = logging.get_logger(__name__)


class SelfSupSeq2SeqTrainer(BaseSeq2SeqTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        model_args: ModelArguments = None,
        data_args: DataTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        BaseSeq2SeqTrainer.__init__(self, model, args, model_args, data_args,
                                    data_collator, train_dataset, eval_dataset,
                                    tokenizer, model_init, compute_metrics,
                                    callbacks, optimizers)

        # NEW: load cluster conditions
        if self.args.cluster_conditions_path:
            self.cluster_conditions = np.load(
                self.args.cluster_conditions_path)
            self.cluster_conditions = torch.tensor(self.cluster_conditions).to(
                self.model.device)
            self.cluster_conditions = self.cluster_conditions.type(
                self.model.lm_head.weight.dtype)
            self.rouge = Rouge()
        else:
            self.cluster_conditions = None

        if self.args.predict_with_loss_per_token:
            self.token_losses = []

        # NEW: adapter fusing
        if self.args.adapter_fusing_coeff:
            assert self.args.insert_conditional_adapters == True
            assert self.model_args.adapter_type == "sw"
            self.adapter_fusing(self.model)

    def adapter_fusing(self, model):
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, ConditionalSwitchFF):
                for param_main, param_side in zip(
                        sub_module.adapter_dict['main'].parameters(),
                        sub_module.adapter_dict['side'].parameters()):
                    param_main.data = param_main.data * (
                        1 - self.args.adapter_fusing_coeff
                    ) + param_side.data * self.args.adapter_fusing_coeff

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        """
        Modification:
            - distribute conditions to each conditional module.
        """

        # NEW: distribute conditions to each adapter
        if self.args.insert_conditional_adapters:
            self._distribute_conditions(model, inputs.pop("conditions"))

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _distribute_conditions(self, model, conditions):
        # Modules that use conditions
        condition_module_types = (ConditionalAdapter, ConditionalLN)

        # Set attribute for each model
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, condition_module_types):
                sub_module.conditions = conditions

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        """
        Modification:
            - distribute conditions to each conditional module.
            - preference-aware decoding
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # NEW: distribute conditions to each adapters
        if self.args.insert_conditional_adapters:
            self._distribute_conditions(model, inputs.pop("conditions"))

        gen_kwargs = {
            "max_length":
                self._max_length if self._max_length is not None else model.config.max_length,
            "num_beams":
                self._num_beams
                if self._num_beams is not None else model.config.num_beams,
            "output_scores":  # NEW
                self.args.predict_with_loss_per_token,
            "return_dict_in_generate":  # NEW
                self.args.predict_with_loss_per_token,
        }

        # NEW: generate with cluster conditions
        if self.cluster_conditions is not None:
            generated_tokens = self.generate_with_cluster_conditions(
                inputs["input_ids"],
                inputs["labels"],
                attention_mask=inputs["attention_mask"],
                gen_kwargs=gen_kwargs,
            )
        else:
            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        # NEW: for scores
        if self.args.predict_with_loss_per_token:
            generated_scores = generated_tokens["scores"]
            generated_tokens = generated_tokens["sequences"]

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"])

        # NEW: for scores
        if self.args.predict_with_loss_per_token:

            # Stack all tokens and compute probability
            generated_scores = torch.stack(generated_scores, dim=1)
            generated_probs = generated_scores.softmax(-1)

            # Pad labels to max length
            pad_labels = torch.nn.functional.pad(
                inputs['labels'],
                (0, gen_kwargs["max_length"] - inputs['labels'].shape[1]),
                value=-100)
            labels_mask = (pad_labels >= 0).int()

            # Pad scores to max length
            probs_shape = generated_probs.shape
            zeros = torch.zeros(probs_shape[0],
                                gen_kwargs["max_length"] - probs_shape[1],
                                probs_shape[2]).to(generated_probs.device)
            pad_probs = torch.cat([generated_probs, zeros], dim=1)

            # Gather NLL losses
            nll_loss = -torch.log(
                torch.gather(
                    pad_probs, 2,
                    (pad_labels * labels_mask).unsqueeze(-1)).squeeze(-1))
            self.token_losses += nll_loss[labels_mask.bool()].tolist()

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else
                            outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels,
                                                  gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def generate_with_cluster_conditions(self, input_ids, labels,
                                         attention_mask, gen_kwargs):

        generated_tokens_pool = []
        generated_scores_pool = []  # NEW
        scores_pool = []

        batch_size = input_ids.shape[0]
        for i in range(self.cluster_conditions.shape[0]):
            # Distribute specific conditions
            conditions = torch.cat(batch_size *
                                   [self.cluster_conditions[i].unsqueeze(0)])
            self._distribute_conditions(self.model, conditions)

            # Generation
            generated_tokens = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            # NEW : for scores
            if self.args.predict_with_loss_per_token:
                generated_scores = generated_tokens["scores"]
                generated_tokens = generated_tokens["sequences"]

            # Collect generated tokens
            generated_tokens_pool.append(generated_tokens)

            # NEW : for scores
            if self.args.predict_with_loss_per_token:
                generated_scores_pool.append(generated_scores)

            # Collect ROUGE scores for this generation
            gen_summ = self.tokenizer.batch_decode(generated_tokens,
                                                   skip_special_tokens=True)
            gold_summ = self.tokenizer.batch_decode(labels * (labels > 0),
                                                    skip_special_tokens=True)
            gen_summ = [s if s != "" else " " for s in gen_summ]
            gold_summ = [s if s != "" else " " for s in gold_summ]
            gen_summ = [
                s if not all([w in string.punctuation for w in s]) else " "
                for s in gen_summ
            ]
            gold_summ = [
                s if not all([w in string.punctuation for w in s]) else " "
                for s in gold_summ
            ]
            scores = self.rouge.get_scores(gen_summ, gold_summ)
            scores_pool.append([s['rouge-1']['f'] for s in scores])

        # Pad generated tokens to same length
        max_token_len = max([g.shape[1] for g in generated_tokens_pool])

        def _pad(x):
            return torch.nn.functional.pad(x,
                                           pad=(0, max_token_len - x.shape[1]))

        generated_tokens_pool = [_pad(g) for g in generated_tokens_pool]
        generated_tokens_pool = torch.cat(
            [g.unsqueeze(1) for g in generated_tokens_pool], dim=1)

        # Select best generation
        best_condition_indices = torch.tensor(
            np.argmax(np.array(scores_pool),
                      axis=0)).long().to(self.model.device)
        best_generated_tokens = []
        for g, i in zip(generated_tokens_pool, best_condition_indices):
            best_generated_tokens.append(g[i])
        generated_tokens = torch.cat(
            [g.unsqueeze(0) for g in best_generated_tokens])

        # NEW : for scores
        if self.args.predict_with_loss_per_token:

            # Stack each generated scores
            generated_scores_pool = [
                torch.stack(s, dim=1) for s in generated_scores_pool
            ]

            # Pad each generated scores to shared max length
            max_score_len = max([s.shape[1] for s in generated_scores_pool])

            def _pad(x):
                pad_zeros = torch.zeros(x.shape[0], max_score_len - x.shape[1],
                                        x.shape[2]).to(x.device)
                return torch.cat([x, pad_zeros], dim=1)

            generated_scores_pool = [_pad(s) for s in generated_scores_pool]

            # Stack generated scores over conditions
            generated_scores_pool = torch.stack(generated_scores_pool, dim=1)

            # Pick best condition for each data
            best_generated_scores = []
            for s, i in zip(generated_scores_pool, best_condition_indices):
                best_generated_scores.append(s[i])
            best_generated_scores = torch.stack(best_generated_scores, dim=0)

            # Separate tokens in to tuple
            generated_scores = tuple([
                best_generated_scores[:, i, :]
                for i in range(best_generated_scores.shape[1])
            ])

        # NEW : for scores
        if self.args.predict_with_loss_per_token:
            return {"sequences": generated_tokens, "scores": generated_scores}
        else:
            return generated_tokens
