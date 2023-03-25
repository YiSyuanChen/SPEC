""" Trainer for meta-transfer learning

The code is developed based on:
    https://github.com/huggingface/transformers/blob/v4.4.2/src/transformers/trainer.py
"""
import os
import time
import string
import shutil
import collections
from typing import Optional, Union, Dict, Any, Callable, Tuple, List
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data.dataset import Dataset

from transformers import Seq2SeqTrainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

from pipelines.self_sup_trainer import SelfSupSeq2SeqTrainer
from models.conditions import ConditionalSwitchFF, ConditionalSplitFF
from data.sampler import PerTaskPerBatchSampler, MixTaskPerBatchSampler
from others.arguments import ModelArguments, DataTrainingArguments

import higher

from rouge import Rouge
import pdb

logger = logging.get_logger(__name__)


class MetaSeq2SeqTrainer(SelfSupSeq2SeqTrainer):
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
        SelfSupSeq2SeqTrainer.__init__(self, model, args, model_args,
                                       data_args, data_collator, train_dataset,
                                       eval_dataset, tokenizer, model_init,
                                       compute_metrics, callbacks, optimizers)

        # NEW: Initiali optimizer for inner loop
        self.inner_optimizer = optim.SGD(self.model.parameters(),
                                         lr=self.args.inner_learning_rate,
                                         momentum=0.9)

        # NOTE: sanity check
        assert self.args.batch_group_method == 'per_task'
        assert self.args.main_adapter_task_ids is not None
        assert len(self.args.main_adapter_task_ids) == len(
            self.data_args.meta_dataset_names) - 1

        # NEW: multi-task balancing
        if self.args.multi_task_balancing:
            self.task_num = len(self.data_args.meta_dataset_names)
            self.task_pre_loss_dict = {task_id: [] for task_id in range(self.task_num)}

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed for current input task.

        - If using switch adapters, then the main adapters will be trained with meta loss,
          and other adapters with full model will be trained with typical loss.
        - If not using switch adapters, then model will be trained with meta loss according
          to trainable_params_config.
        - For evaluation, main adapters will be used, and meta loss will be reported.
        """
        # NOTE: return_outputs will be True in validation
        is_eval = return_outputs
        if not is_eval:
            task_id = inputs.pop('task_id')
            task_id = torch.unique(task_id).tolist()
            assert len(task_id)
            task_id = task_id[0]

            if self.model_args.adapter_type in ("sw", "sp"):
                if task_id in self.args.main_adapter_task_ids:
                    self._freeze_all_params(model)
                    self._unfreeze_specified_params(
                        model, force_config='adapter_no_ln')
                    self._set_adapter_flag(model, 'main')
                    results = self.compute_meta_loss(model, inputs,
                                                     return_outputs)
                else:
                    self._freeze_all_params(model)
                    self._unfreeze_specified_params(model,
                                                    force_config='full_model')
                    self._set_adapter_flag(model, 'side')
                    results = self.compute_normal_loss(model, inputs,
                                                       return_outputs)
            else:
                results = self.compute_meta_loss(model, inputs, return_outputs)

            # NEW: multi-task balancing
            if self.args.multi_task_balancing:
                # Record current loss
                task_pre_loss = self.task_pre_loss_dict[task_id]
                task_pre_loss.insert(0, results.item())
                if len(task_pre_loss) > 2:
                    task_pre_loss.pop(-1)

                # Compute loss weighting
                if all(map(lambda x: len(x) == self.task_num, self.task_pre_loss_dict.values())):
                    loss_ratio = [v[0]/v[1] for v in self.task_pre_loss_dict.values()]
                    loss_ratio_exp = torch.exp(torch.tensor(loss_ratio))
                    loss_weight = (loss_ratio_exp[task_id] / loss_ratio_exp.sum()) * self.task_num
                    results *= loss_weight

        else:
            if self.model_args.adapter_type in ("sw", "sp"):
                self._set_adapter_flag(model, 'main')
            results = self.compute_meta_loss(model, inputs, return_outputs)

        return results

    def compute_normal_loss(self, model, inputs, return_outputs=False):
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

    def compute_meta_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        """
        Modification:
            - distribute conditions to each conditional module.
        """
        # NOTE: return_outputs will be True in validation
        is_eval = return_outputs

        ##### Support / Query Split #####
        s_inputs = {}
        q_inputs = {}
        for key, value in inputs.items():
            split_point = int(len(value) / 2)
            s_inputs[key] = value[:split_point]
            q_inputs[key] = value[split_point:]

        with higher.innerloop_ctx(model,
                                  self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=not is_eval) as (fmodel,
                                                                      diffopt):

            ########## Inner Loop Training ##########

            # NEW: distribute conditions to each adapter
            if self.args.insert_conditional_adapters:
                self._distribute_conditions(fmodel, s_inputs.pop("conditions"))

            if self.label_smoother is not None and "labels" in s_inputs:
                labels = s_inputs.pop("labels")
            else:
                labels = None

            for _ in range(1):

                # Forward
                outputs = fmodel(**s_inputs)

                if labels is not None:
                    loss = self.label_smoother(outputs, labels)
                else:
                    # We don't use .loss here since the model may return tuples instead of ModelOutput.
                    loss = outputs["loss"] if isinstance(outputs,
                                                         dict) else outputs[0]
                # High-level backward
                diffopt.step(loss)

            # NEW: use ANIL
            if self.args.ANIL:
                for param in fmodel.parameters():
                    if param.requires_grad == False:
                        param.requires_grad=True

            ########## Outer Loop Training ##########

            # NEW: distribute conditions to each adapter
            if self.args.insert_conditional_adapters:
                self._distribute_conditions(fmodel, q_inputs.pop("conditions"))

            if self.label_smoother is not None and "labels" in q_inputs:
                labels = q_inputs.pop("labels")
            else:
                labels = None

            outputs = fmodel(**q_inputs)

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs,
                                                     dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

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
            - use fast model to generate
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # NEW: set to main adapter
        self._freeze_all_params(model)
        self._unfreeze_specified_params(model, force_config='full_model')
        if self.model_args.adapter_type in ("sw", "sp"):
            self._set_adapter_flag(model, 'main')

        # NEW: use trained fast model to generate
        ##### Support / Query Split #####
        s_inputs = {}
        q_inputs = {}
        for key, value in inputs.items():
            split_point = int(len(value) / 2)
            s_inputs[key] = value[:split_point]
            q_inputs[key] = value[split_point:]

        with higher.innerloop_ctx(model,
                                  self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=False) as (fmodel,
                                                                diffopt):
            ########## Inner Loop Training ##########

            # NEW: distribute conditions to each adapter
            if self.args.insert_conditional_adapters:
                self._distribute_conditions(fmodel, s_inputs.pop("conditions"))

            if self.label_smoother is not None and "labels" in s_inputs:
                labels = s_inputs.pop("labels")
            else:
                labels = None

            for _ in range(1):

                # Forward
                outputs = fmodel(**s_inputs)

                if labels is not None:
                    loss = self.label_smoother(outputs, labels)
                else:
                    # We don't use .loss here since the model may return tuples instead of ModelOutput.
                    loss = outputs["loss"] if isinstance(outputs,
                                                         dict) else outputs[0]
                # High-level backward
                diffopt.step(loss)

            ########## Outer Loop Inference ##########

            # NEW: distribute conditions to each adapters
            if self.args.insert_conditional_adapters:
                self._distribute_conditions(fmodel, q_inputs.pop("conditions"))

            gen_kwargs = {
                "max_length":
                    self._max_length if self._max_length is not None else
                    fmodel.config.max_length,
                "num_beams":
                    self._num_beams
                    if self._num_beams is not None else fmodel.config.num_beams,
                "output_scores":  # NEW
                    self.args.predict_with_loss_per_token,
                "return_dict_in_generate":  # NEW
                    self.args.predict_with_loss_per_token,
            }

            # NEW: generate with cluster conditions
            if self.cluster_conditions is not None:
                generated_tokens = self.generate_with_cluster_conditions(
                    fmodel,
                    q_inputs["input_ids"],
                    q_inputs["labels"],
                    attention_mask=q_inputs["attention_mask"],
                    gen_kwargs=gen_kwargs,
                )
            else:
                generated_tokens = fmodel.generate(
                    q_inputs["input_ids"],
                    attention_mask=q_inputs["attention_mask"],
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
                    q_inputs['labels'],
                    (0,
                     gen_kwargs["max_length"] - q_inputs['labels'].shape[1]),
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
                        outputs = fmodel(**q_inputs)
                else:
                    outputs = fmodel(**q_inputs)
                if has_labels:
                    if self.label_smoother is not None:
                        loss = self.label_smoother(
                            outputs, q_inputs["labels"]).mean().detach()
                    else:
                        loss = (outputs["loss"] if isinstance(outputs, dict)
                                else outputs[0]).mean().detach()
                else:
                    loss = None

            if self.args.prediction_loss_only:
                return (loss, None, None)

            labels = q_inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels,
                                                      gen_kwargs["max_length"])

            # NOTE: repeat the result twice since we only use half input
            #       this is a trick to avoid error from huggingface
            generated_tokens = torch.cat(2 * [generated_tokens], dim=0)
            labels = torch.cat(2 * [labels], dim=0)

        return (loss, generated_tokens, labels)

    def _set_adapter_flag(self, model, adapter_flag):
        # Modules that use conditions
        condition_module_types = (ConditionalSwitchFF, ConditionalSplitFF)

        # Set attribute for each model
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, condition_module_types):
                sub_module.adapter_flag = adapter_flag

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        """
        Modification:
            - add new sampler options
        """
        if isinstance(self.train_dataset,
                      torch.utils.data.IterableDataset) or not isinstance(
                          self.train_dataset, collections.abc.Sized):
            return None

        # Build the sampler.
        if self.args.batch_group_method:  # NEW: make batch according to task
            model_input_name = self.tokenizer.model_input_names[
                0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                if self.args.batch_group_method == "mix_task":
                    return MixTaskPerBatchSampler(self.train_dataset,
                                                  self.args.train_batch_size)
                elif self.args.batch_group_method == "per_task":
                    return PerTaskPerBatchSampler(self.train_dataset,
                                                  self.args.train_batch_size)
                else:
                    raise ValueError("Invalid batch group method.")
            else:
                raise "Sampler has no implementation for multi-device setting."

        elif self.args.group_by_length:
            model_input_name = self.tokenizer.model_input_names[
                0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(self.train_dataset,
                                            self.args.train_batch_size,
                                            model_input_name=model_input_name)
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    model_input_name=model_input_name,
                )

        else:
            if self.args.world_size <= 1:
                return RandomSampler(self.train_dataset)
            elif self.args.parallel_mode == ParallelMode.TPU and not self.args.dataloader_drop_last:
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                )
            else:
                return DistributedSampler(self.train_dataset,
                                          num_replicas=self.args.world_size,
                                          rank=self.args.process_index)
