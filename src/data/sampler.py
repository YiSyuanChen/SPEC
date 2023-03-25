""" Samplers for meta-learning.

This code is based on Pytorch Sampler class:
    https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#Sampler
"""
from itertools import cycle

import torch
from torch.utils.data.sampler import Sampler
import pdb


class MixTaskPerBatchSampler(Sampler):
    """ Sample data uniformly from each dataset. """
    def __init__(
        self,
        dataset,
        batch_size,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.task_ids = dataset['task_id']

    def __len__(self):
        return len(self.task_ids)

    def __iter__(self):
        indices = get_mix_task_per_batch_indices(self.task_ids,
                                                 self.batch_size)
        return iter(indices)


def get_mix_task_per_batch_indices(task_ids, batch_size, generator=None):
    """ Generate data indices with unifrom task sampling"""

    # Get the number of datasets
    num_task = len(set(task_ids))

    # Collect data indices for each task
    task_data_ids = [[i for i, value in enumerate(task_ids) if value == x]
                     for x in range(num_task)]

    # Oversample when dataset sizes are imbalance
    num_data = max([len(d) for d in task_data_ids]) * len(task_data_ids)

    # Shuffle data indices in each task and make them cyclic
    for task_id in range(num_task):
        rands = torch.randperm(len(task_data_ids[task_id]),
                               generator=generator)
        task_data_ids[task_id] = [task_data_ids[task_id][i] for i in rands]
    task_data_ids = [cycle(ids) for ids in task_data_ids]

    indices = []
    while len(indices) != num_data:
        # Randomly choose a task
        select_task_id = int(
            torch.randint(len(task_data_ids), (1, ), generator=generator))

        # Collect data from selected task
        indices.append(next(task_data_ids[select_task_id]))

    return indices


class PerTaskPerBatchSampler(Sampler):
    """ Sample data uniformly from each dataset. """
    def __init__(
        self,
        dataset,
        batch_size,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.task_ids = dataset['task_id']

    def __len__(self):
        return len(self.task_ids)

    def __iter__(self):
        indices = get_per_task_per_batch_indices(self.task_ids,
                                                 self.batch_size)
        return iter(indices)


def get_per_task_per_batch_indices(task_ids, batch_size, generator=None):
    """ Generate data indices with unifrom task sampling"""

    # Get the number of datasets
    num_task = len(set(task_ids))

    # Collect data indices for each task
    task_data_ids = [[i for i, value in enumerate(task_ids) if value == x]
                     for x in range(num_task)]

    # Oversample when dataset sizes are imbalance
    num_data = max([len(d) for d in task_data_ids]) * len(task_data_ids)

    # Shuffle data indices in each task and make them cyclic
    for task_id in range(num_task):
        rands = torch.randperm(len(task_data_ids[task_id]),
                               generator=generator)
        task_data_ids[task_id] = [task_data_ids[task_id][i] for i in rands]
    task_data_ids = [cycle(ids) for ids in task_data_ids]

    indices = []
    cur_task_id = 0
    cur_batch_size = 0
    while len(indices) != num_data:

        # Same task until making a complete batch
        select_task_id = cur_task_id

        # Collect data from selected task
        indices.append(next(task_data_ids[select_task_id]))

        # Change task when making a complete batch
        cur_batch_size += 1
        if cur_batch_size == batch_size:
            cur_batch_size = 0
            cur_task_id = (cur_task_id + 1) % num_task

    return indices
