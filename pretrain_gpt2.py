# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT2"""

import torch

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt2_dataset import build_train_valid_test_datasets
from megatron.model import GPT2Model
from megatron.training import pretrain, on_demand_checkpoint
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import reduce_losses

import signal

def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_tokentypes=0, parallel_output=True)

    return model


def get_batch(data_iterator, device=None):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype, device=device)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)


    if args.varuna:
        inputs = dict({
            "input_ids": tokens,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "labels": labels
        })
        return inputs
    
    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    print(f'prepare to get batch')
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch generator').stop()
    # Forward model.
    print(f'prepare to get losses')
    losses = model(tokens, position_ids, attention_mask, labels=labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}

def varuna_step(data_iterator, model):

    args = get_args()
    timers = get_timers()

    # Get the batch.
    print(f'varuna prepare to get batch')
    timers('batch generator').start()
    inputs = get_batch(data_iterator)
    timers('batch generator').stop()
    
    print('enter varuna_step')

    # if torch.distributed.get_rank() == 0:
    #     print(inputs["input_ids"])
    print(f'varuna prepare to get losses input size: {len(inputs)}')
    loss, overflow, global_norm = model.step(inputs)
    loss = torch.Tensor([loss]).cuda()
    # Reduce loss for logging.
    # reduced_loss = reduce_losses([loss])
    print(f'varuna finish a step')

    return loss, {'lm loss': loss}, overflow, global_norm

def varuna_evaluate(data_iterator, model):
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    inputs = get_batch(data_iterator)
    timers('batch generator').stop()

    loss = model.evaluate(inputs)
    
    # Reduce loss for logging.
    loss = torch.Tensor([loss]).cuda()
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT2 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT2 datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    print("hit main")
    
    def handler(signum,_):
        print(torch.distributed.get_rank(), 'signal handler called with signal', signum)
        on_demand_checkpoint()
        exit()

    signal.signal(signal.SIGUSR1, handler)

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step, 
            get_batch=get_batch, varuna_step_func=varuna_step, varuna_eval_func=varuna_evaluate,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
