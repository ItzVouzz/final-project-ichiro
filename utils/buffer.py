# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch
import numpy as np


class ExperienceBuffer:

    def __init__(self, horizon_length, num_envs, device):
        self.tensor_dict = {}
        self.horizon_length = horizon_length
        self.num_envs = num_envs
        self.device = device

    def add_buffer(self, name, shape, dtype=None):
        self.tensor_dict[name] = torch.zeros(self.horizon_length, self.num_envs, *shape, dtype=dtype, device=self.device)

    def update_data(self, name, idx, data):
        self.tensor_dict[name][idx, :] = data

    def __len__(self):
        return len(self.tensor_dict)

    def __getitem__(self, buf_name):
        return self.tensor_dict[buf_name]

    def keys(self):
        return self.tensor_dict.keys()


class ReplayBuffer:

    def __init__(self, obs_dim, buffer_size, device):
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states, next_states):
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step : self.buffer_size] = states[: self.buffer_size - self.step]
            self.next_states[self.step : self.buffer_size] = next_states[: self.buffer_size - self.step]
            self.states[: end_idx - self.buffer_size] = states[self.buffer_size - self.step :]
            self.next_states[: end_idx - self.buffer_size] = next_states[self.buffer_size - self.step :]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device), self.next_states[sample_idxs].to(self.device))
