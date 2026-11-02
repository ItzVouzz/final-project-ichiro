# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import os
import torch


class ActorCritic(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.critics = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.cfg["env"]["num_critic_obs"], 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ELU(),
                    torch.nn.Linear(128, 1),
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(self.cfg["env"]["num_critic_obs"], 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ELU(),
                    torch.nn.Linear(128, 1),
                ),
            ]
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(
                self.cfg["env"]["num_observations"] + self.cfg["algorithm"]["history_embedding_dim"],
                256,
            ),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.cfg["env"]["num_actions"]),
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.cfg["env"]["num_observations"] * self.cfg["runner"]["num_stack"], 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.cfg["algorithm"]["history_embedding_dim"]),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.cfg["algorithm"]["history_embedding_dim"], 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.cfg["env"]["num_privileged_obs"]),
        )
        self.logstd = torch.nn.Parameter(torch.full((self.cfg["env"]["num_actions"],), fill_value=-2.0))

    def act(self, obs, stacked_obs):
        embedding = self.encoder(stacked_obs.flatten(start_dim=-2))
        action_mean = self.actor(torch.cat((obs, embedding), dim=-1))
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        privileged_obs_est = self.decoder(embedding)
        return dist, privileged_obs_est

    def est_values(self, critic_obs):
        return torch.stack([critic(critic_obs).squeeze(-1) for critic in self.critics], dim=-1)

    def forward(self, obs, stacked_obs):
        embedding = self.encoder(stacked_obs.flatten(start_dim=-2))
        return self.actor(torch.cat((obs, embedding), dim=-1))


class Discriminator(torch.nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(0.2),
        )
        self.amp_linear = torch.nn.Linear(512, 1)

    def forward(self, x):
        return self.amp_linear(self.trunk(x))


class Normalizer:

    def __init__(self, input_dim, device):
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.mean = torch.zeros(input_dim, dtype=torch.float, device=device)
        self.var = torch.ones(input_dim, dtype=torch.float, device=device)
        self.count = 0

    def update(self, input):
        batch_mean = torch.mean(input, dim=0)
        batch_var = torch.var(input, dim=0, unbiased=False)
        batch_count = input.shape[0] * self.world_size
        if self.world_size > 1:
            torch.distributed.all_reduce(batch_mean, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(batch_var, op=torch.distributed.ReduceOp.AVG)
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + torch.square(batch_mean - self.mean) * self.count * batch_count / (self.count + batch_count)
        ) / (self.count + batch_count)
        self.mean += (batch_mean - self.mean) * batch_count / (self.count + batch_count)
        self.count += batch_count

    def normalize(self, input):
        return torch.clamp((input - self.mean) / (self.var + 1e-4), -10.0, 10.0)
