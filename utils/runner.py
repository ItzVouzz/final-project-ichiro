# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio
import torch
import torch.nn.functional as F
from utils.wrapper import StackObservationWrapper
from utils.model import *
from utils.buffer import *
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder
from utils.motion import Motion
from envs import *


class Runner:

    def __init__(self):
        # prepare the environment
        self._get_args()
        self._update_cfg_from_args()
        self._configure_multi_gpu()
        self._set_seed()
        task_class = task_registry.get_task_class(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)
        self.env = StackObservationWrapper(self.env, self.cfg["runner"]["num_stack"])
        self.device = self.cfg["basic"]["device"]

        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.model = ActorCritic(self.cfg).to(self.device)
        self.motion_sampler = Motion(self.cfg, self.device)
        self.amp_normalizer = Normalizer(self.cfg["env"]["num_amp_obs"], self.device)
        self.discriminator = Discriminator(self.cfg["env"]["num_amp_obs"] * 2).to(self.device)
        self.amp_storage = ReplayBuffer(self.cfg["env"]["num_amp_obs"], self.cfg["motion"]["replay_buffer_size"], self.device)
        self.optimizer = torch.optim.Adam([{"name": "actor_critic", "params": self.model.parameters(), "lr": self.learning_rate}])
        self.discriminator_optimizer = torch.optim.RMSprop(
            [
                {
                    "name": "amp_trunk",
                    "params": self.discriminator.trunk.parameters(),
                    "weight_decay": 1e-2,
                    "lr": self.cfg["algorithm"]["discriminator_lr"],
                },
                {
                    "name": "amp_head",
                    "params": self.discriminator.amp_linear.parameters(),
                    "weight_decay": 5e-1,
                    "lr": self.cfg["algorithm"]["discriminator_lr"],
                },
            ]
        )
        self._load_checkpoint()
        if self.world_size > 1:
            for p in self.model.parameters():
                if p.data is not None:
                    torch.distributed.broadcast(p.data, src=0)
            for p in self.model.buffers():
                if p.data is not None:
                    torch.distributed.broadcast(p.data, src=0)
            for p in self.discriminator.parameters():
                if p.data is not None:
                    torch.distributed.broadcast(p.data, src=0)
            for p in self.discriminator.buffers():
                if p.data is not None:
                    torch.distributed.broadcast(p.data, src=0)

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("critic_obses", (self.env.num_critic_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("stacked_obses", (self.env.num_stack, self.env.num_obs))
        self.buffer.add_buffer("mirrored_obses", (self.env.num_obs,))
        self.buffer.add_buffer("mirrored_stacked_obses", (self.env.num_stack, self.env.num_obs))
        self.buffer.add_buffer("rewards", (2,))
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument(
            "--headless", action="store_true", default=None, help="Run headless without creating a viewer window. Overrides config file if provided."
        )
        parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
        parser.add_argument("--device", type=str, help="Device for simulation and training. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
        parser.add_argument("--save_path", type=str, help="Save path of logs. Overrides config file if provided.")
        parser.add_argument("--motion_path", type=str, help="Path of motion files. Overrides config file if provided.")
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", task_registry.get_task_cfg(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                elif arg == "save_path":
                    self.cfg["runner"][arg] = getattr(self.args, arg)
                elif arg == "motion_path":
                    self.cfg["motion"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)

    def _configure_multi_gpu(self):
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        if self.world_size == 1:
            return
        self.cfg["basic"]["device"] = f"cuda:{self.rank}"
        torch.distributed.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load_checkpoint(self):
        if not self.cfg["basic"]["checkpoint"]:
            return
        if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
            self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)
        try:
            self.discriminator.load_state_dict(model_dict["discriminator"], strict=False)
        except Exception as e:
            print(f"Failed to load discriminator: {e}")

    def train(self):
        if self.rank == 0:
            self.recorder = Recorder(self.cfg)
        obs, infos = self.env.reset()
        critic_obs = infos["critic_obs"]
        privileged_obs = infos["privileged_obs"]
        stacked_obs = infos["stacked_obs"]
        amp_obs = infos["amp_obs"]
        for it in range(self.cfg["basic"]["max_iterations"]):
            # within horizon_length, env.step() is called with same act
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("critic_obses", n, critic_obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                self.buffer.update_data("stacked_obses", n, stacked_obs)
                self.buffer.update_data("mirrored_obses", n, self.env.mirror_obs(obs))
                self.buffer.update_data("mirrored_stacked_obses", n, self.env.mirror_obs(stacked_obs))
                with torch.no_grad():
                    dist, _ = self.model.act(obs, stacked_obs)
                    act = dist.sample()
                obs, rew, done, infos = self.env.step(act)
                critic_obs = infos["critic_obs"]
                privileged_obs = infos["privileged_obs"]
                stacked_obs = infos["stacked_obs"]
                last_amp_obs = infos["last_amp_obs"]
                self.amp_storage.insert(amp_obs, last_amp_obs)
                with torch.no_grad():
                    amp_obs = self.amp_normalizer.normalize(amp_obs)
                    last_amp_obs = self.amp_normalizer.normalize(last_amp_obs)
                    amp_rew = 1 + torch.tanh(0.4 * self.discriminator(torch.cat([amp_obs, last_amp_obs], dim=-1)).squeeze(-1))
                    reward = rew.clone()
                    reward[:, 1] += self.cfg["algorithm"]["amp_reward_coef"] * amp_rew
                amp_obs = infos["amp_obs"]
                self.buffer.update_data("actions", n, act)
                self.buffer.update_data("rewards", n, reward)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"])
                ep_info = {
                    "reward": reward.sum(dim=-1),
                    "success_rate": infos["success"],
                    "fall_rate": infos["fall"],
                }
                infos["rew_terms"].update({"amp": self.cfg["algorithm"]["amp_reward_coef"] * amp_rew})
                ep_info.update(infos["rew_terms"])
                if self.rank == 0:
                    self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))

            with torch.no_grad():
                old_dist, _ = self.model.act(self.buffer["obses"], self.buffer["stacked_obses"])
                old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

                values = self.model.est_values(self.buffer["critic_obses"])
                last_values = self.model.est_values(critic_obs)
                self.buffer["rewards"][self.buffer["time_outs"]] += self.cfg["algorithm"]["gamma"] * values[self.buffer["time_outs"]]
                advantages = discount_values(
                    self.buffer["rewards"],
                    self.buffer["dones"] | self.buffer["time_outs"],
                    values,
                    last_values,
                    self.cfg["algorithm"]["gamma"],
                    self.cfg["algorithm"]["lam"],
                )
                if self.world_size > 1:
                    all_advantages = [torch.empty_like(advantages) for _ in range(self.world_size)]
                    torch.distributed.all_gather(all_advantages, advantages)
                    all_advantages = torch.cat(all_advantages, dim=1)
                else:
                    all_advantages = advantages
                returns = values + advantages
                advantages = (advantages - all_advantages.mean(dim=(0, 1))) / (all_advantages.std(dim=(0, 1)) + 1e-8)

            statistics = {}

            mini_batch_size = (self.env.num_envs * self.cfg["runner"]["horizon_length"]) // self.cfg["algorithm"]["num_mini_batches"]
            indices = torch.randperm(mini_batch_size * self.cfg["algorithm"]["num_mini_batches"], requires_grad=False, device=self.device)
            amp_policy_generator = self.amp_storage.feed_forward_generator(
                self.cfg["algorithm"]["num_learning_epochs"] * self.cfg["algorithm"]["num_mini_batches"],
                mini_batch_size,
            )
            amp_expert_generator = self.motion_sampler.feed_forward_generator(
                self.cfg["algorithm"]["num_learning_epochs"] * self.cfg["algorithm"]["num_mini_batches"],
                mini_batch_size,
            )
            for n in range(self.cfg["algorithm"]["num_learning_epochs"]):
                for i in range(self.cfg["algorithm"]["num_mini_batches"]):
                    start = i * mini_batch_size
                    end = (i + 1) * mini_batch_size
                    batch_idx = indices[start:end]
                    dist, privileged_obs_est = self.model.act(
                        self.buffer["obses"].flatten(0, 1)[batch_idx], self.buffer["stacked_obses"].flatten(0, 1)[batch_idx]
                    )

                    values = self.model.est_values(self.buffer["critic_obses"].flatten(0, 1)[batch_idx])
                    value_loss = F.mse_loss(values, returns.flatten(0, 1)[batch_idx])

                    actions_log_prob = dist.log_prob(self.buffer["actions"].flatten(0, 1)[batch_idx]).sum(dim=-1)
                    actor_loss = surrogate_loss(
                        old_actions_log_prob.flatten(0, 1)[batch_idx],
                        actions_log_prob,
                        (advantages.flatten(0, 1)[batch_idx] * torch.tensor(self.cfg["algorithm"]["advantage_coef"], device=self.device)).sum(dim=-1),
                    )

                    policy_state, policy_next_state = next(amp_policy_generator)
                    expert_state, expert_next_state = next(amp_expert_generator)
                    with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize(policy_state)
                        policy_next_state = self.amp_normalizer.normalize(policy_next_state)
                        expert_state = self.amp_normalizer.normalize(expert_state)
                        expert_next_state = self.amp_normalizer.normalize(expert_next_state)
                    policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                    expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                    amp_loss = -torch.tanh(0.4 * expert_d).mean() + torch.tanh(0.4 * policy_d).mean()

                    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
                    policy_data = torch.cat([policy_state, policy_next_state], dim=-1)
                    alpha = torch.rand(expert_data.size(0), 1, device=self.device)
                    interpolates = alpha * expert_data + (1 - alpha) * policy_data
                    interpolates.requires_grad = True
                    interpolates_d = self.discriminator(interpolates)
                    grad = torch.autograd.grad(
                        outputs=interpolates_d,
                        inputs=interpolates,
                        grad_outputs=torch.ones(interpolates_d.size(), device=self.device),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    grad_pen_loss = 10 * ((torch.sqrt(torch.sum(grad.view(grad.size(0), -1) ** 2, dim=1) + 1e-12) - 1) ** 2).mean()

                    bound_loss = (
                        torch.clip(dist.loc - self.cfg["normalization"]["clip_actions"], min=0.0).square().mean()
                        + torch.clip(dist.loc + self.cfg["normalization"]["clip_actions"], max=0.0).square().mean()
                    )

                    entropy = dist.entropy().sum(dim=-1).mean()

                    reconstruction_loss = F.mse_loss(privileged_obs_est, self.buffer["privileged_obses"].flatten(0, 1)[batch_idx])

                    mirrored_dist, _ = self.model.act(
                        self.buffer["mirrored_obses"].flatten(0, 1)[batch_idx], self.buffer["mirrored_stacked_obses"].flatten(0, 1)[batch_idx]
                    )
                    symmetric_loss = F.mse_loss(dist.loc, self.env.mirror_act(mirrored_dist.loc))

                    loss = (
                        value_loss
                        + actor_loss
                        + self.cfg["algorithm"]["bound_coef"] * bound_loss
                        + self.cfg["algorithm"]["entropy_coef"] * entropy
                        + self.cfg["algorithm"]["reconstruction_coef"] * reconstruction_loss
                        + self.cfg["algorithm"]["amp_coef"] * amp_loss
                        + self.cfg["algorithm"]["grad_pen_coef"] * grad_pen_loss
                        + self.cfg["algorithm"]["symmetric_coef"] * symmetric_loss
                    )
                    self.optimizer.zero_grad()
                    self.discriminator_optimizer.zero_grad()
                    (loss / self.world_size).backward()
                    if self.world_size > 1:
                        for p in self.model.parameters():
                            if p.grad is not None:
                                torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM)
                        for p in self.discriminator.parameters():
                            if p.grad is not None:
                                torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.discriminator_optimizer.step()
                    for key in [
                        "value_loss",
                        "actor_loss",
                        "bound_loss",
                        "entropy",
                        "reconstruction_loss",
                        "amp_loss",
                        "grad_pen_loss",
                        "symmetric_loss",
                    ]:
                        statistics[key] = statistics.get(key, 0) + eval(key).item()

                    self.amp_normalizer.update(policy_state)
                    self.amp_normalizer.update(expert_state)

                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(dist.scale / old_dist.scale.flatten(0, 1)[batch_idx])
                            + 0.5
                            * (torch.square(old_dist.scale.flatten(0, 1)[batch_idx]) + torch.square(dist.loc - old_dist.loc.flatten(0, 1)[batch_idx]))
                            / torch.square(dist.scale)
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if self.world_size > 1:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.world_size
                        if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate
                        statistics["policy_pred"] = statistics.get("policy_pred", 0) + policy_d.mean()
                        statistics["expert_pred"] = statistics.get("expert_pred", 0) + expert_d.mean()
                        statistics["kl_mean"] = statistics.get("kl_mean", 0) + kl_mean
                        statistics["lr"] = statistics.get("lr", 0) + self.learning_rate
            for key in statistics.keys():
                statistics[key] /= self.cfg["algorithm"]["num_learning_epochs"] * self.cfg["algorithm"]["num_mini_batches"]

            if self.rank == 0:
                self.recorder.record_statistics(statistics, it)
                if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                    self.recorder.save({"model": self.model.state_dict(), "discriminator": self.discriminator.state_dict()}, it + 1)
                print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))

    def play(self):
        obs, infos = self.env.reset()
        privileged_obs = infos["privileged_obs"]
        stacked_obs = infos["stacked_obs"]
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        while True:
            with torch.no_grad():
                act = self.model(obs, stacked_obs)
                obs, rew, done, infos = self.env.step(act)
                privileged_obs = infos["privileged_obs"]
                stacked_obs = infos["stacked_obs"]
            if self.cfg["viewer"]["record_video"]:
                record_time -= self.env.dt
                if record_time < 0:
                    record_time += self.cfg["viewer"]["record_interval"]
                    self.interrupt = False
                    signal.signal(signal.SIGINT, self.interrupt_handler)
                    with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as writer:
                        for frame in self.env.camera_frames:
                            writer.append_data(frame)
                    if self.interrupt:
                        raise KeyboardInterrupt
                    signal.signal(signal.SIGINT, signal.default_int_handler)

    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True
