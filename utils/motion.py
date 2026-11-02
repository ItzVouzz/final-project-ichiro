# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaacgym.torch_utils import quat_rotate_inverse, quat_mul
import os
import csv
import torch
import numpy as np
import pinocchio as pin


class Motion:

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        file_list = []
        if os.path.isdir(self.cfg["motion"]["motion_path"]):
            for root, dirs, files in os.walk(self.cfg["motion"]["motion_path"]):
                file_list.extend([os.path.join(root, file) for file in files if file[0] != "."])
        else:
            file_list.append(self.cfg["motion"]["motion_path"])
        origin_data = []
        valid_idxs = []
        start_idx = 0
        for file in file_list:
            with open(file, "r") as f:
                motion_data = torch.from_numpy(np.array(list(csv.reader(f))[1:], dtype=np.float32)).to(self.device)
                origin_data.append(motion_data)
                end_idx = start_idx + motion_data.shape[0]
                valid_idxs.append(torch.arange(start_idx, end_idx - 1, device=self.device))
                start_idx = end_idx
        origin_data = torch.cat(origin_data, dim=0)
        self.valid_idxs = torch.cat(valid_idxs, dim=0)

        self.base_height = origin_data[:, 2].clone()
        self.base_quat = origin_data[:, 3:7].clone()
        self.data = torch.zeros(origin_data.shape[0], self.cfg["env"]["num_amp_obs"], dtype=torch.float32, device=self.device)
        self.data[:, 0:3] = quat_rotate_inverse(
            origin_data[:, 3:7], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).unsqueeze(0).expand(origin_data.shape[0], -1)
        )
        self.data[:, 3:9] = origin_data[:, 7:13]
        self.data[:, 9:35] = origin_data[:, 13:39]
        pin_model = pin.buildModelFromUrdf(self.cfg["asset"]["file"])
        pin_data = pin_model.createData()
        for idx in range(self.data.shape[0]):
            pin.forwardKinematics(pin_model, pin_data, np.concatenate(([0, 0], self.data[idx, 9:22].cpu().numpy())))
            pin.updateFramePlacements(pin_model, pin_data)
            self.data[idx, 35:38] = torch.tensor(
                pin_data.oMf[pin_model.getFrameId("left_foot_link")].translation, dtype=torch.float32, device=self.device
            )
            self.data[idx, 38:41] = torch.tensor(
                pin_data.oMf[pin_model.getFrameId("right_foot_link")].translation, dtype=torch.float32, device=self.device
            )

    def sample_batch(self, batch_size):
        sampled_idxs = self.valid_idxs[torch.randint(0, self.valid_idxs.shape[0], (batch_size,), device=self.device)]
        yaw_offset = torch.rand(batch_size, dtype=torch.float32, device=self.device) * (2 * np.pi)
        yaw_quat = torch.stack(
            [torch.zeros_like(yaw_offset), torch.zeros_like(yaw_offset), torch.sin(yaw_offset / 2), torch.cos(yaw_offset / 2)], dim=1
        )
        return {
            "base_height": self.base_height[sampled_idxs],
            "base_quat": quat_mul(yaw_quat, self.base_quat[sampled_idxs]),
            "base_lin_vel": self.data[sampled_idxs, 3:6],
            "base_ang_vel": self.data[sampled_idxs, 6:9],
            "dof_pos": self.data[sampled_idxs, 9:22],
            "dof_vel": self.data[sampled_idxs, 22:35],
        }

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sampled_idxs = self.valid_idxs[torch.randint(0, self.valid_idxs.shape[0], (mini_batch_size,), device=self.device)]
            yield torch.cat(
                (
                    self.data[sampled_idxs, 0:22],
                    self.data[sampled_idxs, 22:35] * self.cfg["normalization"]["dof_vel"],
                    self.data[sampled_idxs, 35:41],
                ),
                dim=-1,
            ), torch.cat(
                (
                    self.data[sampled_idxs + 1, 0:22],
                    self.data[sampled_idxs + 1, 22:35] * self.cfg["normalization"]["dof_vel"],
                    self.data[sampled_idxs + 1, 35:41],
                ),
                dim=-1,
            )
