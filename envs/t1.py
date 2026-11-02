# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
    quat_mul,
)
import os
import torch
import numpy as np
from .base_task import BaseTask
from utils.utils import apply_randomization
from utils.motion import Motion


class T1(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()
        self.motion_sampler = Motion(self.cfg, self.device)

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])
        self.head_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["head_name"])

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)

        # create ball asset
        ball_options = gymapi.AssetOptions()
        ball_options.density = 100
        ball_asset = self.gym.create_sphere(self.sim, 0.1, ball_options)
        self.ball_indice = self.num_bodies

        # create goal asset
        goal_options = gymapi.AssetOptions()
        goal_options.fix_base_link = 1
        goal_asset = self.gym.load_asset(self.sim, asset_root, "goal.urdf", goal_options)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.ball_handles = []
        self.goal_handles = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            start_pos = gymapi.Transform()
            start_pos.p = gymapi.Vec3(*(self.env_origins[i]))

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pos, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.actor_handles.append(actor_handle)

            ball_handle = self.gym.create_actor(env_handle, ball_asset, start_pos, "ball", i, 0, 0)
            ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            ball_body_props[0].mass = apply_randomization(0.0, self.cfg["randomization"].get("ball_mass"))
            self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_body_props, recomputeInertia=True)
            ball_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, ball_handle)
            ball_shape_props[0].restitution = apply_randomization(0.0, self.cfg["randomization"].get("ball_restitution"))
            self.gym.set_actor_rigid_shape_properties(env_handle, ball_handle, ball_shape_props)
            self.ball_handles.append(ball_handle)

            goal_handle = self.gym.create_actor(env_handle, goal_asset, start_pos, "goal", i, 0, 0)
            self.goal_handles.append(goal_handle)

            self.envs.append(env_handle)

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True
                )
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = max(1.0, np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.terrain_heights(self.env_origins)

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_critic_obs = self.cfg["env"]["num_critic_obs"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_amp_obs = self.cfg["env"]["num_amp_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.critic_obs_buf = torch.zeros(self.num_envs, self.num_critic_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.amp_obs_buf = torch.zeros(self.num_envs, self.num_amp_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.extras = {}
        self.extras["rew_terms"] = {}

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, -1, 13)
        self.base_pos = self.root_states[:, 0, 0:3].clone()
        self.base_pos[:, 0:2] -= self.env_origins[:, 0:2]
        self.base_quat = self.root_states[:, 0, 3:7]
        self.base_yaw = get_euler_xyz(self.base_quat)[2]
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3].clone()
        self.feet_pos[:, :, 0:2] -= self.env_origins[:, 0:2].unsqueeze(1)
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.body_states[:, self.feet_indices, 7:9]
        self.head_quat = self.body_states[:, self.head_indice, 3:7]
        self.ball_pos = self.root_states[:, 1, 0:3].clone()
        self.ball_pos[:, 0:2] -= self.env_origins[:, 0:2]
        self.ball_vel = self.root_states[:, 1, 7:9]

        # initialize some data used later on
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 0, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.relative_ball_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.relative_goal_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.last_ball_pos = torch.zeros_like(self.ball_pos[:, 0:2])
        self.last_base_pos = torch.zeros_like(self.base_pos[:, 0:2])
        self.goal_cnt = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.pushing_forces = torch.zeros(self.num_envs, self.gym.get_env_rigid_body_count(self.envs[0]), 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.gym.get_env_rigid_body_count(self.envs[0]), 3, dtype=torch.float, device=self.device)
        self.ball_friction_forces = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_pitch = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.last_feet_yaw = torch.zeros_like(self.feet_yaw)
        self.feet_edge_pos = torch.zeros(
            self.num_envs, len(self.feet_indices), len(self.cfg["asset"]["feet_edge_pos"]), 3, dtype=torch.float, device=self.device
        )
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.feet_solid_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.camera_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.base_pos_buffer = torch.zeros(self.num_envs, 50, 3, dtype=torch.float, device=self.device)
        self.buffer_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.buffer_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ball_obs_buffer = torch.zeros(self.num_envs, 20, 3, device=self.device)
        self.ball_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.default_dof_pos = torch.full(
            (self.num_envs, self.num_dofs), self.cfg["init_state"]["default_joint_angles"]["default"], dtype=torch.float, device=self.device
        )
        for i, name in enumerate(self.dof_names):
            for key, value in self.cfg["init_state"]["default_joint_angles"].items():
                if key in name:
                    self.default_dof_pos[:, i] = value
                    break
        self.default_dof_pos[:] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("dof_offset"))

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def reset(self):
        """Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        robot_reset_env_ids = (self.reset_buf & ~self.ball_out_buf & ~self.time_out_buf).nonzero(as_tuple=False).flatten()
        init_state = None
        if len(robot_reset_env_ids) > 0:
            self.buffer_count[robot_reset_env_ids] = 0
            self.buffer_idx[robot_reset_env_ids] = 0
            init_state = self.motion_sampler.sample_batch(len(robot_reset_env_ids))
            self._reset_dofs(robot_reset_env_ids, init_state)
        self._reset_root_states(env_ids, robot_reset_env_ids, init_state)

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 0, 7:13]
        self.episode_length_buf[env_ids] = 0

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.ball_delay_steps[env_ids] = apply_randomization(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device), self.cfg["randomization"].get("ball_delay_step")
        ).long()
        self.ball_obs_buffer[env_ids] = 0.0
        self.extras["time_outs"] = self.time_out_buf | self.ball_reset_buf

    def _reset_dofs(self, env_ids, init_state):
        self.dof_pos[env_ids, :] = apply_randomization(self.default_dof_pos[env_ids, :], self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids, :] = 0.0
        if self.cfg["motion"]["reference_state_init"]:
            self.dof_pos[env_ids, 2:] = init_state["dof_pos"]
            self.dof_vel[env_ids, 2:] = init_state["dof_vel"]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.gym.get_actor_count(self.envs[0]) * env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids, robot_reset_env_ids, init_state):
        if len(robot_reset_env_ids) > 0:
            self.root_states[robot_reset_env_ids, 0, :] = self.base_init_state
        self.root_states[env_ids, 1, :] = self.base_init_state
        self.root_states[env_ids, :2, 0] = torch_rand_float(-7, 7, (len(env_ids), 2), device=self.device)
        self.root_states[env_ids, :2, 1] = torch_rand_float(-4.5, 4.5, (len(env_ids), 2), device=self.device)
        self.root_states[env_ids, :2, 0] -= (
            0.5
            * (torch.norm(self.root_states[env_ids, :2, 0:2] - torch.tensor([7, 1.3], dtype=torch.float, device=self.device), dim=-1) < 0.5).float()
        )
        self.root_states[env_ids, :2, 0] -= (
            0.5
            * (torch.norm(self.root_states[env_ids, :2, 0:2] - torch.tensor([7, -1.3], dtype=torch.float, device=self.device), dim=-1) < 0.5).float()
        )
        self.root_states[env_ids, 0, 0] -= (self.root_states[env_ids, 0, 0] > 0).float() * (
            torch.norm(self.root_states[env_ids, 0, 0:2] - self.root_states[env_ids, 1, 0:2], dim=-1) < 0.5
        ).float()
        self.root_states[env_ids, :2, :2] += self.env_origins[env_ids, :2].unsqueeze(1)
        self.root_states[env_ids, 1, 2] = 0.11 + self.terrain.terrain_heights(self.root_states[env_ids, 1, :2])
        if len(robot_reset_env_ids) > 0:
            if self.cfg["motion"]["reference_state_init"]:
                self.root_states[robot_reset_env_ids, 0, 2] = (
                    init_state["base_height"] + self.terrain.terrain_heights(self.root_states[robot_reset_env_ids, 0, :2]) + 0.01
                )
                self.root_states[robot_reset_env_ids, 0, 3:7] = init_state["base_quat"]
                self.root_states[robot_reset_env_ids, 0, 7:10] = quat_rotate(init_state["base_quat"], init_state["base_lin_vel"])
                self.root_states[robot_reset_env_ids, 0, 10:13] = quat_rotate(init_state["base_quat"], init_state["base_ang_vel"])
            else:
                self.root_states[robot_reset_env_ids, 0, 2] += self.terrain.terrain_heights(self.root_states[robot_reset_env_ids, 0, :2])
                self.root_states[robot_reset_env_ids, 0, 3:7] = quat_from_euler_xyz(
                    torch.zeros(len(robot_reset_env_ids), dtype=torch.float, device=self.device),
                    torch.zeros(len(robot_reset_env_ids), dtype=torch.float, device=self.device),
                    torch.rand(len(robot_reset_env_ids), dtype=torch.float, device=self.device) * (2 * torch.pi),
                )
                self.root_states[robot_reset_env_ids, 0, 7:9] = apply_randomization(
                    torch.zeros(len(robot_reset_env_ids), 2, dtype=torch.float, device=self.device),
                    self.cfg["randomization"].get("init_base_lin_vel_xy"),
                )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.base_pos[env_ids, :] = self.root_states[env_ids, 0, 0:3]
        self.base_pos[env_ids, 0:2] -= self.env_origins[env_ids, 0:2]
        self.last_base_pos[env_ids, :] = self.root_states[env_ids, 0, 0:2] - self.env_origins[env_ids, 0:2]
        self.ball_pos[env_ids, :] = self.root_states[env_ids, 1, 0:3]
        self.ball_pos[env_ids, 0:2] -= self.env_origins[env_ids, 0:2]
        self.last_ball_pos[env_ids, :] = self.root_states[env_ids, 1, 0:2] - self.env_origins[env_ids, 0:2]

    def step(self, actions):
        # pre physics step
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions

        # perform physics step
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = torch.clip(
                self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel,
                min=-self.torque_limits,
                max=self.torque_limits,
            )
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.render()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_pos[:] = self.root_states[:, 0, 0:3]
        self.base_pos[:, 0:2] -= self.env_origins[:, 0:2]
        self.base_quat[:] = self.root_states[:, 0, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.ball_pos[:] = self.root_states[:, 1, 0:3]
        self.ball_pos[:, 0:2] -= self.env_origins[:, 0:2]
        self.ball_vel[:] = self.root_states[:, 1, 7:9]
        self.base_yaw[:] = get_euler_xyz(self.base_quat)[2]
        self.relative_ball_pos[:, 0] = torch.cos(-self.base_yaw) * (self.ball_pos[:, 0] - self.base_pos[:, 0]) - torch.sin(-self.base_yaw) * (
            self.ball_pos[:, 1] - self.base_pos[:, 1]
        )
        self.relative_ball_pos[:, 1] = torch.sin(-self.base_yaw) * (self.ball_pos[:, 0] - self.base_pos[:, 0]) + torch.cos(-self.base_yaw) * (
            self.ball_pos[:, 1] - self.base_pos[:, 1]
        )
        self.relative_goal_pos[:, 0] = torch.cos(-self.base_yaw) * (7 - self.base_pos[:, 0]) - torch.sin(-self.base_yaw) * (-self.base_pos[:, 1])
        self.relative_goal_pos[:, 1] = torch.sin(-self.base_yaw) * (7 - self.base_pos[:, 0]) + torch.cos(-self.base_yaw) * (-self.base_pos[:, 1])

        self._refresh_feet_state()
        self.head_quat[:] = self.body_states[:, self.head_indice, 3:7]
        self.camera_pos[:] = self.body_states[:, self.head_indice, 0:3] + quat_rotate(
            self.head_quat, to_torch(self.cfg["asset"]["camera_pos"], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        )
        self.camera_pos[:, 0:2] -= self.env_origins[:, 0:2]

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._kick_robots()
        self._push_robots()
        self._kick_balls()
        self._reset_balls()
        self._check_termination()
        self._compute_reward()
        self.extras["last_amp_obs"] = self._compute_amp_obs()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 0, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        self.last_feet_yaw[:] = self.feet_yaw
        self.last_base_pos[:] = self.base_pos[:, 0:2]
        self.last_ball_pos[:] = self.ball_pos[:, 0:2]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 0, 7:10] = apply_randomization(self.root_states[:, 0, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 0, 10:13] = apply_randomization(self.root_states[:, 0, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, self.base_indice, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, self.base_indice, :]),
                self.cfg["randomization"].get("push_torque"),
            )
            self.ball_friction_forces[:] = apply_randomization(
                torch.zeros_like(self.ball_friction_forces), self.cfg["randomization"].get("ball_friction")
            )
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
        self.pushing_forces[:, self.ball_indice, 0] = (
            -self.ball_friction_forces * self.ball_vel[:, 0] / torch.norm(self.ball_vel[:, :2], dim=-1).clip(min=0.1)
        )
        self.pushing_forces[:, self.ball_indice, 1] = (
            -self.ball_friction_forces * self.ball_vel[:, 1] / torch.norm(self.ball_vel[:, :2], dim=-1).clip(min=0.1)
        )
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def _kick_balls(self):
        env_ids = (
            torch.bernoulli(torch.full((self.num_envs,), self.cfg["randomization"]["kick_ball_prob"], device=self.device))
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) == 0:
            return
        self.root_states[env_ids, 1, 7:9] += (
            (torch.norm(self.root_states[env_ids, 1, 7:9], dim=-1) < 0.1).float().unsqueeze(-1)
            * self.cfg["randomization"]["kick_ball_vel"]
            * torch.randn_like(self.root_states[env_ids, 1, 7:9])
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _reset_balls(self):
        self.ball_reset_buf[:] = torch.bernoulli(torch.full((self.num_envs,), self.cfg["randomization"]["reset_ball_prob"], device=self.device))
        env_ids = self.ball_reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        self.root_states[env_ids, 1, 0] = torch_rand_float(-7, 7, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 1, 1] = torch_rand_float(-4.5, 4.5, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 1, 0] -= (
            0.5
            * (torch.norm(self.root_states[env_ids, 1, 0:2] - torch.tensor([7, 1.3], dtype=torch.float, device=self.device), dim=-1) < 0.5).float()
        )
        self.root_states[env_ids, 1, 0] -= (
            0.5
            * (torch.norm(self.root_states[env_ids, 1, 0:2] - torch.tensor([7, -1.3], dtype=torch.float, device=self.device), dim=-1) < 0.5).float()
        )
        self.root_states[env_ids, 1, 0] -= (self.root_states[env_ids, 1, 0] > 0).float() * (
            torch.norm(self.base_pos[env_ids, 0:2] - self.root_states[env_ids, 1, 0:2], dim=-1) < 0.5
        ).float()
        self.root_states[env_ids, 1, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, 1, 2] = 0.11 + self.terrain.terrain_heights(self.root_states[env_ids, 1, :2])
        self.root_states[env_ids, 1, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.ball_pos[env_ids, :] = self.root_states[env_ids, 1, 0:3]
        self.ball_pos[env_ids, 0:2] -= self.env_origins[env_ids, 0:2]
        self.last_ball_pos[env_ids, :] = self.root_states[env_ids, 1, 0:2] - self.env_origins[env_ids, 0:2]

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_pos[:, :, 0:2] -= self.env_origins[:, 0:2].unsqueeze(1)
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.body_states[:, self.feet_indices, 7:9]
        roll, pitch, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_pitch[:] = (pitch.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        self.feet_contact[:] = torch.any(
            (
                feet_edge_pos[:, 2]
                - self.terrain.terrain_heights(
                    feet_edge_pos[:, 0:2] + self.env_origins[:, 0:2].repeat_interleave(feet_edge_pos.shape[0] // self.env_origins.shape[0], dim=0)
                )
                < 0.01
            ).reshape(self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]),
            dim=2,
        )
        self.feet_solid_contact[:] = (
            torch.sum(
                (
                    feet_edge_pos[:, 2]
                    - self.terrain.terrain_heights(
                        feet_edge_pos[:, 0:2] + self.env_origins[:, 0:2].repeat_interleave(feet_edge_pos.shape[0] // self.env_origins.shape[0], dim=0)
                    )
                    < 0.01
                ).reshape(self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]),
                dim=2,
            )
            > feet_edge_relative_pos.shape[2] - 1
        )
        self.feet_edge_pos[:] = feet_edge_pos.reshape(self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2], 3)

    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf[:] = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf[:] |= self.root_states[:, 0, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.reset_buf[:] |= (
            (self.root_states[:, 0, 0] < -0.9 * self.terrain.border_size)
            | (self.root_states[:, 0, 0] > self.terrain.env_width + 0.9 * self.terrain.border_size)
            | (self.root_states[:, 0, 1] < -0.9 * self.terrain.border_size)
            | (self.root_states[:, 0, 1] > self.terrain.env_length + 0.9 * self.terrain.border_size)
        )
        self.reset_buf[:] |= (
            self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos[:, 0:2] + self.env_origins[:, 0:2])
            < self.cfg["rewards"]["terminate_height"]
        )
        self.extras["fall"] = self.reset_buf.float()
        self.ball_out_buf[:] = (
            (self.ball_pos[:, 0] < -7) | ((self.ball_pos[:, 0] > 7) & (torch.abs(self.ball_pos[:, 1]) > 1.3)) | (torch.abs(self.ball_pos[:, 1]) > 4.5)
        )
        goal = ((self.ball_pos[:, 0] > 7) & (torch.abs(self.ball_pos[:, 1]) < 1.3)).float()
        self.goal_cnt[:] = (self.goal_cnt + goal) * goal
        self.extras["success"] = (self.goal_cnt >= 50).float()
        self.time_out_buf[:] = (self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)) | (self.goal_cnt >= 50)
        self.reset_buf[:] |= self.ball_out_buf | self.time_out_buf

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if name in self.cfg["rewards"]["rew_group_item"]:
                self.rew_buf[:, 0] += rew
            else:
                self.rew_buf[:, 1] += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _compute_observations(self):
        """Computes observations"""
        base_yaw_with_noise = apply_randomization(self.base_yaw, self.cfg["noise"].get("base_yaw"))
        ball_to_camera = quat_rotate_inverse(
            quat_mul(
                self.head_quat, torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            ),
            self.ball_pos - self.camera_pos,
        )
        condition = (
            (ball_to_camera[:, 2] > 0)
            & (torch.abs(torch.atan2(ball_to_camera[:, 0], ball_to_camera[:, 2])) < 0.5 * self.cfg["asset"]["camera_fov"][0] / 180 * torch.pi)
            & (torch.abs(torch.atan2(ball_to_camera[:, 1], ball_to_camera[:, 2])) < 0.5 * self.cfg["asset"]["camera_fov"][1] / 180 * torch.pi)
        ).float()
        ball_in_view = (
            condition
            * (
                torch.rand(self.num_envs, dtype=torch.float, device=self.device)
                < 2.475 - 0.225 * torch.norm(self.relative_ball_pos, dim=-1).clip(min=7)
            ).float()
            + (1 - condition)
            * (self.relative_ball_pos[:, 0] > -2).float()
            * (torch.rand(self.num_envs, dtype=torch.float, device=self.device) < 0.01).float()
        )
        self.ball_obs_buffer[:] = torch.roll(self.ball_obs_buffer, 1, dims=1)
        mask = self.episode_length_buf % 2 == 0
        self.ball_obs_buffer[mask, 0, 0:2] = ball_in_view[mask].unsqueeze(-1) * (
            self.relative_ball_pos[mask, :]
            + (0.149 + 0.124 * torch.norm(self.relative_ball_pos[mask, :], dim=-1)).unsqueeze(-1) * torch.randn_like(self.relative_ball_pos[mask, :])
        )
        self.ball_obs_buffer[mask, 0, 2] = ball_in_view[mask]
        self.ball_obs_buffer[~mask, 0, :] = self.ball_obs_buffer[~mask, 1, :]
        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                self.ball_obs_buffer[torch.arange(self.num_envs), self.ball_delay_steps, :],
                apply_randomization(self.relative_goal_pos, self.cfg["noise"].get("goal_pos")),
                torch.cos(base_yaw_with_noise).unsqueeze(-1),
                torch.sin(base_yaw_with_noise).unsqueeze(-1),
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )
        self.critic_obs_buf = torch.cat(
            (
                self.projected_gravity * self.cfg["normalization"]["gravity"],
                self.base_ang_vel * self.cfg["normalization"]["ang_vel"],
                self.relative_ball_pos,
                ball_in_view.unsqueeze(-1),
                self.relative_goal_pos,
                torch.cos(self.base_yaw).unsqueeze(-1),
                torch.sin(self.base_yaw).unsqueeze(-1),
                (self.dof_pos - self.default_dof_pos) * self.cfg["normalization"]["dof_pos"],
                self.dof_vel * self.cfg["normalization"]["dof_vel"],
                self.actions,
                self.base_mass_scaled,
                self.base_lin_vel * self.cfg["normalization"]["lin_vel"],
                (self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos[:, 0:2] + self.env_origins[:, 0:2])).unsqueeze(-1),
                self.pushing_forces[:, self.ball_indice, 0:2],
                self.ball_vel,
            ),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                self.base_lin_vel * self.cfg["normalization"]["lin_vel"],
                (self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos[:, 0:2] + self.env_origins[:, 0:2])).unsqueeze(-1),
                self.pushing_forces[:, self.ball_indice, 0:2],
                self.relative_ball_pos,
                self.ball_vel,
            ),
            dim=-1,
        )
        self.amp_obs_buf = self._compute_amp_obs()
        self.extras["critic_obs"] = self.critic_obs_buf
        self.extras["privileged_obs"] = self.privileged_obs_buf
        self.extras["amp_obs"] = self.amp_obs_buf

    def _compute_amp_obs(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_pos[:, :, 0:2] -= self.env_origins[:, 0:2].unsqueeze(1)
        return torch.cat(
            (
                self.projected_gravity,
                self.base_lin_vel,
                self.base_ang_vel,
                self.dof_pos[:, 2:],
                self.dof_vel[:, 2:] * self.cfg["normalization"]["dof_vel"],
                quat_rotate_inverse(self.base_quat, self.feet_pos[:, 0] - self.base_pos),
                quat_rotate_inverse(self.base_quat, self.feet_pos[:, 1] - self.base_pos),
            ),
            dim=-1,
        )

    # ------------ reward functions----------------
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_termination(self):
        return (self.reset_buf & ~self.ball_out_buf & ~self.time_out_buf).float()

    def _reward_pos_still(self):
        self.base_pos_buffer[torch.arange(self.num_envs), self.buffer_idx, 0:2] = self.base_pos[:, 0:2]
        self.base_pos_buffer[torch.arange(self.num_envs), self.buffer_idx, 2] = self.base_yaw
        self.buffer_count[:] = torch.clip(self.buffer_count + 1, max=self.base_pos_buffer.shape[1])
        self.buffer_idx[:] = (self.buffer_idx + 1) % self.base_pos_buffer.shape[1]
        pos_dist = torch.norm(
            self.base_pos_buffer[:, :, 0:2] - self.base_pos_buffer[torch.arange(self.num_envs), self.buffer_idx, 0:2].unsqueeze(1), dim=-1
        )
        yaw_diff = self.base_pos_buffer[:, :, 2] - self.base_pos_buffer[torch.arange(self.num_envs), self.buffer_idx, 2].unsqueeze(1)
        yaw_dist = torch.abs(torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)))
        return (
            (self.buffer_count == self.base_pos_buffer.shape[1])
            & torch.all(pos_dist < self.cfg["rewards"]["penalize_pos"], dim=-1)
            & (torch.all(yaw_dist < self.cfg["rewards"]["penalize_yaw"], dim=-1) | torch.all(pos_dist < 0.3, dim=-1))
            & (torch.norm(self.relative_ball_pos, dim=-1) > self.cfg["rewards"]["penalize_pos_distance"])
        ).float()

    def _reward_goal(self):
        return ((self.ball_pos[:, 0] > 7) & (torch.abs(self.ball_pos[:, 1]) < 1.3)).float()

    def _reward_ball_distance(self):
        return torch.norm(self.ball_pos[:, 0:2] - self.last_base_pos, dim=-1) - torch.norm(self.ball_pos[:, 0:2] - self.base_pos[:, 0:2], dim=-1)

    def _reward_goal_distance(self):
        last_value = (1.0 - ((self.last_ball_pos[:, 0] > 7) & (torch.abs(self.last_ball_pos[:, 1]) < 1.3)).float()) * torch.norm(
            self.last_ball_pos - torch.tensor([7, 0], dtype=torch.float, device=self.device), dim=-1
        )
        value = (1.0 - ((self.ball_pos[:, 0] > 7) & (torch.abs(self.ball_pos[:, 1]) < 1.3)).float()) * torch.norm(
            self.ball_pos[:, 0:2] - torch.tensor([7, 0], dtype=torch.float, device=self.device), dim=-1
        )
        return last_value - value

    def _reward_kick_ball(self):
        edge_to_ball = torch.norm(
            self.feet_edge_pos[:, :, 0:2, 0:2] - self.ball_pos[:, 0:2].unsqueeze(1).unsqueeze(1).expand(-1, self.feet_edge_pos.shape[1], 2, -1),
            dim=-1,
        )
        feet_vel_x = torch.cos(self.feet_yaw) * self.feet_vel[:, :, 0] + torch.sin(self.feet_yaw) * self.feet_vel[:, :, 1]
        return torch.sum(torch.all((edge_to_ball < 0.2), dim=-1).float() * feet_vel_x.clip(min=0.0), dim=-1)

    def _reward_side_kick_ball(self):
        edge_to_ball = torch.norm(
            self.feet_edge_pos[:, :, :, 0:2]
            - self.ball_pos[:, 0:2].unsqueeze(1).unsqueeze(1).expand(-1, self.feet_edge_pos.shape[1], self.feet_edge_pos.shape[2], -1),
            dim=-1,
        )
        feet_vel_y = -torch.sin(self.feet_yaw) * self.feet_vel[:, :, 0] + torch.cos(self.feet_yaw) * self.feet_vel[:, :, 1]
        return ((edge_to_ball[:, 0, 1] < 0.3) & (edge_to_ball[:, 0, 3] < 0.3)).float() * (-feet_vel_y[:, 0]).clip(
            min=0.0, max=self.cfg["rewards"]["max_kick_vel"]
        ) + ((edge_to_ball[:, 1, 0] < 0.3) & (edge_to_ball[:, 1, 2] < 0.3)).float() * feet_vel_y[:, 1].clip(
            min=0.0, max=self.cfg["rewards"]["max_kick_vel"]
        )

    def _reward_face_ball_pitch(self):
        ball_to_camera = quat_rotate_inverse(
            quat_mul(
                self.head_quat, torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            ),
            self.ball_pos - self.camera_pos,
        )
        ball_to_camera[:, 2] = torch.where(torch.abs(ball_to_camera[:, 2]) < 1e-8, torch.sign(ball_to_camera[:, 2]) * 1e-8, ball_to_camera[:, 2])
        return torch.square(torch.atan2(ball_to_camera[:, 1], ball_to_camera[:, 2]))

    def _reward_face_ball_yaw(self):
        ball_to_camera = quat_rotate_inverse(
            quat_mul(
                self.head_quat, torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            ),
            self.ball_pos - self.camera_pos,
        )
        ball_to_camera[:, 2] = torch.where(torch.abs(ball_to_camera[:, 2]) < 1e-8, torch.sign(ball_to_camera[:, 2]) * 1e-8, ball_to_camera[:, 2])
        return torch.square(torch.atan2(ball_to_camera[:, 0], ball_to_camera[:, 2]))

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 0, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions)[:, 2:], dim=-1)

    def _reward_head_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions)[:, 0:2], dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum((lower - self.dof_pos).clip(min=0.0) + (self.dof_pos - upper).clip(min=0.0), dim=-1)

    def _reward_feet_min_distance(self):
        feet_min_distance = (
            torch.cdist(self.feet_edge_pos[:, 0, :, :2], self.feet_edge_pos[:, 1, :, :2]).reshape(self.num_envs, -1).min(dim=-1).values
        )
        return (feet_min_distance < self.cfg["rewards"]["feet_min_distance"]).float()

    @staticmethod
    def mirror_obs(obs):
        mat = torch.zeros(58, 58, dtype=torch.float, device=obs.device)
        mat[0:16, 0:16] = torch.eye(16)
        mat[16:22, 22:28] = torch.eye(6)
        mat[22:28, 16:22] = torch.eye(6)
        mat[28:31, 28:31] = torch.eye(3)
        mat[31:37, 37:43] = torch.eye(6)
        mat[37:43, 31:37] = torch.eye(6)
        mat[43:46, 43:46] = torch.eye(3)
        mat[46:52, 52:58] = torch.eye(6)
        mat[52:58, 46:52] = torch.eye(6)
        flip_val = torch.ones(58, dtype=torch.float, device=obs.device)
        inverse_ids = [1, 3, 5, 7, 10, 12, 13, 15, 17, 18, 21, 23, 24, 27, 28, 30, 32, 33, 36, 38, 39, 42, 43, 45, 47, 48, 51, 53, 54, 57]
        flip_val[inverse_ids] = -1
        flip_mat = torch.diag(flip_val)
        mirror_transform_mat = torch.matmul(mat, flip_mat)
        mirrored_obs = torch.matmul(mirror_transform_mat, obs.unsqueeze(-1)).squeeze(-1)
        return mirrored_obs

    @staticmethod
    def mirror_act(act):
        mat = torch.zeros(15, 15, dtype=torch.float, device=act.device)
        mat[0:3, 0:3] = torch.eye(3)
        mat[3:9, 9:15] = torch.eye(6)
        mat[9:15, 3:9] = torch.eye(6)
        flip_val = torch.ones(15, dtype=torch.float, device=act.device)
        inverse_ids = [0, 2, 4, 5, 8, 10, 11, 14]
        flip_val[inverse_ids] = -1
        flip_mat = torch.diag(flip_val)
        mirror_transform_mat = torch.matmul(mat, flip_mat)
        mirrored_act = torch.matmul(mirror_transform_mat, act.unsqueeze(-1)).squeeze(-1)
        return mirrored_act
