import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjwarp

from .base_task import BaseTask
from utils.torch_math import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
    quat_mul,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from utils.utils import apply_randomization
from utils.motion import Motion


class K1(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_curriculum()
        self._create_envs()
        self._init_buffers()
        self._prepare_reward_function()
        self.motion_sampler = Motion(self.cfg, self.device)

    def _init_curriculum(self):
        curr_cfg = self.cfg.get("curriculum", {})
        self.curriculum_enabled = curr_cfg.get("enabled", False)
        self.curriculum_progress = curr_cfg.get("init_progress", 0.15) if self.curriculum_enabled else 1.0
        self.curriculum_threshold = curr_cfg.get("advance_threshold", 0.35)
        self.curriculum_step = curr_cfg.get("step_size", 0.05)
        self.ema_success_rate = 0.0

    def update_curriculum(self, batch_success_rate: float):
        if not self.curriculum_enabled:
            return
        self.ema_success_rate = 0.9 * self.ema_success_rate + 0.1 * batch_success_rate
        if self.ema_success_rate > self.curriculum_threshold and self.curriculum_progress < 1.0:
            self.curriculum_progress = min(1.0, self.curriculum_progress + self.curriculum_step)
            self.ema_success_rate = 0.0

    def _create_envs(self):
        asset_cfg = self.cfg["asset"]
        self.num_dofs = self.mj_model.nu  # 14 DOFs untuk K1
        self.num_bodies = self.mj_model.nbody

        self.dof_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.num_dofs)
        ]
        # Indeks joint di mj_model (melewati world_joint di index 0)
        act_jnt_ids = self.mj_model.actuator_trnid[:, 0]
        self.dof_pos_limits = torch.tensor(
            self.mj_model.jnt_range[act_jnt_ids], dtype=torch.float, device=self.device
        )
        self.torque_limits = torch.tensor(
            self.mj_model.actuator_ctrlrange[:, 1], dtype=torch.float, device=self.device
        )

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

        body_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(self.num_bodies)
        ]
        self.base_indice = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, asset_cfg["base_name"])
        self.head_indice = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, asset_cfg["head_name"])
        self.ball_indice = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball")

        self.feet_indices = torch.tensor(
            [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in asset_cfg["foot_names"]],
            dtype=torch.long,
            device=self.device,
        )

        penalized_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_names.extend([s for s in body_names if name in s])
        self.penalized_contact_indices = torch.tensor(
            [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in set(penalized_names)],
            dtype=torch.long,
            device=self.device,
        )

        base_init_state_list = (
            self.cfg["init_state"]["pos"]
            + self.cfg["init_state"]["rot"]
            + self.cfg["init_state"]["lin_vel"]
            + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        self.env_origins = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # Simulasi domain randomization massa & COM base
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        base_mass_nom = float(self.mj_model.body_mass[self.base_indice])
        for k in range(3):
            _, self.base_mass_scaled[:, k] = apply_randomization(
                torch.zeros(self.num_envs, device=self.device),
                self.cfg["randomization"].get("base_com"),
                return_noise=True,
            )
        self.rand_base_mass, self.base_mass_scaled[:, 3] = apply_randomization(
            torch.full((self.num_envs,), base_mass_nom, device=self.device),
            self.cfg["randomization"].get("base_mass"),
            return_noise=True,
        )
        self.delta_base_weight = (self.rand_base_mass - base_mass_nom) * (-9.81)

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_critic_obs = self.cfg["env"]["num_critic_obs"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_amp_obs = self.cfg["env"]["num_amp_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.sim_dt = self.cfg["sim"]["dt"]
        self.dt = self.cfg["control"]["decimation"] * self.sim_dt

        # Zero-copy GPU views ke buffer mujoco_warp
        self.mj_qpos = wp.to_torch(self.mjw_data.qpos)
        self.mj_qvel = wp.to_torch(self.mjw_data.qvel)
        self.mj_ctrl = wp.to_torch(self.mjw_data.ctrl)
        self.mj_xpos = wp.to_torch(self.mjw_data.xpos)
        self.mj_xquat = wp.to_torch(self.mjw_data.xquat)
        self.mj_xfrc = wp.to_torch(self.mjw_data.xfrc_applied)
        self.has_rne_contact = hasattr(mjwarp, "rne_postconstraint") and hasattr(self.mjw_data, "cfrc_ext")
        if self.has_rne_contact:
            self.mj_cfrc_ext = wp.to_torch(self.mjw_data.cfrc_ext)

        # Slice qpos & qvel untuk robot (14 DOF) dan bola
        self.dof_pos = self.mj_qpos[:, 7 : 7 + self.num_dofs]
        self.dof_vel = self.mj_qvel[:, 6 : 6 + self.num_dofs]
        self.ball_qpos_idx = 7 + self.num_dofs  # 21
        self.ball_qvel_idx = 6 + self.num_dofs  # 20

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
        self.extras = {"rew_terms": {}}

        # State buffers kompatibel dengan t1.py (quaternion xyzw)
        self.root_states = torch.zeros(self.num_envs, 2, 13, dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.base_quat[:, 3] = 1.0
        self.base_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.feet_pos = torch.zeros(self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device)
        self.prev_substep_feet_pos = torch.zeros(self.num_envs, len(self.feet_indices), 2, dtype=torch.float, device=self.device)
        self.feet_quat = torch.zeros(self.num_envs, len(self.feet_indices), 4, dtype=torch.float, device=self.device)
        self.feet_vel = torch.zeros(self.num_envs, len(self.feet_indices), 2, dtype=torch.float, device=self.device)
        self.head_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_vel = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)

        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device)
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.relative_ball_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.relative_goal_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.last_ball_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.last_base_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.goal_cnt = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
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
            (self.num_envs, self.num_dofs),
            self.cfg["init_state"]["default_joint_angles"]["default"],
            dtype=torch.float,
            device=self.device,
        )
        for i, name in enumerate(self.dof_names):
            for key, value in self.cfg["init_state"]["default_joint_angles"].items():
                if key in name:
                    self.default_dof_pos[:, i] = value
                    break
        self.default_dof_pos[:] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("dof_offset"))
        self._sync_states_from_mj()

    def _sync_states_from_mj(self):
        # Robot root
        self.base_pos[:] = self.mj_qpos[:, 0:3]
        self.base_quat[:] = wxyz_to_xyzw(self.mj_qpos[:, 3:7])
        world_lin_vel = self.mj_qvel[:, 0:3]
        local_ang_vel = self.mj_qvel[:, 3:6]
        world_ang_vel = quat_rotate(self.base_quat, local_ang_vel)

        self.root_states[:, 0, 0:3] = self.base_pos
        self.root_states[:, 0, 3:7] = self.base_quat
        self.root_states[:, 0, 7:10] = world_lin_vel
        self.root_states[:, 0, 10:13] = world_ang_vel

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, world_lin_vel)
        self.base_ang_vel[:] = local_ang_vel
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_yaw[:] = get_euler_xyz(self.base_quat)[2]

        # Ball root
        b_idx = self.ball_qpos_idx
        bv_idx = self.ball_qvel_idx
        self.ball_pos[:] = self.mj_qpos[:, b_idx : b_idx + 3]
        self.ball_vel[:] = self.mj_qvel[:, bv_idx : bv_idx + 2]
        self.root_states[:, 1, 0:3] = self.ball_pos
        self.root_states[:, 1, 3:7] = wxyz_to_xyzw(self.mj_qpos[:, b_idx + 3 : b_idx + 7])
        self.root_states[:, 1, 7:10] = self.mj_qvel[:, bv_idx : bv_idx + 3]

    def _write_states_to_mj(self, env_ids):
        if len(env_ids) == 0:
            return
        self.mj_qpos[env_ids, 0:3] = self.root_states[env_ids, 0, 0:3]
        self.mj_qpos[env_ids, 3:7] = xyzw_to_wxyz(self.root_states[env_ids, 0, 3:7])
        self.mj_qvel[env_ids, 0:3] = self.root_states[env_ids, 0, 7:10]
        self.mj_qvel[env_ids, 3:6] = quat_rotate_inverse(
            self.root_states[env_ids, 0, 3:7], self.root_states[env_ids, 0, 10:13]
        )

        b_idx = self.ball_qpos_idx
        bv_idx = self.ball_qvel_idx
        self.mj_qpos[env_ids, b_idx : b_idx + 3] = self.root_states[env_ids, 1, 0:3]
        self.mj_qpos[env_ids, b_idx + 3 : b_idx + 7] = xyzw_to_wxyz(self.root_states[env_ids, 1, 3:7])
        self.mj_qvel[env_ids, bv_idx : bv_idx + 3] = self.root_states[env_ids, 1, 7:10]
        self.mj_qvel[env_ids, bv_idx + 3 : bv_idx + 6] = 0.0

    def _prepare_reward_function(self):
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name in self.reward_scales.keys():
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._sim_forward()
        self._sync_states_from_mj()
        self._refresh_feet_state(reset_vel=True)
        self._update_camera_and_rel_pos()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _sample_spawn_positions(self, n):
        p = self.curriculum_progress
        # Saat p kecil (awal kurikulum): bola dekat gawang (x=4.5..6.0), meluas ke [-7, 7] saat p -> 1.0
        ball_x_min = 5.0 * (1.0 - p) + (-7.0) * p
        ball_x_max = 6.2 * (1.0 - p) + 7.0 * p
        ball_y_lim = 1.0 * (1.0 - p) + 4.5 * p
        ball_x = torch_rand_float(ball_x_min, ball_x_max, (n, 1), self.device).squeeze(-1)
        ball_y = torch_rand_float(-ball_y_lim, ball_y_lim, (n, 1), self.device).squeeze(-1)

        if p < 0.99:
            # Robot spawn 0.8..2.5m di belakang bola saat kurikulum awal, meluas bertahap
            dist_back = torch_rand_float(0.8, 1.2 + 4.0 * p, (n, 1), self.device).squeeze(-1)
            lat_offset = torch_rand_float(-0.5 - 3.5 * p, 0.5 + 3.5 * p, (n, 1), self.device).squeeze(-1)
            robot_x = (ball_x - dist_back).clamp(-7.0, 6.5)
            robot_y = (ball_y + lat_offset).clamp(-4.5, 4.5)
        else:
            robot_x = torch_rand_float(-7.0, 7.0, (n, 1), self.device).squeeze(-1)
            robot_y = torch_rand_float(-4.5, 4.5, (n, 1), self.device).squeeze(-1)

        return robot_x, robot_y, ball_x, ball_y

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
        self.goal_cnt[env_ids] = 0.0

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.ball_delay_steps[env_ids] = (
            apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("ball_delay_step"),
            )
            .clamp(0, 19)
            .long()
        )
        self.ball_obs_buffer[env_ids] = 0.0
        self.extras["time_outs"] = self.time_out_buf | self.ball_reset_buf

    def _reset_dofs(self, env_ids, init_state):
        self.dof_pos[env_ids, :] = apply_randomization(
            self.default_dof_pos[env_ids, :], self.cfg["randomization"].get("init_dof_pos")
        )
        self.dof_vel[env_ids, :] = 0.0
        if self.cfg["motion"]["reference_state_init"]:
            self.dof_pos[env_ids, 2:] = init_state["dof_pos"]
            self.dof_vel[env_ids, 2:] = init_state["dof_vel"]

    def _reset_root_states(self, env_ids, robot_reset_env_ids, init_state):
        if len(robot_reset_env_ids) > 0:
            self.root_states[robot_reset_env_ids, 0, :] = self.base_init_state
        self.root_states[env_ids, 1, :] = self.base_init_state

        robot_x, robot_y, ball_x, ball_y = self._sample_spawn_positions(len(env_ids))
        self.root_states[env_ids, 0, 0] = robot_x
        self.root_states[env_ids, 0, 1] = robot_y
        self.root_states[env_ids, 1, 0] = ball_x
        self.root_states[env_ids, 1, 1] = ball_y

        # Hindari spawn bertumpuk dengan tiang gawang [7, +/-1.3] dan bola
        for post_y in (1.3, -1.3):
            post = torch.tensor([7.0, post_y], dtype=torch.float, device=self.device)
            self.root_states[env_ids, :2, 0] -= 0.5 * (
                torch.norm(self.root_states[env_ids, :2, 0:2] - post, dim=-1) < 0.5
            ).float()
        self.root_states[env_ids, 0, 0] -= (self.root_states[env_ids, 0, 0] > 0).float() * (
            torch.norm(self.root_states[env_ids, 0, 0:2] - self.root_states[env_ids, 1, 0:2], dim=-1) < 0.5
        ).float()
        self.root_states[env_ids, 1, 2] = 0.11
        self.root_states[env_ids, 1, 7:13] = 0.0

        if len(robot_reset_env_ids) > 0:
            if self.cfg["motion"]["reference_state_init"]:
                self.root_states[robot_reset_env_ids, 0, 2] = init_state["base_height"] + 0.01
                if self.curriculum_progress < 0.99:
                    # Arahkan yaw robot mendekati arah bola pada level awal kurikulum
                    dx = self.root_states[robot_reset_env_ids, 1, 0] - self.root_states[robot_reset_env_ids, 0, 0]
                    dy = self.root_states[robot_reset_env_ids, 1, 1] - self.root_states[robot_reset_env_ids, 0, 1]
                    target_yaw = torch.atan2(dy, dx) + (
                        (torch.rand(len(robot_reset_env_ids), device=self.device) - 0.5)
                        * 2.0
                        * np.pi
                        * self.curriculum_progress
                    )
                    self.root_states[robot_reset_env_ids, 0, 3:7] = quat_from_euler_xyz(
                        torch.zeros_like(target_yaw), torch.zeros_like(target_yaw), target_yaw
                    )
                else:
                    self.root_states[robot_reset_env_ids, 0, 3:7] = init_state["base_quat"]
                self.root_states[robot_reset_env_ids, 0, 7:10] = quat_rotate(
                    self.root_states[robot_reset_env_ids, 0, 3:7], init_state["base_lin_vel"]
                )
                self.root_states[robot_reset_env_ids, 0, 10:13] = quat_rotate(
                    self.root_states[robot_reset_env_ids, 0, 3:7], init_state["base_ang_vel"]
                )
            else:
                yaw = torch.rand(len(robot_reset_env_ids), dtype=torch.float, device=self.device) * (2 * np.pi)
                self.root_states[robot_reset_env_ids, 0, 3:7] = quat_from_euler_xyz(
                    torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
                )

        self._write_states_to_mj(env_ids)
        self.base_pos[env_ids, :] = self.root_states[env_ids, 0, 0:3]
        self.last_base_pos[env_ids, :] = self.root_states[env_ids, 0, 0:2]
        self.ball_pos[env_ids, :] = self.root_states[env_ids, 1, 0:3]
        self.last_ball_pos[env_ids, :] = self.root_states[env_ids, 1, 0:2]

    def _update_camera_and_rel_pos(self):
        self.relative_ball_pos[:, 0] = torch.cos(-self.base_yaw) * (
            self.ball_pos[:, 0] - self.base_pos[:, 0]
        ) - torch.sin(-self.base_yaw) * (self.ball_pos[:, 1] - self.base_pos[:, 1])
        self.relative_ball_pos[:, 1] = torch.sin(-self.base_yaw) * (
            self.ball_pos[:, 0] - self.base_pos[:, 0]
        ) + torch.cos(-self.base_yaw) * (self.ball_pos[:, 1] - self.base_pos[:, 1])

        self.relative_goal_pos[:, 0] = torch.cos(-self.base_yaw) * (7.0 - self.base_pos[:, 0]) - torch.sin(
            -self.base_yaw
        ) * (-self.base_pos[:, 1])
        self.relative_goal_pos[:, 1] = torch.sin(-self.base_yaw) * (7.0 - self.base_pos[:, 0]) + torch.cos(
            -self.base_yaw
        ) * (-self.base_pos[:, 1])

        self.head_quat[:] = wxyz_to_xyzw(self.mj_xquat[:, self.head_indice, :])
        head_pos = self.mj_xpos[:, self.head_indice, 0:3]
        cam_offset = to_torch(self.cfg["asset"]["camera_pos"], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        self.camera_pos[:] = head_pos + quat_rotate(self.head_quat, cam_offset)

    def step(self, actions):
        self.actions[:] = torch.clip(
            actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"]
        )
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions

        decimation = self.cfg["control"]["decimation"]
        for i in range(decimation):
            if i == decimation - 1:
                self.prev_substep_feet_pos[:] = self.mj_xpos[:, self.feet_indices, 0:2]
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = torch.clip(
                self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel,
                min=-self.torque_limits,
                max=self.torque_limits,
            )
            self.mj_ctrl[:] = dof_torques
            self._sim_step()

        if self.has_rne_contact:
            with wp.ScopedDevice(self.wp_device):
                mjwarp.rne_postconstraint(self.mjw_model, self.mjw_data)

        self.render()
        self._sync_states_from_mj()
        self._refresh_feet_state(reset_vel=False)
        self._update_camera_and_rel_pos()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._kick_robots()
        self._push_robots()
        self._kick_balls()
        self._reset_balls()
        self._check_termination()
        self._compute_reward()
        self.extras["last_amp_obs"] = self._compute_amp_obs()
        self.extras["ball_to_target_error"] = torch.norm(
            self.ball_pos[:, 0:2] - torch.tensor([7.0, 0.0], dtype=torch.float, device=self.device), dim=-1
        )

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_idx(env_ids)
            self._sim_forward()
            self._sync_states_from_mj()
            self._refresh_feet_state(reset_vel=True)
            self._update_camera_and_rel_pos()

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
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 0, 7:10] = apply_randomization(
                self.root_states[:, 0, 7:10], self.cfg["randomization"].get("kick_lin_vel")
            )
            self.root_states[:, 0, 10:13] = apply_randomization(
                self.root_states[:, 0, 10:13], self.cfg["randomization"].get("kick_ang_vel")
            )
            self.mj_qvel[:, 0:3] = self.root_states[:, 0, 7:10]
            self.mj_qvel[:, 3:6] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])

    def _push_robots(self):
        push_steps = int(np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt))
        dur_steps = int(np.ceil(self.cfg["randomization"]["push_duration_s"] / self.dt))
        if self.common_step_counter % push_steps == 0:
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
        elif self.common_step_counter % push_steps == dur_steps:
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()

        ball_speed = torch.norm(self.ball_vel[:, :2], dim=-1).clip(min=0.1)
        self.pushing_forces[:, self.ball_indice, 0] = -self.ball_friction_forces * self.ball_vel[:, 0] / ball_speed
        self.pushing_forces[:, self.ball_indice, 1] = -self.ball_friction_forces * self.ball_vel[:, 1] / ball_speed

        # Terapkan gaya eksternal & variasi massa base ke mjw_data.xfrc_applied
        self.mj_xfrc.zero_()
        world_push_f = quat_rotate(self.base_quat, self.pushing_forces[:, self.base_indice, :])
        world_push_f[:, 2] += self.delta_base_weight
        world_push_t = quat_rotate(self.base_quat, self.pushing_torques[:, self.base_indice, :])
        self.mj_xfrc[:, self.base_indice, 0:3] = world_push_f
        self.mj_xfrc[:, self.base_indice, 3:6] = world_push_t
        self.mj_xfrc[:, self.ball_indice, 0:3] = self.pushing_forces[:, self.ball_indice, :]

    def _kick_balls(self):
        env_ids = (
            torch.bernoulli(torch.full((self.num_envs,), self.cfg["randomization"]["kick_ball_prob"], device=self.device))
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) == 0:
            return
        bv_idx = self.ball_qvel_idx
        cur_vel = self.mj_qvel[env_ids, bv_idx : bv_idx + 2]
        delta_vel = (
            (torch.norm(cur_vel, dim=-1) < 0.1).float().unsqueeze(-1)
            * self.cfg["randomization"]["kick_ball_vel"]
            * torch.randn_like(cur_vel)
        )
        self.mj_qvel[env_ids, bv_idx : bv_idx + 2] += delta_vel
        self.root_states[env_ids, 1, 7:9] = self.mj_qvel[env_ids, bv_idx : bv_idx + 2]

    def _reset_balls(self):
        self.ball_reset_buf[:] = torch.bernoulli(
            torch.full((self.num_envs,), self.cfg["randomization"]["reset_ball_prob"], device=self.device)
        ).bool()
        env_ids = self.ball_reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        _, _, ball_x, ball_y = self._sample_spawn_positions(len(env_ids))
        self.root_states[env_ids, 1, 0] = ball_x
        self.root_states[env_ids, 1, 1] = ball_y
        self.root_states[env_ids, 1, 2] = 0.11
        self.root_states[env_ids, 1, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.root_states[env_ids, 1, 7:13] = 0.0

        b_idx = self.ball_qpos_idx
        bv_idx = self.ball_qvel_idx
        self.mj_qpos[env_ids, b_idx : b_idx + 3] = self.root_states[env_ids, 1, 0:3]
        self.mj_qpos[env_ids, b_idx + 3 : b_idx + 7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.mj_qvel[env_ids, bv_idx : bv_idx + 6] = 0.0
        self.ball_pos[env_ids, :] = self.root_states[env_ids, 1, 0:3]
        self.last_ball_pos[env_ids, :] = self.root_states[env_ids, 1, 0:2]

    def _refresh_feet_state(self, reset_vel=False):
        self.feet_pos[:] = self.mj_xpos[:, self.feet_indices, 0:3]
        self.feet_quat[:] = wxyz_to_xyzw(self.mj_xquat[:, self.feet_indices, :])
        if reset_vel:
            self.feet_vel.zero_()
            self.prev_substep_feet_pos[:] = self.feet_pos[:, :, 0:2]
        else:
            self.feet_vel[:] = (self.feet_pos[:, :, 0:2] - self.prev_substep_feet_pos) / self.sim_dt

        roll, pitch, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_pitch[:] = (pitch.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi

        feet_edge_rel = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        exp_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_rel.shape[2], -1).reshape(-1, 3)
        exp_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_rel.shape[2], -1).reshape(-1, 4)
        edge_pos = (exp_pos + quat_rotate(exp_quat, feet_edge_rel.reshape(-1, 3))).reshape(
            self.num_envs, len(self.feet_indices), feet_edge_rel.shape[2], 3
        )
        self.feet_edge_pos[:] = edge_pos
        self.feet_contact[:] = torch.any(edge_pos[..., 2] < 0.01, dim=2)
        self.feet_solid_contact[:] = torch.sum(edge_pos[..., 2] < 0.01, dim=2) > (feet_edge_rel.shape[2] - 1)

    def _check_termination(self):
        self.reset_buf[:] = self.root_states[:, 0, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        border = 0.9 * self.terrain.border_size
        self.reset_buf[:] |= (torch.abs(self.base_pos[:, 0]) > border) | (torch.abs(self.base_pos[:, 1]) > border)
        self.reset_buf[:] |= self.base_pos[:, 2] < self.cfg["rewards"]["terminate_height"]
        self.extras["fall"] = self.reset_buf.float()

        self.ball_out_buf[:] = (
            (self.ball_pos[:, 0] < -7.0)
            | ((self.ball_pos[:, 0] > 7.0) & (torch.abs(self.ball_pos[:, 1]) > 1.3))
            | (torch.abs(self.ball_pos[:, 1]) > 4.5)
        )
        goal = ((self.ball_pos[:, 0] > 7.0) & (torch.abs(self.ball_pos[:, 1]) < 1.3)).float()
        self.goal_cnt[:] = (self.goal_cnt + goal) * goal
        self.extras["success"] = (self.goal_cnt >= 50).float()
        self.time_out_buf[:] = (
            self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        ) | (self.goal_cnt >= 50)
        self.reset_buf[:] |= self.ball_out_buf | self.time_out_buf

    def _compute_reward(self):
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
        base_yaw_with_noise = apply_randomization(self.base_yaw, self.cfg["noise"].get("base_yaw"))
        cam_rot = quat_mul(
            self.head_quat,
            torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, -1),
        )
        ball_to_camera = quat_rotate_inverse(cam_rot, self.ball_pos - self.camera_pos)
        condition = (
            (ball_to_camera[:, 2] > 0)
            & (torch.abs(torch.atan2(ball_to_camera[:, 0], ball_to_camera[:, 2])) < 0.5 * self.cfg["asset"]["camera_fov"][0] / 180 * torch.pi)
            & (torch.abs(torch.atan2(ball_to_camera[:, 1], ball_to_camera[:, 2])) < 0.5 * self.cfg["asset"]["camera_fov"][1] / 180 * torch.pi)
        ).float()
        ball_in_view = (
            condition
            * (
                torch.rand(self.num_envs, dtype=torch.float, device=self.device)
                < 2.475 - 0.225 * torch.norm(self.relative_ball_pos, dim=-1).clip(min=7.0)
            ).float()
            + (1.0 - condition)
            * (self.relative_ball_pos[:, 0] > -2.0).float()
            * (torch.rand(self.num_envs, dtype=torch.float, device=self.device) < 0.01).float()
        )

        self.ball_obs_buffer[:] = torch.roll(self.ball_obs_buffer, 1, dims=1)
        mask = self.episode_length_buf % 2 == 0
        self.ball_obs_buffer[mask, 0, 0:2] = ball_in_view[mask].unsqueeze(-1) * (
            self.relative_ball_pos[mask, :]
            + (0.149 + 0.124 * torch.norm(self.relative_ball_pos[mask, :], dim=-1)).unsqueeze(-1)
            * torch.randn_like(self.relative_ball_pos[mask, :])
        )
        self.ball_obs_buffer[mask, 0, 2] = ball_in_view[mask]
        self.ball_obs_buffer[~mask, 0, :] = self.ball_obs_buffer[~mask, 1, :]

        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                self.ball_obs_buffer[torch.arange(self.num_envs, device=self.device), self.ball_delay_steps, :],
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
                self.base_pos[:, 2:3],
                self.pushing_forces[:, self.ball_indice, 0:2],
                self.ball_vel,
            ),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                self.base_lin_vel * self.cfg["normalization"]["lin_vel"],
                self.base_pos[:, 2:3],
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

    # ------------ Reward Functions ----------------
    def _reward_survival(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_termination(self):
        return (self.reset_buf & ~self.ball_out_buf & ~self.time_out_buf).float()

    def _reward_pos_still(self):
        env_arange = torch.arange(self.num_envs, device=self.device)
        self.base_pos_buffer[env_arange, self.buffer_idx, 0:2] = self.base_pos[:, 0:2]
        self.base_pos_buffer[env_arange, self.buffer_idx, 2] = self.base_yaw
        self.buffer_count[:] = torch.clip(self.buffer_count + 1, max=self.base_pos_buffer.shape[1])
        self.buffer_idx[:] = (self.buffer_idx + 1) % self.base_pos_buffer.shape[1]
        pos_dist = torch.norm(
            self.base_pos_buffer[:, :, 0:2] - self.base_pos_buffer[env_arange, self.buffer_idx, 0:2].unsqueeze(1), dim=-1
        )
        yaw_diff = self.base_pos_buffer[:, :, 2] - self.base_pos_buffer[env_arange, self.buffer_idx, 2].unsqueeze(1)
        yaw_dist = torch.abs(torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)))
        return (
            (self.buffer_count == self.base_pos_buffer.shape[1])
            & torch.all(pos_dist < self.cfg["rewards"]["penalize_pos"], dim=-1)
            & (torch.all(yaw_dist < self.cfg["rewards"]["penalize_yaw"], dim=-1) | torch.all(pos_dist < 0.3, dim=-1))
            & (torch.norm(self.relative_ball_pos, dim=-1) > self.cfg["rewards"]["penalize_pos_distance"])
        ).float()

    def _reward_goal(self):
        return ((self.ball_pos[:, 0] > 7.0) & (torch.abs(self.ball_pos[:, 1]) < 1.3)).float()

    def _reward_ball_distance(self):
        return torch.norm(self.ball_pos[:, 0:2] - self.last_base_pos, dim=-1) - torch.norm(
            self.ball_pos[:, 0:2] - self.base_pos[:, 0:2], dim=-1
        )

    def _reward_goal_distance(self):
        goal_pt = torch.tensor([7.0, 0.0], dtype=torch.float, device=self.device)
        last_val = (1.0 - ((self.last_ball_pos[:, 0] > 7.0) & (torch.abs(self.last_ball_pos[:, 1]) < 1.3)).float()) * torch.norm(
            self.last_ball_pos - goal_pt, dim=-1
        )
        cur_val = (1.0 - ((self.ball_pos[:, 0] > 7.0) & (torch.abs(self.ball_pos[:, 1]) < 1.3)).float()) * torch.norm(
            self.ball_pos[:, 0:2] - goal_pt, dim=-1
        )
        return last_val - cur_val

    def _reward_kick_ball(self):
        edge_to_ball = torch.norm(
            self.feet_edge_pos[:, :, 0:2, 0:2]
            - self.ball_pos[:, 0:2].unsqueeze(1).unsqueeze(1).expand(-1, self.feet_edge_pos.shape[1], 2, -1),
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
        cam_rot = quat_mul(
            self.head_quat,
            torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, -1),
        )
        ball_to_camera = quat_rotate_inverse(cam_rot, self.ball_pos - self.camera_pos)
        z = torch.where(torch.abs(ball_to_camera[:, 2]) < 1e-8, torch.ones_like(ball_to_camera[:, 2]) * 1e-8, ball_to_camera[:, 2])
        return torch.square(torch.atan2(ball_to_camera[:, 1], z))

    def _reward_face_ball_yaw(self):
        cam_rot = quat_mul(
            self.head_quat,
            torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, -1),
        )
        ball_to_camera = quat_rotate_inverse(cam_rot, self.ball_pos - self.camera_pos)
        z = torch.where(torch.abs(ball_to_camera[:, 2]) < 1e-8, torch.ones_like(ball_to_camera[:, 2]) * 1e-8, ball_to_camera[:, 2])
        return torch.square(torch.atan2(ball_to_camera[:, 0], z))

    def _reward_collision(self):
        if len(self.penalized_contact_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.has_rne_contact:
            # cfrc_ext[..., 3:6] adalah gaya kontak linear pada body di MuJoCo
            forces = self.mj_cfrc_ext[:, self.penalized_contact_indices, 3:6]
            return torch.sum(torch.norm(forces, dim=-1) > 1.0, dim=-1).float()
        return torch.sum(self.mj_xpos[:, self.penalized_contact_indices, 2] < 0.08, dim=-1).float()

    def _reward_root_acc(self):
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 0, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions)[:, 2:], dim=-1)

    def _reward_head_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions)[:, 0:2], dim=-1)

    def _reward_dof_pos_limits(self):
        span = self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1.0 - self.cfg["rewards"]["soft_dof_pos_limit"]) * span
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1.0 - self.cfg["rewards"]["soft_dof_pos_limit"]) * span
        return torch.sum((lower - self.dof_pos).clip(min=0.0) + (self.dof_pos - upper).clip(min=0.0), dim=-1)

    def _reward_feet_min_distance(self):
        feet_min_dist = (
            torch.cdist(self.feet_edge_pos[:, 0, :, :2], self.feet_edge_pos[:, 1, :, :2]).reshape(self.num_envs, -1).min(dim=-1).values
        )
        return (feet_min_dist < self.cfg["rewards"]["feet_min_distance"]).float()

    @staticmethod
    def mirror_obs(obs):
        mat = torch.zeros(55, 55, dtype=torch.float, device=obs.device)
        mat[0:15, 0:15] = torch.eye(15)
        mat[15:21, 21:27] = torch.eye(6)
        mat[21:27, 15:21] = torch.eye(6)
        mat[27:29, 27:29] = torch.eye(2)
        mat[29:35, 35:41] = torch.eye(6)
        mat[35:41, 29:35] = torch.eye(6)
        mat[41:43, 41:43] = torch.eye(2)
        mat[43:49, 49:55] = torch.eye(6)
        mat[49:55, 43:49] = torch.eye(6)

        flip_val = torch.ones(55, dtype=torch.float, device=obs.device)
        inverse_ids = [
            1, 3, 5, 7, 10, 12,
            13, 16, 17, 20, 22, 23, 26,
            27, 30, 31, 34, 36, 37, 40,
            41, 44, 45, 48, 50, 51, 54,
        ]
        flip_val[inverse_ids] = -1.0
        mirror_transform_mat = torch.matmul(mat, torch.diag(flip_val))
        return torch.matmul(mirror_transform_mat, obs.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def mirror_act(act):
        mat = torch.zeros(14, 14, dtype=torch.float, device=act.device)
        mat[0:2, 0:2] = torch.eye(2)
        mat[2:8, 8:14] = torch.eye(6)
        mat[8:14, 2:8] = torch.eye(6)

        flip_val = torch.ones(14, dtype=torch.float, device=act.device)
        inverse_ids = [0, 3, 4, 7, 9, 10, 13]
        flip_val[inverse_ids] = -1.0
        mirror_transform_mat = torch.matmul(mat, torch.diag(flip_val))
        return torch.matmul(mirror_transform_mat, act.unsqueeze(-1)).squeeze(-1)