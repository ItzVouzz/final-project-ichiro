import os
import time
import glob
import yaml
import argparse
import numpy as np
import torch
import mujoco, mujoco.viewer
from utils.model import *


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w])


def sample_curriculum_spawn(p=0.20):
    ball_x = np.random.uniform(5.0 * (1.0 - p) - 7.0 * p, 6.2 * (1.0 - p) + 7.0 * p)
    ball_y = np.random.uniform(-(1.0 * (1.0 - p) + 4.5 * p), 1.0 * (1.0 - p) + 4.5 * p)
    dist_back = np.random.uniform(0.8, 1.2 + 4.0 * p)
    lat_offset = np.random.uniform(-0.5 - 3.5 * p, 0.5 + 3.5 * p)
    robot_x = np.clip(ball_x - dist_back, -7.0, 6.5)
    robot_y = np.clip(ball_y + lat_offset, -4.5, 4.5)
    yaw_spread = max(0.42, p) * 2.0 * np.pi
    yaw = np.arctan2(ball_y - robot_y, ball_x - robot_x) + (np.random.rand() - 0.5) * yaw_spread
    return robot_x, robot_y, yaw, ball_x, ball_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    model = ActorCritic(cfg)
    model.eval()

    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(model_dict["model"], strict=False)

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)

    num_dofs = cfg["env"]["num_actions"]
    init_z = cfg["init_state"]["pos"][2]
    qpos_ball = 7 + num_dofs
    qvel_ball = 6 + num_dofs

    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    for i in range(mj_model.nu):
        act_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]
        for name, val in cfg["init_state"]["default_joint_angles"].items():
            if name in act_name:
                default_dof_pos[i] = val
                break
        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in act_name:
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            raise ValueError(f"PD gain of joint {act_name} were not defined")

    actions = np.zeros(num_dofs, dtype=np.float32)
    stacked_obs = np.zeros((cfg["runner"]["num_stack"], cfg["env"]["num_observations"]), dtype=np.float32)
    dof_targets = default_dof_pos.copy()
    ball_obs_buffer = np.zeros((cfg["randomization"]["ball_delay_step"]["range"][0], 3), dtype=np.float32)
    goal_cnt = 0
    out_cnt = 0
    fall_cnt = -1

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 4.5
        viewer.cam.elevation = -20
        init = True
        base_pos = np.zeros(3, dtype=np.float32)
        ball_pos = np.zeros(3, dtype=np.float32)
        while viewer.is_running():
            step_start = time.time()
            if init or base_pos[2] < cfg["rewards"]["terminate_height"]:
                init = False
                fall_cnt += 1
                ball_obs_buffer[:] = 0.0
                actions[:] = 0.0
                it = 0
                rx, ry, yaw, bx, by = sample_curriculum_spawn(p=0.20)
                mj_data.qpos[0:7] = np.array([rx, ry, init_z, np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)])
                mj_data.qpos[7:qpos_ball] = default_dof_pos
                mj_data.qpos[qpos_ball : qpos_ball + 7] = np.array([bx, by, 0.11, 1.0, 0.0, 0.0, 0.0])
                mj_data.qvel[:] = 0.0
                mujoco.mj_forward(mj_model, mj_data)
                print(f"Goal: {goal_cnt}, Out: {out_cnt}, Fall: {fall_cnt}")
            elif np.abs(ball_pos[0]) > 7.0 or np.abs(ball_pos[1]) > 4.5:
                if (ball_pos[0] > 7.0) and (np.abs(ball_pos[1]) < 1.3):
                    goal_cnt += 1
                else:
                    out_cnt += 1
                rx, ry, yaw, bx, by = sample_curriculum_spawn(p=0.20)
                mj_data.qpos[0:7] = np.array([rx, ry, init_z, np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)])
                mj_data.qpos[7:qpos_ball] = default_dof_pos
                mj_data.qpos[qpos_ball : qpos_ball + 7] = np.array([bx, by, 0.11, 1.0, 0.0, 0.0, 0.0])
                mj_data.qvel[:] = 0.0
                ball_obs_buffer[:] = 0.0
                actions[:] = 0.0
                it = 0
                mujoco.mj_forward(mj_model, mj_data)
                print(f"Goal: {goal_cnt}, Out: {out_cnt}, Fall: {fall_cnt}")

            dof_pos = mj_data.qpos.astype(np.float32)[7:qpos_ball]
            dof_vel = mj_data.qvel.astype(np.float32)[6:qvel_ball]
            base_pos = mj_data.sensor("position").data.astype(np.float32)
            base_quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            camera_pos = mj_data.sensor("camera_position").data.astype(np.float32)
            camera_quat = mj_data.sensor("camera_orientation").data[[1, 2, 3, 0]].astype(np.float32)
            projected_gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
            ball_pos = mj_data.qpos.astype(np.float32)[qpos_ball : qpos_ball + 3]

            sin_yaw = 2.0 * (base_quat[3] * base_quat[2] + base_quat[0] * base_quat[1])
            cos_yaw = base_quat[3] ** 2 + base_quat[0] ** 2 - base_quat[1] ** 2 - base_quat[2] ** 2
            base_yaw = np.arctan2(sin_yaw, cos_yaw)
            relative_ball_pos = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]]) @ (ball_pos - base_pos)[0:2]
            relative_goal_pos = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]]) @ (np.array([7.0, 0.0]) - base_pos[0:2])
            ball_to_camera = quat_rotate_inverse(quat_mul(camera_quat, [0.5, 0.5, 0.5, 0.5]), ball_pos - camera_pos)
            ball_in_view = float(
                (ball_to_camera[2] > 0)
                & (np.linalg.norm(relative_ball_pos, axis=-1) < 11.0)
                & (np.abs(np.arctan2(ball_to_camera[0], ball_to_camera[2])) < 0.5 * cfg["asset"]["camera_fov"][0] / 180.0 * np.pi)
                & (np.abs(np.arctan2(ball_to_camera[1], ball_to_camera[2])) < 0.5 * cfg["asset"]["camera_fov"][1] / 180.0 * np.pi)
            )

            if it % cfg["control"]["decimation"] == 0:
                # Update buffer kamera terlebih dahulu agar pada it==0 bola langsung terlihat
                ball_obs_buffer[1:, :] = ball_obs_buffer[:-1, :]
                if (it // cfg["control"]["decimation"]) % 2 == 0:
                    ball_obs_buffer[0, 0:2] = ball_in_view * relative_ball_pos
                    ball_obs_buffer[0, 2] = ball_in_view
                if it == 0:
                    ball_obs_buffer[:] = ball_obs_buffer[0:1, :]

                obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
                obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
                obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
                obs[6:9] = ball_obs_buffer[-1, :]
                obs[9:11] = relative_goal_pos
                obs[11] = np.cos(base_yaw)
                obs[12] = np.sin(base_yaw)
                obs[13 : 13 + num_dofs] = (dof_pos - default_dof_pos) * cfg["normalization"]["dof_pos"]
                obs[13 + num_dofs : 13 + 2 * num_dofs] = dof_vel * cfg["normalization"]["dof_vel"]
                obs[13 + 2 * num_dofs : 13 + 3 * num_dofs] = actions
                if it == 0:
                    stacked_obs[:] = obs[np.newaxis, :]
                stacked_obs[1:, :] = stacked_obs[:-1, :]
                stacked_obs[0, :] = obs

                with torch.no_grad():
                    actions[:] = model(torch.tensor(obs).unsqueeze(0), torch.tensor(stacked_obs).unsqueeze(0)).squeeze(0).detach().numpy()
                actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
                action_scale = np.full(num_dofs, cfg["control"]["action_scale"], dtype=np.float32)
                action_scale[0:2] = cfg["control"].get("head_action_scale", cfg["control"]["action_scale"])
                dof_targets[:] = default_dof_pos + action_scale * actions

            mj_data.ctrl[:] = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(mj_model, mj_data)
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
            it += 1
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)