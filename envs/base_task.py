import sys
import inspect
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco.viewer
import mujoco_warp as mjwarp
import atexit


class FlatTerrain:
    def __init__(self, cfg):
        self.border_size = cfg.get("border_size", 20.0)
        self.env_width = cfg.get("env_width", 0.0)
        self.env_length = cfg.get("env_length", 0.0)

    def terrain_heights(self, points):
        return torch.zeros(points.shape[0], dtype=torch.float, device=points.device)


class BaseTask:

    def __init__(self, cfg):
        self.cfg = cfg
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.create_sim()
        self.terrain = FlatTerrain(self.cfg["terrain"])
        self.set_viewer()

    def create_sim(self):
        sim_cfg = self.cfg["sim"]
        self.device = self.cfg["basic"]["device"]
        self.headless = self.cfg["basic"]["headless"]
        self.up_axis_idx = 2 if sim_cfg.get("up_axis", "z") == "z" else 1
        self.num_envs = self.cfg["env"]["num_envs"]

        wp.init()
        self.wp_device = wp.get_device(self.device if "cuda" in self.device else "cpu")

        # Load MJCF XML scene
        xml_path = self.cfg["asset"]["mujoco_file"]
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.timestep = sim_cfg["dt"]
        self.mj_model.opt.gravity[:] = sim_cfg["gravity"]
        self.mj_data_vis = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data_vis)

        # Allocate parallel worlds on GPU via mujoco_warp
        with wp.ScopedDevice(self.wp_device):
            self.mjw_model = mjwarp.put_model(self.mj_model)
            sig = inspect.signature(mjwarp.make_data)
            make_kwargs = {"nworld": self.num_envs}
            if "naconmax" in sig.parameters:
                make_kwargs["naconmax"] = self.num_envs * 64
            elif "nconmax" in sig.parameters:
                make_kwargs["nconmax"] = 64
            if "njmax" in sig.parameters:
                make_kwargs["njmax"] = 128
            self.mjw_data = mjwarp.make_data(self.mj_model, **make_kwargs)
            mjwarp.forward(self.mjw_model, self.mjw_data)

            self.use_cuda_graph = sim_cfg.get("use_cuda_graph", True) and ("cuda" in self.device)
            self.step_graph = None
            if self.use_cuda_graph:
                with wp.ScopedCapture() as capture:
                    mjwarp.step(self.mjw_model, self.mjw_data)
                self.step_graph = capture.graph

    def _sim_step(self):
        if self.step_graph is not None:
            wp.capture_launch(self.step_graph)
        else:
            with wp.ScopedDevice(self.wp_device):
                mjwarp.step(self.mjw_model, self.mjw_data)

    def _sim_forward(self):
        with wp.ScopedDevice(self.wp_device):
            mjwarp.forward(self.mjw_model, self.mjw_data)

    def set_viewer(self):
        self.viewer = None
        self.renderer = None
        self.camera_frames = []
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data_vis)
            self.viewer.cam.distance = 4.5
            self.viewer.cam.elevation = -20.0
            atexit.register(self.viewer.close)
        if self.cfg["viewer"]["record_video"]:
            self.renderer = mujoco.Renderer(self.mj_model, height=720, width=1280)

    def render(self):
        if self.viewer is not None:
            if not self.viewer.is_running():
                sys.exit()
            env_idx = self.cfg["viewer"].get("record_env_idx", 0)
            self.mj_data_vis.qpos[:] = wp.to_torch(self.mjw_data.qpos)[env_idx].detach().cpu().numpy()
            self.mj_data_vis.qvel[:] = wp.to_torch(self.mjw_data.qvel)[env_idx].detach().cpu().numpy()
            mujoco.mj_forward(self.mj_model, self.mj_data_vis)
            self.viewer.cam.lookat[:] = self.mj_data_vis.qpos[0:3]
            self.viewer.sync()

        if self.cfg["viewer"]["record_video"] and self.renderer is not None:
            env_idx = self.cfg["viewer"].get("record_env_idx", 0)
            self.mj_data_vis.qpos[:] = wp.to_torch(self.mjw_data.qpos)[env_idx].detach().cpu().numpy()
            self.mj_data_vis.qvel[:] = wp.to_torch(self.mjw_data.qvel)[env_idx].detach().cpu().numpy()
            mujoco.mj_forward(self.mj_model, self.mj_data_vis)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = self.mj_data_vis.qpos[0:3]
            cam.distance = 3.5
            cam.elevation = -20.0
            cam.azimuth = 135.0
            self.renderer.update_scene(self.mj_data_vis, camera=cam)
            img = self.renderer.render()
            self.camera_frames.append(img)