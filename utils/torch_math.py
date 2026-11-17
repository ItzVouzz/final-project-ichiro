import torch
import numpy as np


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def get_axis_params(value, axis_idx, x_value=0.0, dtype=np.float32, n_dims=3):
    zs = np.zeros((n_dims,), dtype=dtype)
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


def wxyz_to_xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.cat([q_wxyz[..., 1:4], q_wxyz[..., 0:1]], dim=-1)


def xyzw_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    return torch.cat([q_xyzw[..., 3:4], q_xyzw[..., 0:3]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., 0:3]
    q_w = q[..., 3:4]
    a = v * (2.0 * q_w.square() - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a + b + c


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., 0:3]
    q_w = q[..., 3:4]
    a = v * (2.0 * q_w.square() - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qx, qy, qz, qw], dim=-1)


def get_euler_xyz(q: torch.Tensor):
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = qw * qw - qx * qx - qy * qy + qz * qz
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = torch.where(torch.abs(sinp) >= 1.0, torch.sign(sinp) * (np.pi / 2.0), torch.asin(sinp))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


def torch_rand_float(lower: float, upper: float, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower