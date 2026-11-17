# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from envs.k1 import K1
from utils.task_registry import task_registry

task_registry.register("K1", K1, "K1.yaml")

try:
    from envs.t1 import T1
    task_registry.register("T1", T1, "T1.yaml")
except ImportError:
    pass
