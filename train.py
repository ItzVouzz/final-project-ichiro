# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaacgym
from utils.runner import Runner


if __name__ == "__main__":
    runner = Runner()
    runner.train()
