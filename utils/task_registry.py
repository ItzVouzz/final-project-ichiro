# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


class TaskRegistry:

    def __init__(self):
        self.task_classes = {}
        self.task_cfgs = {}

    def register(self, name, task_class, task_cfg):
        self.task_classes[name] = task_class
        self.task_cfgs[name] = task_cfg

    def get_task_class(self, name):
        return self.task_classes[name]

    def get_task_cfg(self, name):
        return self.task_cfgs[name]


task_registry = TaskRegistry()
