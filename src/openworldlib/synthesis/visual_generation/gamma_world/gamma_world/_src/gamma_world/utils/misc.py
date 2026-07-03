# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from functools import wraps

import torch

from gamma_world._src.imaginaire.utils import log


class sync_timer:


    def __init__(self, name=None, flag_env="SYNC_TIMER"):
        self.name = name
        self.flag_env = flag_env

    def __enter__(self):
        if os.environ.get(self.flag_env, "0") == "1":
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        if os.environ.get(self.flag_env, "0") == "1":
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                log.info(f"{self.name} takes {self.time / 1000:.4f}s", rank0_only=False)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper
