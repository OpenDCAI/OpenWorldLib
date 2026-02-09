# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
try:
    from easydict import EasyDict  # type: ignore
except Exception:
    class EasyDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value
