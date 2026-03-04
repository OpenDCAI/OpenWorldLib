import torch

from ..commons.infer_state import get_infer_state


def torch_compile_wrapper():
    """返回一个装饰器，延迟决定是否使用torch.compile"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if get_infer_state() and get_infer_state().enable_torch_compile:
                compiled_func = torch.compile(func)
                return compiled_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
