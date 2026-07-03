# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .__about__ import __version__ as __version__


def _check_cuda_extra():
    """Check if CUDA extra is installed."""
    try:
        import cosmos_cuda
    except ImportError:
        raise RuntimeError("CUDA extra not installed. Please run 'uv sync --extra=<cuda_name>'") from None

    if __version__ != cosmos_cuda.__version__:
        raise RuntimeError(
            f"CUDA extra version mismatch: {cosmos_cuda.__version__} != {__version__}. Please run 'uv sync --extra=<cuda_name>'"
        )


_check_cuda_extra()
