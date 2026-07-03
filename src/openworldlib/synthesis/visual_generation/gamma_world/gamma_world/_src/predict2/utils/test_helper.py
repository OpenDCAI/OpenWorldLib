# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for comparing PyTorch tensors with detailed difference reporting."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor


@dataclass
class TensorDifference:
    """Contains detailed information about differences between two tensors."""

    name: str
    max_absolute_diff: float
    max_relative_diff: float
    abs_diff_index: List[int]
    rel_diff_index: List[int]
    absolute_tolerance: float
    relative_tolerance: float
    tensor_shape: Tuple[int, ...]
    error_message: str

    def __str__(self) -> str:
        """Formats the difference information as a human-readable string."""
        return (
            f"{self.name}:\n"
            f"  Shape: {self.tensor_shape}\n"
            f"  Max absolute difference: {self.max_absolute_diff:.6f} at index {self.abs_diff_index}"
            f" (tolerance: {self.absolute_tolerance})\n"
            f"  Max relative difference: {self.max_relative_diff:.6f} at index {self.rel_diff_index}"
            f" (tolerance: {self.relative_tolerance})\n"
            f"  Details: {self.error_message}"
        )


def compute_tensor_differences(tensor_a: Tensor, tensor_b: Tensor, epsilon: float = 1e-12) -> Tuple[Tensor, Tensor]:
    """Computes absolute and relative differences between two tensors.

    Args:
        tensor_a: First tensor for comparison.
        tensor_b: Second tensor for comparison.
        epsilon: Small value to prevent division by zero in relative difference.

    Returns:
        Tuple of (absolute_differences, relative_differences) tensors.
    """
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / (torch.abs(tensor_b) + epsilon)
    return abs_diff, rel_diff


def get_max_difference_info(differences: Tensor, flatten: bool = True) -> Tuple[float, List[int]]:
    """Gets the maximum difference and its location in the tensor.

    Args:
        differences: Tensor containing differences.
        flatten: Whether to flatten the tensor before finding max.

    Returns:
        Tuple of (max_difference, index_of_max).
    """
    if flatten:
        differences = differences.flatten()
    max_diff = differences.max().item()
    max_idx = differences.argmax().tolist()
    return max_diff, [max_idx] if isinstance(max_idx, int) else max_idx


def compare_tensors(
    names: Sequence[str],
    tensors_a: Sequence[Tensor],
    tensors_b: Sequence[Tensor],
    atol: float = 0.5,
    rtol: float = 0.05,
    raise_on_mismatch: bool = True,
    verbose: bool = True,
) -> List[Optional[TensorDifference]]:
    """Compares two sets of tensors and provides detailed mismatch information.

    Args:
        names: Sequence of tensor names or layer identifiers.
        tensors_a: First sequence of tensors for comparison.
        tensors_b: Second sequence of tensors for comparison.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        raise_on_mismatch: Whether to raise ValueError on tolerance violations.
        verbose: Whether to print detailed comparison information.

    Returns:
        List of TensorDifference objects for mismatched tensors, None for matched ones.

    Raises:
        ValueError: If tensors don't match and raise_on_mismatch is True.
        RuntimeError: If input sequences have different lengths.
    """
    if len(names) != len(tensors_a) or len(tensors_a) != len(tensors_b):
        raise RuntimeError(
            f"Input sequence lengths must match: "
            f"names({len(names)}), tensors_a({len(tensors_a)}), "
            f"tensors_b({len(tensors_b)})"
        )

    differences: List[Optional[TensorDifference]] = []
    mismatched_names: List[str] = []

    for name, tensor_a, tensor_b in zip(names, tensors_a, tensors_b):
        if tensor_a.shape != tensor_b.shape:
            diff = TensorDifference(
                name=name,
                max_absolute_diff=float("inf"),
                max_relative_diff=float("inf"),
                abs_diff_index=[],
                rel_diff_index=[],
                absolute_tolerance=atol,
                relative_tolerance=rtol,
                tensor_shape=tensor_a.shape,
                error_message=f"Shape mismatch: {tensor_a.shape} vs {tensor_b.shape}",
            )
            differences.append(diff)
            mismatched_names.append(name)
            if verbose:
                print(str(diff))
            continue

        try:
            torch.testing.assert_close(tensor_a, tensor_b, atol=atol, rtol=rtol, check_device=False)
            differences.append(None)
        except AssertionError as e:
            abs_diff, rel_diff = compute_tensor_differences(tensor_a, tensor_b)
            max_abs_diff, max_abs_idx = get_max_difference_info(abs_diff)
            max_rel_diff, max_rel_idx = get_max_difference_info(rel_diff)

            diff = TensorDifference(
                name=name,
                max_absolute_diff=max_abs_diff,
                max_relative_diff=max_rel_diff,
                abs_diff_index=max_abs_idx,
                rel_diff_index=max_rel_idx,
                absolute_tolerance=atol,
                relative_tolerance=rtol,
                tensor_shape=tensor_a.shape,
                error_message=str(e),
            )
            differences.append(diff)
            mismatched_names.append(name)

            if verbose:
                print(str(diff))

    if mismatched_names and raise_on_mismatch:
        raise ValueError(f"Tensors did not match within tolerances for: {', '.join(mismatched_names)}")

    return differences
