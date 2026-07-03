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

import torch


def compare_models_thoroughly(model1, model2, verbose=True):
    """
    Thoroughly compare two models by checking all parameters and buffers.

    Args:
        model1: First PyTorch model
        model2: Second PyTorch model
        verbose: If True, prints detailed comparison information

    Returns:
        dict: Comparison results containing:
            - mismatched_params: List of parameter names that don't match
            - mismatched_buffers: List of buffer names that don't match
            - is_identical: Boolean indicating if models are identical
    """

    def print_if_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    mismatches = {"mismatched_params": [], "mismatched_buffers": [], "is_identical": True}

    # Compare parameters
    print_if_verbose("\n=== Comparing Parameters ===")
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # Check parameter names
    param_names1 = set(params1.keys())
    param_names2 = set(params2.keys())
    if param_names1 != param_names2:
        mismatches["is_identical"] = False
        extra_in_1 = param_names1 - param_names2
        extra_in_2 = param_names2 - param_names1
        if extra_in_1:
            print_if_verbose(f"Parameters only in model1: {extra_in_1}")
            mismatches["mismatched_params"].extend(list(extra_in_1))
        if extra_in_2:
            print_if_verbose(f"Parameters only in model2: {extra_in_2}")
            mismatches["mismatched_params"].extend(list(extra_in_2))

    # Compare common parameters
    common_params = param_names1 & param_names2
    for name in common_params:
        param1 = params1[name]
        param2 = params2[name]

        # Compare shapes
        if param1.shape != param2.shape:
            mismatches["is_identical"] = False
            mismatches["mismatched_params"].append(name)
            print_if_verbose(f"Shape mismatch for parameter {name}:")
            print_if_verbose(f"  Model1: {param1.shape}")
            print_if_verbose(f"  Model2: {param2.shape}")
            continue

        # Compare values
        if not torch.equal(param1.cpu(), param2.cpu()):
            mismatches["is_identical"] = False
            mismatches["mismatched_params"].append(name)
            print_if_verbose(f"Value mismatch for parameter {name}")

    # Compare all buffers (both persistent and non-persistent)
    print_if_verbose("\n=== Comparing Buffers ===")
    buffers1 = dict(model1.named_buffers())
    buffers2 = dict(model2.named_buffers())

    # Check buffer names
    buffer_names1 = set(buffers1.keys())
    buffer_names2 = set(buffers2.keys())
    if buffer_names1 != buffer_names2:
        mismatches["is_identical"] = False
        extra_in_1 = buffer_names1 - buffer_names2
        extra_in_2 = buffer_names2 - buffer_names1
        if extra_in_1:
            print_if_verbose(f"Buffers only in model1: {extra_in_1}")
            mismatches["mismatched_buffers"].extend(list(extra_in_1))
        if extra_in_2:
            print_if_verbose(f"Buffers only in model2: {extra_in_2}")
            mismatches["mismatched_buffers"].extend(list(extra_in_2))

    # Compare common buffers
    common_buffers = buffer_names1 & buffer_names2
    for name in common_buffers:
        buf1 = buffers1[name]
        buf2 = buffers2[name]

        # Compare shapes
        if buf1.shape != buf2.shape:
            mismatches["is_identical"] = False
            mismatches["mismatched_buffers"].append(name)
            print_if_verbose(f"Shape mismatch for buffer {name}:")
            print_if_verbose(f"  Model1: {buf1.shape}")
            print_if_verbose(f"  Model2: {buf2.shape}")
            continue

        # Compare values
        try:
            if not torch.equal(buf1.cpu(), buf2.cpu()):
                mismatches["is_identical"] = False
                mismatches["mismatched_buffers"].append(name)
                print_if_verbose(f"Value mismatch for buffer {name}")
        except RuntimeError as e:
            print_if_verbose(f"Error comparing buffer {name}: {e}")
            mismatches["mismatched_buffers"].append(name)

    # Print summary
    print_if_verbose("\n=== Summary ===")
    print_if_verbose(f"Total parameters checked: {len(common_params)}")
    print_if_verbose(f"Total buffers checked: {len(common_buffers)}")
    print_if_verbose(f"Mismatched parameters: {len(mismatches['mismatched_params'])}")
    print_if_verbose(f"Mismatched buffers: {len(mismatches['mismatched_buffers'])}")
    print_if_verbose(f"Models are {'identical' if mismatches['is_identical'] else 'different'}")

    return mismatches
