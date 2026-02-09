/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <torch/extension.h>

using torch::Tensor;

std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute_topK_op(
    Tensor              input,
    Tensor              indices,
    int64_t             num_out_tokens,
    std::vector<Tensor> workspace,
    int64_t             max_expanded_token_num);

torch::Tensor moe_recover_topK_op(
    torch::Tensor  input,
    torch::Tensor  row_id_map,
    torch::Tensor  prob_opt,
    int64_t num_tokens,
    int64_t num_topK);

std::tuple<torch::Tensor, torch::Tensor> moe_recover_topK_bwd_op(
    Tensor  input_bwd,
    Tensor  input_fwd,
    Tensor  row_id_map,
    Tensor  prob);
