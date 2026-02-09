/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "permute.h"

#include <torch/torch.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ATen/cuda/CUDAContext.h"

#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"


using torch::Tensor;


template <typename T>
inline T *get_ptr(torch::Tensor &t)
{
    return reinterpret_cast<T *>(t.data_ptr());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Top K
//
/////////////////////////////////////////////////////////////////////////////////////////////////

static __global__ void moe_permute_topK_row_map(
    const int *sorted_row_id,
    int *row_id_map,
    const int num_rows,
    const int num_topK,
    const int num_out_tokens)
{
    // Each block corresponds to one source token
    // row_id_map[num_topK][num_rows]
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;

    if (idx >= num_rows * num_topK)
        return;

    int source_row = sorted_row_id[idx];
    int source_token_id = source_row / num_topK;
    int source_topK_id = source_row % num_topK;

    if (idx >= num_out_tokens)
    {
        row_id_map[source_topK_id * num_rows + source_token_id] = -1;
    }
    else
    {
        row_id_map[source_topK_id * num_rows + source_token_id] = idx;
    }
}

template <typename T, typename TCompute, int kElementsPerAccess, bool hasProb>
__global__ void moe_recover_topK_kernel(const T *input,
                                        T *unpermuted_output,
                                        const int *row_id_map,
                                        const float *prob,
                                        const int num_rows,
                                        const int num_topK,
                                        const int num_cols)
{
    extern __shared__ int8_t s_mem[];
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<TCompute, kElementsPerAccess>;

    cutlass::NumericArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    cutlass::NumericArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    // each block corresponds to one source token
    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        for (int i = tid; i < num_topK; i += blockDim.x * blockDim.y)
        {
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
        __syncthreads();
    }

    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        FragmentLoadStore frag_load_store;
        FragmentCompute frag_elem;
        FragmentCompute frag_sum;

        int source_row = row_id_map[source_token];

        if (source_row != -1)
        {
            const T *source_row_ptr = input + source_row * num_cols;

            cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                frag_load_store, (source_row_ptr + i), true);
            frag_sum = src_converter(frag_load_store);

            if (hasProb)
            {
                frag_sum = frag_sum * s_prob[0];
            }
        }
        else
        {
            frag_sum.clear();
        }

        for (int k = 1; k < num_topK; k++)
        {
            source_row = row_id_map[k * num_rows + source_token];

            if (source_row == -1)
                continue;

            const T *source_row_ptr = input + source_row * num_cols;

            cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                frag_load_store, (source_row_ptr + i), true);
            frag_elem = src_converter(frag_load_store);

            if (hasProb)
            {
                frag_elem = frag_elem * s_prob[k];
            }

            for (int e = 0; e < kElementsPerAccess; e++)
            {
                frag_sum.at(e) = frag_sum.at(e) + frag_elem.at(e);
            }
        }

        T *dest_row_ptr = unpermuted_output + source_token * num_cols;
        frag_load_store = dst_converter(frag_sum);
        *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.data());
    }
}

template <typename T,
          typename TCompute,
          int kElementsPerAccess,
          int topKTile,
          bool hasProb>
__global__ void moe_permute_topK_kernel(const T *input_bwd,
                                        const T *input_fwd,
                                        T *act_grad,
                                        const float *prob,
                                        float *prob_grad,
                                        const int *row_id_map,
                                        const int num_rows,
                                        const int num_topK,
                                        const int num_cols)
{
    extern __shared__ int8_t s_mem[];
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<TCompute, kElementsPerAccess>;

    cutlass::NumericArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    cutlass::NumericArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        for (int i = tid; i < num_topK; i += blockDim.x)
        {
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
        __syncthreads();
    }

    float accum[topKTile] = {0.0f};
    FragmentLoadStore frag_load_store;

    const T *source_row_ptr = input_bwd + source_token * num_cols;
    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
            frag_load_store, (source_row_ptr + i), true);
        FragmentCompute frag_src = src_converter(frag_load_store);

        int index = source_token;

        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK) break;

            int dest_row = row_id_map[index];
            index += num_rows;

            if (dest_row == -1)
                continue;

            if (hasProb)
            {
                frag_load_store = dst_converter(frag_src * s_prob[k]);
            }
            else
            {
                frag_load_store = dst_converter(frag_src);
            }

            T *dest_row_ptr = act_grad + dest_row * num_cols;
            *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.data());

            if (hasProb)
            {
                const T *input_fwd_ptr = input_fwd + dest_row * num_cols;
                cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                    frag_load_store, (input_fwd_ptr + i), true);
                FragmentCompute frag_input_fwd = src_converter(frag_load_store);

                for (int e = 0; e < kElementsPerAccess; e++)
                {
                    accum[k] += float(frag_src.at(e) * frag_input_fwd.at(e));
                }
            }
        }
    }

    if (hasProb)
    {
        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK) break;

            for (int mask = 16; mask > 0; mask /= 2)
            {
                accum[k] = accum[k] + __shfl_xor_sync(0xffffffff, accum[k], mask, 32);
            }
        }

        if (tid == 0)
        {
            for (int k = 0; k < topKTile; k++)
            {
                if (k == num_topK) break;
                prob_grad[source_token * num_topK + k] = accum[k];
            }
        }
    }
}


template <typename T, typename TCompute, bool FWD, int kElementsPerAccess>
void moe_permute_topK_kernel_launcher(
    const T *input,
    T *output,
    const int *sorted_row_id,
    int *row_id_map,
    const float *prob,
    const int num_rows,
    const int num_topK,
    const int num_cols,
    const int num_out_tokens,
    cudaStream_t stream,
    float *prob_grad = nullptr,
    const T *input_fwd = nullptr)
{
    if (FWD)
    {
        if (prob_grad == nullptr)
        {
            // permute_topK fwd
            int threads = 64;
            int blocks = (num_rows * num_topK + threads - 1) / threads;
            moe_permute_topK_row_map<<<blocks, threads, 0, stream>>>(
                sorted_row_id,
                row_id_map,
                num_rows,
                num_topK,
                num_out_tokens);

            blocks = num_rows;
            threads = std::min(num_cols / kElementsPerAccess, 1024);
            moe_permute_topK_kernel<T, T, kElementsPerAccess, 128, false><<<blocks, threads, 0, stream>>>(
                input,
                nullptr,
                output,
                nullptr,
                nullptr,
                row_id_map,
                num_rows,
                num_topK,
                num_cols);
        }
        else
        {
            // unpermute_topK bwd
            int blocks = num_rows;
            int threads = 32;
            size_t smem_bytes = num_topK * sizeof(TCompute);

            if (num_topK == 1)
            {
                moe_permute_topK_kernel<T, T, kElementsPerAccess, 1, false><<<blocks, threads, 0, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 8)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 8, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 16)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 16, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 32)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 32, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 64)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 64, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 128)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 128, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else
            {
                throw std::runtime_error("num_topK cannot exceed 128.");
            }
        }
    }
    else
    {
        int blocks = num_rows;
        int threads = std::min(num_cols / kElementsPerAccess, 1024);
        size_t smem_bytes = num_topK * sizeof(TCompute);


        if (num_topK == 1)
        {
            // permute_topK bwd with topK==1
            moe_recover_topK_kernel<T, T, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
        else if (prob == nullptr)
        {
            // permute_topK bwd
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
        else
        {
            // unpermute_topK fwd
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, true><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Permute_topK OP
//
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute_topK_op(
    Tensor              input,
    Tensor              indices,
    int64_t             num_out_tokens,
    std::vector<Tensor> workspace,
    int64_t             max_expanded_token_num)
{
    const int num_tokens = input.size(0);
    const int num_cols = input.size(1);
    const int num_topK = indices.size(1);

    // initialize the workspace on the first run
    if (workspace.empty()) {
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

        Tensor sorted_indices = torch::empty(max_expanded_token_num, options);
        Tensor row_id = torch::range(0, max_expanded_token_num - 1, 1, options);
        Tensor sorted_row_id =
            torch::empty(max_expanded_token_num, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

        size_t temp_storage_bytes = 0;
        int *temp_ptr = nullptr;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                        temp_ptr, temp_ptr,
                                        temp_ptr, temp_ptr, max_expanded_token_num);
        Tensor temp_storage =
            torch::empty(temp_storage_bytes, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(sorted_indices);
        workspace.push_back(row_id);
        workspace.push_back(sorted_row_id);
        workspace.push_back(temp_storage);
    }

    int *indices_ptr = get_ptr<int>(indices);
    int *sorted_indices_ptr = get_ptr<int>(workspace[0]);
    int *row_id_ptr = get_ptr<int>(workspace[1]);
    int *sorted_row_id_ptr = get_ptr<int>(workspace[2]);

    void *d_temp_storage = get_ptr<void>(workspace[3]);
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    indices_ptr, sorted_indices_ptr,
                                    row_id_ptr, sorted_row_id_ptr, num_tokens * num_topK);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    num_out_tokens = (num_out_tokens > 0) ? num_out_tokens : num_tokens * num_topK;
    Tensor permuted_output =
        torch::empty({num_out_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor row_id_map =
        torch::empty({num_tokens * num_topK}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 4>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = cutlass::bfloat16_t;
        using dTypeCompute = cutlass::bfloat16_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(permuted_output, row_id_map, workspace);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Unpermute_topK OP
//
/////////////////////////////////////////////////////////////////////////////////////////////////

Tensor moe_recover_topK_op(
    Tensor  input,
    Tensor  row_id_map,
    Tensor  prob,
    int64_t num_tokens,
    int64_t num_topK)
{
    const int num_cols = input.size(1);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    Tensor unpermuted_output =
        torch::empty({num_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = (prob.defined()) ? get_ptr<float>(prob) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 4>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = cutlass::bfloat16_t;
        using dTypeCompute = cutlass::bfloat16_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return unpermuted_output;
}

std::tuple<Tensor, Tensor> moe_recover_topK_bwd_op(
    Tensor  input_bwd,
    Tensor  input_fwd,
    Tensor  row_id_map,
    Tensor  prob)
{
    const int num_tokens = prob.size(0);
    const int num_topK = prob.size(1);
    const int num_cols = input_bwd.size(1);

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = get_ptr<float>(prob);

    // activations type
    const at::ScalarType _st = input_bwd.scalar_type();

    // Output buffer alloc
    Tensor act_grad =
        torch::empty({input_fwd.size(0), num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor prob_grad =
        torch::empty({num_tokens, num_topK}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    float *prob_grad_ptr = get_ptr<float>(prob_grad);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 4>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = cutlass::bfloat16_t;
        using dTypeCompute = cutlass::bfloat16_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(act_grad, prob_grad);
}
