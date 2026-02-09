#include "dual_asym_grouped_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

#define NUM_STREAM 4

#define CUDA_CALL(code)                               \
    do                                                \
    {                                                 \
        cudaError_t status = code;                    \
        std::string err = cudaGetErrorString(status); \
        TORCH_CHECK(status == cudaSuccess, err);      \
    } while (0)

#define CUBLAS_CALL(code)                                             \
    do                                                                \
    {                                                                 \
        cublasStatus_t status = code;                                 \
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CuBLAS Error"); \
    } while (0)

#define GROUPED_GEMM_STRINGIFY_HELPER(x) #x
#define GROUPED_GEMM_STRINGIFY(x) \
GROUPED_GEMM_STRINGIFY_HELPER(x)

template <typename T>
torch::Tensor CopyToDevice(const std::vector<T> &x, const torch::Device &device)
{
    size_t bytes = x.size() * sizeof(T);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
    torch::Tensor out = torch::empty(bytes, options);

    CUDA_CALL(cudaMemcpyAsync(out.data_ptr(),
                                x.data(), bytes,
                                cudaMemcpyHostToDevice,
                                c10::cuda::getCurrentCUDAStream()));
    return out;
}

using DualExpertGemmKernelNN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ::cutlass::bfloat16_t,
    ::cutlass::layout::RowMajor,
    ::cutlass::ComplexTransform::kNone,
    8,
    ::cutlass::bfloat16_t,
    ::cutlass::layout::RowMajor,
    ::cutlass::ComplexTransform::kNone,
    8,
    ::cutlass::bfloat16_t,
    ::cutlass::layout::RowMajor,
    float,
    ::cutlass::arch::OpClassTensorOp,
    ::cutlass::arch::Sm80,
    ::cutlass::gemm::GemmShape<128, 128, 32>,
    ::cutlass::gemm::GemmShape<64, 64, 32>,
    ::cutlass::gemm::GemmShape<16, 8, 16>,
    ::cutlass::epilogue::thread::LinearCombination<::cutlass::bfloat16_t, 8, float, float>,
    ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

using DualExpertGemmKernelTN = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ::cutlass::bfloat16_t,
    ::cutlass::layout::ColumnMajor,
    ::cutlass::ComplexTransform::kNone,
    8,
    ::cutlass::bfloat16_t,
    ::cutlass::layout::RowMajor,
    ::cutlass::ComplexTransform::kNone,
    8,
    ::cutlass::bfloat16_t,
    ::cutlass::layout::RowMajor,
    float,
    ::cutlass::arch::OpClassTensorOp,
    ::cutlass::arch::Sm80,
    ::cutlass::gemm::GemmShape<128, 128, 32>,
    ::cutlass::gemm::GemmShape<64, 64, 32>,
    ::cutlass::gemm::GemmShape<16, 8, 16>,
    ::cutlass::epilogue::thread::LinearCombination<::cutlass::bfloat16_t, 8, float, float>,
    ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

using DualExpertGemmKernelNT = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ::cutlass::bfloat16_t,
    ::cutlass::layout::RowMajor,
    ::cutlass::ComplexTransform::kNone,
    8,
    ::cutlass::bfloat16_t,
    ::cutlass::layout::ColumnMajor,
    ::cutlass::ComplexTransform::kNone,
    8,
    ::cutlass::bfloat16_t,
    ::cutlass::layout::RowMajor,
    float,
    ::cutlass::arch::OpClassTensorOp,
    ::cutlass::arch::Sm80,
    ::cutlass::gemm::GemmShape<128, 128, 32>,
    ::cutlass::gemm::GemmShape<64, 64, 32>,
    ::cutlass::gemm::GemmShape<16, 8, 16>,
    ::cutlass::epilogue::thread::LinearCombination<::cutlass::bfloat16_t, 8, float, float>,
    ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

using DualExpertGemmKernelTT = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ::cutlass::bfloat16_t,
    ::cutlass::layout::ColumnMajor,
    ::cutlass::ComplexTransform::kNone,
    8,
    ::cutlass::bfloat16_t,
    ::cutlass::layout::ColumnMajor,
    ::cutlass::ComplexTransform::kNone,
    8,
    ::cutlass::bfloat16_t,
    ::cutlass::layout::RowMajor,
    float,
    ::cutlass::arch::OpClassTensorOp,
    ::cutlass::arch::Sm80,
    ::cutlass::gemm::GemmShape<128, 128, 32>,
    ::cutlass::gemm::GemmShape<64, 64, 32>,
    ::cutlass::gemm::GemmShape<16, 8, 16>,
    ::cutlass::epilogue::thread::LinearCombination<::cutlass::bfloat16_t, 8, float, float>,
    ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

using DualExpertGemmNN = ::cutlass::gemm::device::GemmGrouped<DualExpertGemmKernelNN>;
using DualExpertGemmTN = ::cutlass::gemm::device::GemmGrouped<DualExpertGemmKernelTN>;
using DualExpertGemmNT = ::cutlass::gemm::device::GemmGrouped<DualExpertGemmKernelNT>;
using DualExpertGemmTT = ::cutlass::gemm::device::GemmGrouped<DualExpertGemmKernelTT>;

template <typename Gemm>
typename Gemm::Arguments MakeAsymmetricArgumentsSeparated(
    torch::Tensor input_expert0,
    torch::Tensor input_expert1,
    torch::Tensor weight_expert0,
    torch::Tensor weight_expert1,
    torch::Tensor output_expert0,
    torch::Tensor output_expert1)
{

    TORCH_CHECK(input_expert0.dim() == 2 && input_expert1.dim() == 2,
                "Input tensors must be 2D");
    TORCH_CHECK(weight_expert0.dim() == 2 && weight_expert1.dim() == 2,
                "Weight tensors must be 2D");
    TORCH_CHECK(output_expert0.dim() == 2 && output_expert1.dim() == 2,
                "Output tensors must be 2D");

    using LayoutA = typename Gemm::LayoutA;
    using LayoutB = typename Gemm::LayoutB;
    using LayoutC = typename Gemm::LayoutC;

    bool a_is_column_major = std::is_same_v<LayoutA, cutlass::layout::ColumnMajor>;
    bool b_is_column_major = std::is_same_v<LayoutB, cutlass::layout::ColumnMajor>;

    int64_t m0, k0, n0;
    if (a_is_column_major)
    {
        m0 = input_expert0.size(1);
        k0 = input_expert0.size(0);
    }
    else
    {
        m0 = input_expert0.size(0);
        k0 = input_expert0.size(1);
    }

    if (b_is_column_major)
    {
        n0 = weight_expert0.size(0);
        TORCH_CHECK(weight_expert0.size(1) == k0, "Expert 0: k dimensions must match");
    }
    else
    {
        n0 = weight_expert0.size(1);
        TORCH_CHECK(weight_expert0.size(0) == k0, "Expert 0: k dimensions must match");
    }

    int64_t m1, k1, n1;
    if (a_is_column_major)
    {
        m1 = input_expert1.size(1);
        k1 = input_expert1.size(0);
    }
    else
    {
        m1 = input_expert1.size(0);
        k1 = input_expert1.size(1);
    }

    if (b_is_column_major)
    {
        n1 = weight_expert1.size(0);
        TORCH_CHECK(weight_expert1.size(1) == k1, "Expert 1: k dimensions must match");
    }
    else
    {
        n1 = weight_expert1.size(1);
        TORCH_CHECK(weight_expert1.size(0) == k1, "Expert 1: k dimensions must match");
    }

    std::vector<cutlass::gemm::GemmCoord> problem_sizes_host(2);
    problem_sizes_host[0] = cutlass::gemm::GemmCoord(m0, n0, k0);
    problem_sizes_host[1] = cutlass::gemm::GemmCoord(m1, n1, k1);

    int64_t num_experts = 2;
    int threadblock_count = Gemm::sufficient(problem_sizes_host.data(), num_experts);
    if (!threadblock_count)
    {
        TORCH_CHECK(false, "Dual Expert Grouped GEMM execution not possible with HW");
    }

    std::vector<int64_t> lda_host(num_experts);
    std::vector<int64_t> ldb_host(num_experts);
    std::vector<int64_t> ldc_host(num_experts);

    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementC = typename Gemm::ElementC;

    std::vector<ElementA *> ptr_a_host(num_experts);
    std::vector<ElementB *> ptr_b_host(num_experts);
    std::vector<ElementC *> ptr_c_host(num_experts);

    auto problem_0 = problem_sizes_host[0];
    lda_host[0] = LayoutA::packed({problem_0.m(), problem_0.k()}).stride(0);
    ldb_host[0] = LayoutB::packed({problem_0.k(), problem_0.n()}).stride(0);
    ldc_host[0] = LayoutC::packed({problem_0.m(), problem_0.n()}).stride(0);

    ptr_a_host[0] = (ElementA *)input_expert0.data_ptr();
    ptr_b_host[0] = (ElementB *)weight_expert0.data_ptr();
    ptr_c_host[0] = (ElementC *)output_expert0.data_ptr();

    auto problem_1 = problem_sizes_host[1];
    lda_host[1] = LayoutA::packed({problem_1.m(), problem_1.k()}).stride(0);
    ldb_host[1] = LayoutB::packed({problem_1.k(), problem_1.n()}).stride(0);
    ldc_host[1] = LayoutC::packed({problem_1.m(), problem_1.n()}).stride(0);

    ptr_a_host[1] = (ElementA *)input_expert1.data_ptr();
    ptr_b_host[1] = (ElementB *)weight_expert1.data_ptr();
    ptr_c_host[1] = (ElementC *)output_expert1.data_ptr();

    torch::Tensor lda = CopyToDevice(lda_host, input_expert0.device());
    torch::Tensor ldb = CopyToDevice(ldb_host, input_expert0.device());
    torch::Tensor ldc = CopyToDevice(ldc_host, input_expert0.device());
    torch::Tensor ptr_a = CopyToDevice(ptr_a_host, input_expert0.device());
    torch::Tensor ptr_b = CopyToDevice(ptr_b_host, input_expert0.device());
    torch::Tensor ptr_c = CopyToDevice(ptr_c_host, input_expert0.device());
    torch::Tensor problem_sizes = CopyToDevice(problem_sizes_host, input_expert0.device());

    typename Gemm::EpilogueOutputOp::Params epilogue_op(/*alpha=*/1.0f, /*beta=*/0.0f);
    typename Gemm::Arguments arguments(
        (cutlass::gemm::GemmCoord *)problem_sizes.data_ptr(),
        (int)num_experts,
        (int)threadblock_count,
        epilogue_op,
        (ElementA **)ptr_a.data_ptr(),
        (ElementB **)ptr_b.data_ptr(),
        (ElementC **)ptr_c.data_ptr(),
        (ElementC **)ptr_c.data_ptr(),
        (int64_t *)lda.data_ptr(),
        (int64_t *)ldb.data_ptr(),
        (int64_t *)ldc.data_ptr(),
        (int64_t *)ldc.data_ptr(),
        (cutlass::gemm::GemmCoord *)problem_sizes_host.data());

    return arguments;
}

template <typename Gemm>
void executeDualExpertGemm(
    torch::Tensor input_expert0,
    torch::Tensor input_expert1,
    torch::Tensor weight_expert0,
    torch::Tensor weight_expert1,
    torch::Tensor output_expert0,
    torch::Tensor output_expert1)
{

    Gemm gemm;

    auto arguments = MakeAsymmetricArgumentsSeparated<Gemm>(
        input_expert0, input_expert1,
        weight_expert0, weight_expert1,
        output_expert0, output_expert1);

    int64_t workspace_size = gemm.get_workspace_size(arguments);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(input_expert0.device());
    torch::Tensor workspace = torch::empty(workspace_size, options);

    if (gemm.initialize(arguments, workspace.data_ptr()) != cutlass::Status::kSuccess)
    {
        TORCH_CHECK(false, "Failed to initialize CUTLASS Asymmetric Dual Expert GEMM");
    }

    if (gemm.run(c10::cuda::getCurrentCUDAStream()) != cutlass::Status::kSuccess)
    {
        TORCH_CHECK(false, "Failed to run CUTLASS Asymmetric Dual Expert GEMM");
    }
}

void AsymmetricDualExpertGemm(
    torch::Tensor input_expert0,
    torch::Tensor input_expert1,
    torch::Tensor weight_expert0,
    torch::Tensor weight_expert1,
    torch::Tensor output_expert0,
    torch::Tensor output_expert1,
    bool trans_a, bool trans_b)
{

    TORCH_CHECK(input_expert0.device() == input_expert1.device() &&
                    input_expert0.device() == weight_expert0.device() &&
                    input_expert0.device() == weight_expert1.device() &&
                    input_expert0.device() == output_expert0.device() &&
                    input_expert0.device() == output_expert1.device(),
                "All tensors must be on the same device");

    TORCH_CHECK(input_expert0.device().is_cuda(),
                "All tensors must be on CUDA device for CUTLASS GEMM");

    TORCH_CHECK(input_expert0.dtype() == torch::kBFloat16,
                "All tensors must be BFloat16 for this kernel");

    torch::Tensor input0_contiguous = input_expert0.contiguous();
    torch::Tensor input1_contiguous = input_expert1.contiguous();
    torch::Tensor weight0_contiguous = weight_expert0.contiguous();
    torch::Tensor weight1_contiguous = weight_expert1.contiguous();
    torch::Tensor output0_contiguous = output_expert0.contiguous();
    torch::Tensor output1_contiguous = output_expert1.contiguous();

    if (!trans_a && !trans_b)
    {
        using Gemm = DualExpertGemmNN;
        executeDualExpertGemm<Gemm>(input0_contiguous, input1_contiguous,
                                    weight0_contiguous, weight1_contiguous,
                                    output0_contiguous, output1_contiguous);
    }
    else if (trans_a && !trans_b)
    {
        using Gemm = DualExpertGemmTN;
        executeDualExpertGemm<Gemm>(input0_contiguous, input1_contiguous,
                                    weight0_contiguous, weight1_contiguous,
                                    output0_contiguous, output1_contiguous);
    }
    else if (!trans_a && trans_b)
    {
        using Gemm = DualExpertGemmNT;
        executeDualExpertGemm<Gemm>(input0_contiguous, input1_contiguous,
                                    weight0_contiguous, weight1_contiguous,
                                    output0_contiguous, output1_contiguous);
    }
    else
    {
        using Gemm = DualExpertGemmTT;
        executeDualExpertGemm<Gemm>(input0_contiguous, input1_contiguous,
                                    weight0_contiguous, weight1_contiguous,
                                    output0_contiguous, output1_contiguous);
    }
}
