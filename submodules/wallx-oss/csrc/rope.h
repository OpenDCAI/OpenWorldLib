#include <torch/extension.h>

void launch_multimodal_rope_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin,
    torch::Tensor q_out, torch::Tensor k_out,
    std::vector<int> mrope_section_doubled
);

void launch_multimodal_rope_backward(
    torch::Tensor grad_q_out, torch::Tensor grad_k_out,
    torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin,
    torch::Tensor grad_q, torch::Tensor grad_k,
    std::vector<int> mrope_section_doubled
);
