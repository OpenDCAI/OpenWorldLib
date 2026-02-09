#include <torch/extension.h>

void AsymmetricDualExpertGemm(
    torch::Tensor input_expert0,
    torch::Tensor input_expert1,
    torch::Tensor weight_expert0,
    torch::Tensor weight_expert1,
    torch::Tensor output_expert0,
    torch::Tensor output_expert1,
    bool trans_a, bool trans_b);
