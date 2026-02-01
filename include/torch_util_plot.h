#pragma once

#include <torch/torch.h>
#include <xtensor/xexpression.hpp>
#include <xtensor/xmath.hpp>

namespace torch_u {

extern torch::Tensor plot_ready(torch::Tensor t, torch::ScalarType dtype = torch::kFloat32, bool force_copy = true);

}  // namespace torch_u