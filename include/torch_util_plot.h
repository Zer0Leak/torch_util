#pragma once

#include <torch/torch.h>
#include <xtensor/xexpression.hpp>
#include <xtensor/xmath.hpp>

namespace torch_u {

template <typename T> struct torch_scalar;

template <> struct torch_scalar<bool> {
    static constexpr torch::ScalarType value = torch::kBool;
};

template <> struct torch_scalar<float> {
    static constexpr torch::ScalarType value = torch::kFloat;
};

template <> struct torch_scalar<double> {
    static constexpr torch::ScalarType value = torch::kDouble;
};

template <typename T> inline torch::Tensor plot_ready(const torch::Tensor &t) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, bool>,
                  "plot_ready<T>: T must be float, double, or bool");

    torch::Tensor out = t.detach();

    const bool need_cpu = !out.device().is_cpu();
    const bool need_dtype = (out.scalar_type() != torch_scalar<T>::value);

    if (need_cpu || need_dtype) {
        out = out.to(torch::TensorOptions().device(torch::kCPU).dtype(torch_scalar<T>::value),
                     /*non_blocking=*/false,
                     /*copy=*/false);
    }

    if (!out.is_contiguous())
        out = out.contiguous();

    return out;
}

}  // namespace torch_u