#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <limits>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <vector>

#include "torch_util_plot.h"

namespace torch_u {

// ============================================================
//  Supported sample concept
// ============================================================

template <typename T>
concept MinMaxSample =
    std::same_as<std::remove_cvref_t<T>, torch::Tensor> || std::same_as<std::remove_cvref_t<T>, std::vector<double>>;

// ============================================================
//  Min/max extraction per sample type
// ============================================================

// --- torch::Tensor ---
inline void update_minmax(double &mn, double &mx, const torch::Tensor &t) {
    // Expect CPU double contiguous; enforce defensively
    auto tt = torch_u::plot_ready<double>(t);

    const double *p = tt.data_ptr<double>();
    const std::size_t n = static_cast<std::size_t>(tt.numel());
    if (n == 0)
        return;

    auto [mn_it, mx_it] = std::minmax_element(p, p + n);
    mn = std::min(mn, *mn_it);
    mx = std::max(mx, *mx_it);
}

// --- std::vector<double> ---
inline void update_minmax(double &mn, double &mx, const std::vector<double> &v) {
    if (v.empty())
        return;

    auto [mn_it, mx_it] = std::minmax_element(v.begin(), v.end());
    mn = std::min(mn, *mn_it);
    mx = std::max(mx, *mx_it);
}

// ============================================================
//  minmax for ranges
// ============================================================

template <std::ranges::input_range RX, std::ranges::input_range RY>
    requires MinMaxSample<std::ranges::range_reference_t<RX>> && MinMaxSample<std::ranges::range_reference_t<RY>>
auto minmax(RX &&xs, RY &&ys) -> std::tuple<double, double, double, double> {
    const double init_min = (std::numeric_limits<double>::max)();
    const double init_max = (std::numeric_limits<double>::lowest)();

    double min_x = init_min, max_x = init_max;
    double min_y = init_min, max_y = init_max;

    for (const auto &x : xs)
        update_minmax(min_x, max_x, x);

    for (const auto &y : ys)
        update_minmax(min_y, max_y, y);

    return {min_x, max_x, min_y, max_y};
}

// ============================================================
//  calc_pad for ranges
// ============================================================

template <std::ranges::input_range RX, std::ranges::input_range RY>
    requires MinMaxSample<std::ranges::range_reference_t<RX>> && MinMaxSample<std::ranges::range_reference_t<RY>>
auto calc_pad(RX &&xs, RY &&ys, double pad = 0.1) -> std::tuple<double, double, double, double> {
    auto [min_x, max_x, min_y, max_y] = minmax(xs, ys);

    const double range_x = max_x - min_x;
    const double range_y = max_y - min_y;

    const double min_x_pad = min_x - pad * range_x;
    const double max_x_pad = max_x + pad * range_x;
    const double min_y_pad = min_y - pad * range_y;
    const double max_y_pad = max_y + pad * range_y;

    return {min_x_pad, max_x_pad, min_y_pad, max_y_pad};
}

// ============================================================
//  Convenience overloads for single inputs
// ============================================================

inline auto calc_pad(const torch::Tensor &x, const torch::Tensor &y,
                     double pad = 0.1) -> std::tuple<double, double, double, double> {
    return calc_pad(std::array{x}, std::array{y}, pad);
}

inline auto calc_pad(const std::vector<double> &x, const std::vector<double> &y,
                     double pad = 0.1) -> std::tuple<double, double, double, double> {
    return calc_pad(std::array{x}, std::array{y}, pad);
}

}  // namespace torch_u