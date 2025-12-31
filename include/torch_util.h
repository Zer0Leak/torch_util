#pragma once

#include <format>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <vector>
#include <xtensor/xexpression.hpp>
#include <xtensor/xmath.hpp>

#include <limits>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <xtensor/xexpression.hpp>
#include <xtensor/xmath.hpp>

// #include <ATen/core/Tensor.h>
// #include <ATen/core/Formatting.h>

template <> struct std::formatter<at::Tensor> : std::formatter<std::string_view> {
    auto format(const at::Tensor &t, std::format_context &ctx) const {
        std::ostringstream ss;
        ss << t;
        std::string s = ss.str();

        // LibTorch appends metadata starting with "\n["
        // We find the last occurrence to remove the Type/Shape summary
        size_t last_newline = s.find_last_of('\n');
        if (last_newline != std::string::npos) {
            size_t metadata_start = s.find('[', last_newline);
            if (metadata_start != std::string::npos) {
                // Truncate the string to remove the metadata line
                s.erase(last_newline);
            }
        }

        return std::formatter<std::string_view>::format(s, ctx);
    }
};

template <> struct std::formatter<c10::IntArrayRef> : std::formatter<std::string_view> {
    auto format(c10::IntArrayRef sizes, std::format_context &ctx) const {
        std::stringstream ss;
        ss << sizes; // Use LibTorch's built-in array printer
        return std::formatter<std::string_view>::format(ss.str(), ctx);
    }
};

namespace torch_u {

inline auto f32() -> torch::TensorOptions {
    auto device = torch::kCUDA; // torch::kCUDA or torch::kCPU
    return torch::TensorOptions().dtype(torch::kFloat32).device(device);
}

extern auto dbg(const torch::Tensor &t) -> std::string;

extern auto dbg(const c10::IntArrayRef &t) -> std::string;

extern auto dbgp(const torch::Tensor &t, std::optional<std::string_view> name = {}) -> void;

extern auto dbgp(const c10::IntArrayRef &t, std::optional<std::string_view> name = {}) -> void;

template <std::ranges::input_range RX, std::ranges::input_range RY>
    requires std::same_as<std::remove_cvref_t<std::ranges::range_reference_t<RX>>, torch::Tensor> &&
                 std::same_as<std::remove_cvref_t<std::ranges::range_reference_t<RY>>, torch::Tensor>
auto minmax(RX &&xs, RY &&ys) -> std::tuple<double, double, double, double> {
    auto init_min = (std::numeric_limits<double>::max)();
    auto init_max = (std::numeric_limits<double>::lowest)();

    double min_x = init_min, max_x = init_max;
    double min_y = init_min, max_y = init_max;

    auto update = [](double &mn, double &mx, const torch::Tensor &t) {
        // Expect CPU float contiguous; enforce defensively:
        auto tt = t.detach().to(torch::kCPU, torch::kFloat).contiguous();

        const float *p = tt.data_ptr<float>();
        const std::size_t n = static_cast<std::size_t>(tt.numel());
        if (n == 0)
            return;

        auto [mn_it, mx_it] = std::minmax_element(p, p + n);
        mn = std::min(mn, static_cast<double>(*mn_it));
        mx = std::max(mx, static_cast<double>(*mx_it));
    };

    for (const auto &t : xs)
        update(min_x, max_x, t);
    for (const auto &t : ys)
        update(min_y, max_y, t);

    return {min_x, max_x, min_y, max_y};
}

template <std::ranges::input_range RX, std::ranges::input_range RY>
    requires std::same_as<std::remove_cvref_t<std::ranges::range_reference_t<RX>>, torch::Tensor> &&
                 std::same_as<std::remove_cvref_t<std::ranges::range_reference_t<RY>>, torch::Tensor>
auto calc_pad(RX &&xs, RY &&ys, float pad = 0.1f) -> std::tuple<double, double, double, double> {
    auto [min_x, max_x, min_y, max_y] = minmax(xs, ys);
    auto range_x = max_x - min_x;
    auto range_y = max_y - min_y;
    auto min_x_pad = min_x - pad * range_x;
    auto max_x_pad = max_x + pad * range_x;
    auto min_y_pad = min_y - pad * range_y;
    auto max_y_pad = max_y + pad * range_y;
    return {min_x_pad, max_x_pad, min_y_pad, max_y_pad};
}

} // namespace torch_u

extern "C" {
const char *pt(const torch::Tensor *t);

const char *ptv(const torch::Tensor *t);

const char *ptf(const torch::Tensor *t);

const char *ps(const torch::Tensor *t);
}
