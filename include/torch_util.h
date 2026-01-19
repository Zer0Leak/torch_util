#pragma once

#include <torch_util_pad.h>
#include <torch_util_plot.h>

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

#include <algorithm>
#include <array>
#include <limits>
#include <ranges>
#include <tuple>
#include <type_traits>

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
        ss << sizes;  // Use LibTorch's built-in array printer
        return std::formatter<std::string_view>::format(ss.str(), ctx);
    }
};

namespace torch_u {

struct FormatSettings {
    std::ios_base &(*fmt)(std::ios_base &) = std::scientific;  // std::fixed / std::scientific
    std::_Setprecision precision = std::setprecision(6);       // std::setprecision(6)
    std::ios_base &(*align)(std::ios_base &) = std::right;     // std::right / std::left;
    std::_Setw width = std::setw(0);
};

extern FormatSettings g_default_format_settings;

struct FormatGuard {
    FormatSettings old_settings;  // Where the "default" is saved

    // The constructor takes the NEW settings (created via designated init)
    FormatGuard(FormatSettings new_settings)
        : old_settings(g_default_format_settings)  // STEP A: Copy current globals into old_settings
    {
        g_default_format_settings = new_settings;  // STEP B: Overwrite globals with the new user settings
    }

    ~FormatGuard() {
        g_default_format_settings = old_settings;  // STEP C: Restore globals from the saved copy
    }
};

inline auto f32() -> torch::TensorOptions {
    auto device = torch::kCUDA;  // torch::kCUDA or torch::kCPU
    return torch::TensorOptions().dtype(torch::kFloat32).device(device);
}

inline auto f64() -> torch::TensorOptions {
    auto device = torch::kCUDA;  // torch::kCUDA or torch::kCPU
    return torch::TensorOptions().dtype(torch::kFloat64).device(device);
}

inline auto ptp = [](const torch::Tensor &X) -> torch::Tensor { return std::get<0>(X.max(0)) - std::get<0>(X.min(0)); };

extern auto dbg(const torch::Tensor &t) -> std::string;

extern auto dbg(const c10::IntArrayRef &t) -> std::string;

extern auto dbgp(const torch::Tensor &t, std::optional<std::string_view> name = {}) -> void;

extern auto dbgp(const c10::IntArrayRef &t, std::optional<std::string_view> name = {}) -> void;

extern std::string tstr(const torch::Tensor &t, bool indent = true);

}  // namespace torch_u

extern "C" {
const char *pt(const torch::Tensor *t);

const char *ptv(const torch::Tensor *t);

const char *dtv(const torch::Tensor *t);

const char *ps(const torch::Tensor *t);
}
