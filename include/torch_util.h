#pragma once

#include <format>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <torch/torch.h>

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

} // namespace torch_u

extern "C" {
const char *pt(const torch::Tensor *t);

const char *ptv(const torch::Tensor *t);

const char *ptf(const torch::Tensor *t);

const char *ps(const torch::Tensor *t);
}
