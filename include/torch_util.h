#pragma once

#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <torch/torch.h>

template <> struct std::formatter<at::Tensor> : std::formatter<std::string_view> {
    auto format(const at::Tensor &t, std::format_context &ctx) const {
        std::stringstream ss;
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

namespace torch_u {

inline auto f32_cuda() -> torch::TensorOptions {
    return torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
}

} // namespace torch_u
