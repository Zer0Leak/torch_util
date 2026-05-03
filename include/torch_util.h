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
#include <xtensor/core/xexpression.hpp>
#include <xtensor/core/xmath.hpp>

#include <limits>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <xtensor/core/xexpression.hpp>
#include <xtensor/core/xmath.hpp>

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

// Scaler to sting
extern std::string tsstr(const torch::Tensor &s);

template <typename Loss, typename Optimizer>
void fit_fullbatch(torch::nn::Sequential &model, const torch::Tensor &X, const torch::Tensor &Y, Loss &loss,
                   Optimizer &optimizer, int epochs, bool verbose = false) {
    model->train();

    TORCH_CHECK(X.dim() == 2, "X must be (N,D)");
    TORCH_CHECK(Y.dim() == 2 && Y.size(1) == 1, "Y must be (N,1)");
    TORCH_CHECK(X.size(0) == Y.size(0), "X/Y batch mismatch");

    for (int e = 0; e < epochs; ++e) {
        optimizer.zero_grad();

        auto pred = model->forward(X);
        auto L = loss(pred, Y);

        if (verbose) {
            std::cout << "epoch " << (e + 1) << " loss = " << L.template item<double>() << '\n';
        }

        L.backward();
        optimizer.step();
    }
}

template <typename Loss, typename Optimizer>
void fit_minibatch(torch::nn::Sequential &model, torch::Tensor X, torch::Tensor Y, Loss &loss, Optimizer &optimizer,
                   int epochs, int64_t batch_size = 32, bool verbose = false) {
    TORCH_CHECK(torch::GradMode::is_enabled(), "GradMode is disabled");
    TORCH_CHECK(!c10::InferenceMode::is_enabled(), "InferenceMode is enabled");

    torch::AutoGradMode enable_grad(true);
    c10::InferenceMode inference_guard(false);

    model->train();

    if constexpr (std::is_same_v<Loss, torch::nn::CrossEntropyLoss>) {
        TORCH_CHECK(Y.dim() == 1, "CE: Y must be (N)");
        TORCH_CHECK(Y.scalar_type() == torch::kLong, "CE: Y must be int64");
    } else if constexpr (std::is_same_v<Loss, torch::nn::MSELoss>) {
        TORCH_CHECK(Y.dim() == 2 && Y.size(1) == 1, "MSE: Y must be (N,1)");
    } else if constexpr (std::is_same_v<Loss, torch::nn::BCEWithLogitsLoss> or
                         std::is_same_v<Loss, torch::nn::BCELoss>) {
        TORCH_CHECK(Y.dim() == 2 && Y.size(1) == 1, "BCE: Y must be (N,1)");
        TORCH_CHECK(Y.is_floating_point(), "BCE: Y must be float");
    } else {
        static_assert(!sizeof(Loss), "Unsupported loss type");
    }

    TORCH_CHECK(X.size(0) == Y.size(0), "X/Y batch mismatch");

    const auto N = X.size(0);
    auto device = X.device();
    auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);

    for (int e = 0; e < epochs; ++e) {
        const bool verbose_print = verbose && ((e % std::max<int64_t>(1, epochs / 100) == 0) || e == epochs - 1);

        if (verbose_print) {
            std::cout << "Epoch " << (e + 1) << "/" << epochs << " " << std::flush;
        }

        auto perm = torch::randperm(N, idx_opts);  // shuffle each epoch
        double epoch_cost_sum = 0.0;

        for (int64_t start = 0; start < N; start += batch_size) {
            if (verbose_print) {
                if (start % std::max<int64_t>(1, N / 10) == 0) {
                    std::cout << "=" << std::flush;
                }
            }
            int64_t end = std::min(start + batch_size, N);
            auto idx = perm.slice(0, start, end);

            // std::cout << "idx max: " << idx.max().item<int64_t>() << std::endl;

            auto xb = X.index_select(0, idx);
            auto yb = Y.index_select(0, idx);

            optimizer.zero_grad();

            auto prediction = model->forward(xb);
            auto L = loss(prediction, yb);

#if 0
            if (verbose_print) {
                bool second_or_unique = (batch_size > N ? start == batch_size : start == 0);
                if (second_or_unique) {
                    std::cout << "\n    GradMode enabled: " << torch::GradMode::is_enabled() << "\n";
                    std::cout << "    InferenceMode enabled: " << c10::InferenceMode::is_enabled() << "\n";

                    for (const auto &p : model->parameters()) {
                        std::cout << "    param requires_grad=" << p.requires_grad()
                                  << " grad_defined=" << p.grad().defined() << " sizes=" << p.sizes() << "\n";
                        break;  // just first param
                    }

                    std::cout << "    xb requires_grad=" << xb.requires_grad() << "\n";
                    std::cout << "    prediction requires_grad=" << prediction.requires_grad() << "\n";
                    std::cout << "    L requires_grad=" << L.requires_grad() << "\n";
                }
            }
#endif

            L.backward();
            optimizer.step();

            epoch_cost_sum += L.template item<double>() * (end - start);
        }

        if (verbose_print) {
            double epoch_cost = epoch_cost_sum / static_cast<double>(N);
            std::cout << " cost = " << epoch_cost << std::endl;
        }
    }
}

#include <torch/torch.h>
#include <tuple>

// ------------------------------------------------------------
// gen_data
// ------------------------------------------------------------
// generate a data set based on x^2 with added noise
//
// Python reference:
//
// def gen_data(m, seed=1, scale=0.7):
//     c = 0
//     x_train = np.linspace(0,49,m)
//     np.random.seed(seed)
//     y_ideal = x_train**2 + c
//     y_train = y_ideal + scale * y_ideal*(np.random.sample((m,))-0.5)
//     x_ideal = x_train
//     return x_train, y_train, x_ideal, y_ideal
//
extern auto gen_data(std::int64_t m, std::uint64_t seed = 1, double scale = 0.7, torch::Device device = torch::kCUDA,
                     torch::ScalarType dtype = torch::kFloat64)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

}  // namespace torch_u

extern "C" {
const char *pt(const torch::Tensor *t);

const char *ptv(const torch::Tensor *t);

const char *dtv(const torch::Tensor *t);

const char *ps(const torch::Tensor *t);
}
