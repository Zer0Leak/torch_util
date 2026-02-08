#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include <format>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "torch_util.h"

namespace torch_u {

FormatSettings g_default_format_settings;

[[gnu::used]] auto dbg_tensor(const torch::Tensor &t) -> std::string { return std::format("{}", t); }

[[gnu::used]] auto dbg(const c10::IntArrayRef &t) -> std::string { return std::format("{}", t); }

[[gnu::used]] auto dbgp(const torch::Tensor &t, std::optional<std::string_view> name) -> void {
    if (name.has_value()) {
        std::println("{}:\n{}", *name, t);
    } else {
        std::println("{}", t);
    }
}

[[gnu::used]] auto dbgp(const c10::IntArrayRef &t, std::optional<std::string_view> name) -> void {
    if (name.has_value()) {
        std::println("{}: {}", *name, t);
    } else {
        std::println("{}", t);
    }
}

// Full tensor print (can be large / can sync GPU if CUDA tensor is printed).
[[gnu::used]] static inline std::string tensor_full_str(const torch::Tensor &t) {
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

// Fast-ish header for watch windows: shape + dtype + device (+ grad flag).
// Avoids dumping values and generally avoids expensive ops.
[[gnu::used]] static inline std::string tensor_header_str(const torch::Tensor &t) {
    std::ostringstream oss;
    oss << "Tensor(sizes=" << t.sizes() << ", dtype=" << t.dtype() << ", device=" << t.device()
        << ", requires_grad=" << (t.requires_grad() ? "true" : "false") << ")";
    return oss.str();
}

[[gnu::used]] static inline std::string sizes_str(c10::IntArrayRef s) {
    std::ostringstream oss;
    oss << s;
    return oss.str();
}

#include <sstream>
#include <stdexcept>
#include <torch/torch.h>

std::string tsstr(const torch::Tensor &s) {
    if (!s.defined()) {
        throw std::invalid_argument("scalar_to_string: tensor is undefined");
    }

    if (s.dim() != 0) {
        throw std::invalid_argument("scalar_to_string: tensor is not scalar");
    }

    std::ostringstream ss;
    ss << g_default_format_settings.fmt << g_default_format_settings.precision << g_default_format_settings.align
       << g_default_format_settings.width;

    switch (s.scalar_type()) {

    // --- Floating point ---
    case torch::kFloat32:
        ss << s.item<float>();
        break;
    case torch::kFloat64:
        ss << s.item<double>();
        break;

    // --- Signed integers ---
    case torch::kInt8:
        ss << static_cast<int>(s.item<int8_t>());  // avoid char printing
        break;
    case torch::kInt16:
        ss << s.item<int16_t>();
        break;
    case torch::kInt32:
        ss << s.item<int32_t>();
        break;
    case torch::kInt64:
        ss << s.item<int64_t>();
        break;

    // --- Unsigned integers ---
    case torch::kUInt8:
        ss << static_cast<unsigned int>(s.item<uint8_t>());
        break;

    // --- Boolean ---
    case torch::kBool:
        ss << (s.item<bool>() ? 1 : 0);
        break;

    default:
        throw std::invalid_argument(std::string("scalar_to_string: unsupported dtype ") +
                                    c10::toString(s.scalar_type()));
    }

    return ss.str();
}

// Render tensor values as nested, comma-separated brackets (no spaces/newlines).
// Assumes x is detached. Will work on CPU tensors; if CUDA, move to CPU before calling.
[[gnu::used]] static inline std::string render_tensor_values_compact(const torch::Tensor &x,
                                                                     const int64_t max_scalar_to_show,
                                                                     const int64_t max_scalars_to_show_per_1D_vector,
                                                                     bool indent = false) {
    const auto indent_size = 4;
    if (x.dim() == 0) {
        return "(" + tsstr(x) + ")";
    }
    int64_t scalars_shown = 0;
    bool scalar_dropped = false;

    std::function<std::string(const torch::Tensor &, int)> render = [&](const torch::Tensor &t,
                                                                        int level) -> std::string {
        const int64_t ndim = t.dim();
        if (ndim == 0) {
            scalars_shown++;
            return tsstr(t);
        }

        auto indentation = std::string(level * indent_size, ' ');
        int64_t n = t.sizes()[0];
        if (ndim == 1) {
            n = std::min(t.sizes()[0], max_scalars_to_show_per_1D_vector);
        }
        const bool print_etc = (t.sizes()[0] > n);

        std::string out;

        if (indent) {
            out += indentation + "[";
            if (ndim > 1) {
                out += "\n";
            }
        } else {
            out += "[";
        }

        for (int64_t i = 0; i < n; ++i) {
            if (scalars_shown >= max_scalar_to_show) {
                scalar_dropped = true;
                break;
            }
            if (i > 0) {
                out.push_back(',');
                if (indent) {
                    if (ndim > 1) {
                        out += "\n";
                    } else {
                        out += " ";
                    }
                }
            }
            auto next_level = level + 1;
            out += render(t.select(0, i), next_level);
        }

        if (ndim == 1 && print_etc) {
            out += ",...";
        }

        auto is_root = level == 0;  // t.sizes() == x.sizes();

        if (indent) {
            if (ndim > 1) {
                if (scalar_dropped) {
                    out += ",";
                }
                out += "\n";
            }
        }

        if (scalar_dropped) {
            if (is_root) {
                if (ndim > 1) {
                    auto prev_indentation = std::string((1) * indent_size, ' ');
                    if (indent) {
                        out += prev_indentation + "...\n";
                    } else {
                        out += ",...";
                    }
                } else {  // ndim == 1
                    if (!print_etc) {
                        out += ",...";
                    }
                }
            }
        }

        if (indent && ndim > 1) {
            out += indentation;
        }
        out += "]";

        return out;
    };

    auto output = render(x, 0);

    return output;
}

std::string tstr(const torch::Tensor &t, bool indent) {
    std::ostringstream oss;
    const auto indent_size = 4;

    if (t.numel() == 0) {
        oss << "<empty>, " << "(shape=" << t.sizes() << ", dtype=" << t.dtype() << ", dev=" << t.device()
            << ", req_grad=" << (t.requires_grad() ? "true" : "false") << ")";
        return oss.str();
    }

    torch::Tensor x = t.detach();
    const int64_t dim = x.dim();

    if (x.is_cuda()) {
        x = x.cpu();
    }

    const int64_t max_scalar_to_show = indent ? 64 : 32;

    int64_t max_scalars_to_show_per_1D_vector = max_scalar_to_show;
    if (dim > 1) {
        const int64_t n = x.sizes().back();
        const int64_t last_dim_vectors = x.numel() / n;
        max_scalars_to_show_per_1D_vector = max_scalar_to_show / last_dim_vectors;
        max_scalars_to_show_per_1D_vector = std::max<int64_t>(max_scalars_to_show_per_1D_vector, 1);
    }

    // Then metadata
    oss << "(shape=" << t.sizes() << ", dtype=" << t.dtype() << ", dev=" << t.device()
        << ", req_grad=" << (t.requires_grad() ? "true" : "false") << ")";

    if (indent) {
        oss << "\n" << std::string(indent_size, ' ');
    } else {
        oss << " ";
    }

    // Value first (no spaces/newlines in rendering)
    oss << torch_u::render_tensor_values_compact(x, max_scalar_to_show, max_scalars_to_show_per_1D_vector, indent);

    return oss.str();
}

torch::Tensor plot_ready(torch::Tensor t, torch::ScalarType dtype, bool force_copy) {
    // Break autograd graph
    t = t.detach();

    const bool need_cpu = !t.device().is_cpu();
    const bool need_dtype = (t.scalar_type() != dtype);

    if (need_cpu || need_dtype) {
        // copy=false is fine; PyTorch will still copy if required (device/dtype change)
        t = t.to(torch::TensorOptions().device(torch::kCPU).dtype(dtype),
                 /*non_blocking=*/false,
                 /*copy=*/false);
    }

    // Plotting typically expects contiguous CPU memory
    if (!t.is_contiguous())
        t = t.contiguous();

    // If you need a stable buffer that won't alias training tensors, force a clone
    if (force_copy)
        t = t.clone();

    return t;
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
auto gen_data(std::int64_t m, std::uint64_t seed, double scale, torch::Device device,
              torch::ScalarType dtype) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> {
    // deterministic RNG (matches np.random.seed)
    torch::manual_seed(seed);

    // x_train = linspace(0, 49, m)
    auto X_train = torch::linspace(
                       /*start=*/0.0,
                       /*end=*/49.0,
                       /*steps=*/m, torch::TensorOptions().dtype(dtype).device(device))
                       .unsqueeze(-1);

    // y_ideal = x_train**2
    auto Y_ideal = X_train.square();

    // noise = scale * y_ideal * (rand(m) - 0.5)
    auto noise = scale * Y_ideal * (torch::rand({m, 1}, X_train.options()) - 0.5);

    // y_train = y_ideal + noise
    auto Y_train = Y_ideal + noise;

    // x_ideal = x_train (explicit clone to match Python semantics)
    auto X_ideal = X_train.clone();

    return {X_train, Y_train, X_ideal, Y_ideal};
}

}  // namespace torch_u

extern "C" {
// Tensor summary for watch list (header only).
const char *pt(const torch::Tensor *t) {
    static thread_local std::string buf;
    if (!t)
        return "Error: Tensor is null";
    buf = torch_u::tensor_header_str(*t);
    return buf.c_str();
}

// Full tensor print (values) – use explicitly, not for watch summaries.
const char *ptv(const torch::Tensor *t) {
    static thread_local std::string buf;
    if (!t)
        return "Error: Tensor is null";
    buf = torch_u::tstr(*t, true);
    return buf.c_str();
}

const char *dtv(const torch::Tensor *t) {
    static thread_local std::string buf;
    if (!t)
        return "Error: Tensor is null";
    buf = torch_u::tstr(*t, false);
    return buf.c_str();
}

// Shape-only helper.
const char *ps(const torch::Tensor *t) {
    static thread_local std::string buf;
    if (!t)
        return "Error: Tensor is null";
    buf = torch_u::sizes_str(t->sizes());
    return buf.c_str();
}
}