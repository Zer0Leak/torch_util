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

// Scalar -> string (numeric)
[[gnu::used]] static inline std::string scalar_to_string(const torch::Tensor &s) {
    std::ostringstream ss;
    ss << s.item<double>();
    return ss.str();
}

// Render tensor values as nested, comma-separated brackets.
// Assumes x is detached and on CPU.
[[gnu::used]] static inline std::string render_tensor_values_compact(const torch::Tensor &x, int64_t axis_limit) {
    std::function<std::string(const torch::Tensor &)> render = [&](const torch::Tensor &t) -> std::string {
        const int64_t dim = t.dim();

        if (dim == 0) {
            return scalar_to_string(t);
        }

        const int64_t n = std::min<int64_t>(t.size(0), axis_limit);

        std::string out;
        out.push_back('[');

        for (int64_t i = 0; i < n; ++i) {
            if (i)
                out.push_back(',');
            out += render(t.select(0, i));
        }

        out.push_back(']');
        return out;
    };

    return render(x);
}

[[gnu::used]] static inline std::string tensor_first_slice_str(const torch::Tensor &t) {
    std::ostringstream oss;

    if (t.numel() == 0) {
        oss << "first=<empty> " << "Tensor(sizes=" << t.sizes() << ", dtype=" << t.dtype() << ", device=" << t.device()
            << ", requires_grad=" << (t.requires_grad() ? "true" : "false") << ")";
        return oss.str();
    }

    torch::Tensor x = t.detach();

    // Keep your "first slice" behavior: t[0,0,...,:]
    const int64_t dim = x.dim();
    if (dim > 0) {
        std::vector<torch::indexing::TensorIndex> idx;
        idx.reserve(dim);
        for (int64_t d = 0; d < dim - 1; ++d)
            idx.emplace_back(0);
        idx.emplace_back(torch::indexing::Slice());
        x = x.index(idx);
    }

    if (x.is_cuda()) {
        x = x.cpu();
    }

    const int64_t d = x.dim();
    const int64_t per_axis = (d <= 1) ? 32 : std::max<int64_t>(1, 32 / d);

    // Value first
    oss << "first=" << torch_u::render_tensor_values_compact(x, per_axis) << " ";

    // Then metadata
    oss << "Tensor(sizes=" << t.sizes() << ", dtype=" << t.dtype() << ", device=" << t.device()
        << ", requires_grad=" << (t.requires_grad() ? "true" : "false") << ")";

    return oss.str();
}

} // namespace torch_u

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
    buf = torch_u::tensor_full_str(*t);
    return buf.c_str();
}

const char *ptf(const torch::Tensor *t) {
    static thread_local std::string buf;
    if (!t)
        return "Error: Tensor is null";
    buf = torch_u::tensor_first_slice_str(*t);
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