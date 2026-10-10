#include "rbfl.hpp"

#include <stdexcept>

namespace rbfl {

bool less(Score a, Score b, bool use_forget) {
    if (a.err != b.err) return a.err < b.err;
    if (use_forget && a.forget != b.forget) return a.forget < b.forget;
    return a.cost < b.cost;
}

Model::Model(unsigned seed, Config config) : cfg(config) {
    std::mt19937 rng(seed);
    for (int i = 0; i < N; ++i) {
        a[i] = rng() % SOURCES;
        b[i] = rng() % SOURCES;
        enable[i] = 1;
        q0[i] = 0;
    }
    output = rng() % N;
}

Trace Model::infer(const std::vector<Input>& x) const {
    Trace trace;
    trace.pred.reserve(x.size());
    auto q = q0;

    for (std::size_t step = 0; step < x.size(); ++step) {
        if (!cfg.feedback) q = q0;
        std::array<Bit, N> next{};
        const auto source = [&](Bit id) -> Bit {
            if (id == 0) return 0;
            if (id == 1) return 1;
            if (id < 12) return x[step][id - 2];
            return q[id - 12];
        };

        for (int i = 0; i < N; ++i) {
            const Bit used_m = cfg.trained_used ? source(enable[i]) : 1;
            const Bit av = source(a[i]);
            const Bit bv = source(b[i]);
            const Bit mux = (av & (q[i] ^ 1u)) | (bv & q[i]);
            next[i] = q[i] ^ (used_m & (q[i] ^ mux));

            trace.used_m[i] |= used_m;
            trace.used_a[i] |= used_m & (q[i] ^ 1u);
            trace.used_b[i] |= used_m & q[i];
        }

        for (int i = 0; i < N; ++i) trace.transitions += next[i] != q[i];
        q = next;  // Simultaneous update after every unit read the old q.
        trace.pred.push_back(q[output] ^ invert);
    }

    trace.q = q;
    return trace;
}

Trace Model::run(const Example& example) const {
    if (example.x.size() != example.y.size()) {
        throw std::invalid_argument("Sequence input and target lengths differ");
    }
    Trace trace = infer(example.x);
    trace.change_error.reserve(example.y.size());
    for (std::size_t i = 0; i < example.y.size(); ++i) {
        trace.change_error.push_back(trace.pred[i] ^ example.y[i]);
    }
    return trace;
}

std::size_t Model::parameter_bits() const {
    return N * 10 + 3 + 1 + (cfg.trained_used ? N * 5 : 0) +
           (cfg.trained_q0 ? N : 0);
}

std::size_t Model::storage_bytes() const { return sizeof(*this); }

Accuracy accuracy(const Model& model, const std::vector<Example>& data) {
    std::size_t bits = 0;
    std::size_t correct = 0;
    std::size_t exact = 0;
    for (const auto& example : data) {
        const Trace trace = model.run(example);
        bool all_correct = true;
        for (Bit error : trace.change_error) {
            ++bits;
            correct += !error;
            all_correct &= !error;
        }
        exact += all_correct;
    }
    return {data.empty() ? 0.0 : double(exact) / data.size(),
            bits ? double(correct) / bits : 0.0};
}

}  // namespace rbfl
