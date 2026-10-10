#pragma once

#include "rbfl.hpp"

#include <array>
#include <vector>

namespace rbfl {

// Elman RNN, full BPTT, full-batch Adam, binary cross entropy.
struct RNN {
    static constexpr int H = 8;
    static constexpr int P = H * 10 + H * H + H + H + 1;
    std::array<double, P> w{}, mom{}, vel{};

    explicit RNN(unsigned seed);
    std::vector<Bit> predict(const Example& e) const;
    double fit(const std::vector<Example>& data, int epochs = 500);
};

Accuracy accuracy(const RNN& m, const std::vector<Example>& data);

}  // namespace rbfl
