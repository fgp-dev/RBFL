#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace rbfl {

using Bit = std::uint8_t;
using Input = std::array<Bit, 10>;

struct Example {
    std::vector<Input> x;
    std::vector<Bit> y;
};

constexpr int N = 8;
constexpr int SOURCES = 2 + 10 + N;  // 0, 1, ten inputs, eight old-state bits.

struct Config {
    bool trained_used = false;
    bool feedback = true;
    bool trained_q0 = false;
    bool forget = true;
    int epochs = 4;
};

struct Score {
    int err = 0;
    int forget = 0;
    int cost = 0;
};

bool less(Score a, Score b, bool use_forget = true);

struct Trace {
    std::vector<Bit> pred;
    std::vector<Bit> change_error;
    std::array<Bit, N> q{};
    std::array<Bit, N> used_m{};
    std::array<Bit, N> used_a{};
    std::array<Bit, N> used_b{};
    int transitions = 0;
};

struct Model {
    std::array<Bit, N> a{};
    std::array<Bit, N> b{};
    std::array<Bit, N> enable{};
    std::array<Bit, N> q0{};
    Bit output = 0;
    Bit invert = 0;
    Config cfg;

    explicit Model(unsigned seed, Config config = {});

    // Inference never receives a target. All MUXes read the same old state.
    Trace infer(const std::vector<Input>& x) const;
    Trace run(const Example& example) const;
    std::size_t parameter_bits() const;
    std::size_t storage_bytes() const;
};

struct TrainReport {
    int initial_err = 0;
    int final_err = 0;
    int accepted = 0;
    int candidates = 0;
    int inactive_rewires = 0;
    int q_changes = 0;
    double ms = 0;
};

TrainReport train(Model& model, const std::vector<Example>& data, std::mt19937& rng,
                  std::vector<std::array<double, 4>>* log = nullptr);

struct Accuracy {
    double exact = 0;
    double bit = 0;
};

Accuracy accuracy(const Model& model, const std::vector<Example>& data);

}  // namespace rbfl
