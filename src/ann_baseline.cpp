#include "ann_baseline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

namespace rbfl {
namespace {

constexpr int WH = 80;
constexpr int BH = 144;
constexpr int WY = 152;
constexpr int BY = 160;

double sigmoid(double value) { return 1 / (1 + std::exp(-value)); }

using Hidden = std::array<double, RNN::H>;

struct Forward {
    std::vector<Hidden> h;
    std::vector<double> p;
};

Forward forward(const RNN& model, const Example& example) {
    Forward result;
    result.h.resize(example.x.size() + 1);
    result.p.resize(example.x.size());

    for (std::size_t t = 0; t < example.x.size(); ++t) {
        for (int j = 0; j < RNN::H; ++j) {
            double z = model.w[BH + j];
            for (int i = 0; i < 10; ++i) {
                z += model.w[j * 10 + i] * example.x[t][i];
            }
            for (int i = 0; i < RNN::H; ++i) {
                z += model.w[WH + j * RNN::H + i] * result.h[t][i];
            }
            result.h[t + 1][j] = std::tanh(z);
        }

        double z = model.w[BY];
        for (int j = 0; j < RNN::H; ++j) z += model.w[WY + j] * result.h[t + 1][j];
        result.p[t] = sigmoid(z);
    }
    return result;
}

}  // namespace

RNN::RNN(unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> distribution(0, 0.18);
    for (int i = 0; i < P; ++i) {
        if (i < BH || (i >= WY && i < BY)) w[i] = distribution(rng);
    }
}

std::vector<Bit> RNN::predict(const Example& example) const {
    const Forward result = forward(*this, example);
    std::vector<Bit> output;
    for (double probability : result.p) output.push_back(probability >= 0.5);
    return output;
}

double RNN::fit(const std::vector<Example>& data, int epochs) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    double b1p = 1;
    double b2p = 1;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::array<double, P> gradient{};
        std::size_t steps = 0;
        for (const auto& example : data) {
            const Forward result = forward(*this, example);
            Hidden dh_next{};
            steps += example.x.size();

            for (std::size_t t = example.x.size(); t-- > 0;) {
                const double dz = result.p[t] - example.y[t];
                gradient[BY] += dz;
                Hidden dh{};
                for (int j = 0; j < H; ++j) {
                    gradient[WY + j] += dz * result.h[t + 1][j];
                    dh[j] = dz * w[WY + j] + dh_next[j];
                }

                Hidden previous{};
                for (int j = 0; j < H; ++j) {
                    const double delta = dh[j] *
                        (1 - result.h[t + 1][j] * result.h[t + 1][j]);
                    gradient[BH + j] += delta;
                    for (int i = 0; i < 10; ++i) {
                        gradient[j * 10 + i] += delta * example.x[t][i];
                    }
                    for (int i = 0; i < H; ++i) {
                        gradient[WH + j * H + i] += delta * result.h[t][i];
                        previous[i] += w[WH + j * H + i] * delta;
                    }
                }
                dh_next = previous;
            }
        }

        b1p *= 0.9;
        b2p *= 0.999;
        for (int i = 0; i < P; ++i) {
            double delta = gradient[i] / steps;
            delta = std::clamp(delta, -1.0, 1.0);
            mom[i] = 0.9 * mom[i] + 0.1 * delta;
            vel[i] = 0.999 * vel[i] + 0.001 * delta * delta;
            w[i] -= 0.01 * (mom[i] / (1 - b1p)) /
                    (std::sqrt(vel[i] / (1 - b2p)) + 1e-8);
        }
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Accuracy accuracy(const RNN& model, const std::vector<Example>& data) {
    if (data.empty()) return {};
    std::size_t bits = 0;
    std::size_t correct = 0;
    std::size_t exact = 0;
    for (const auto& example : data) {
        const auto prediction = model.predict(example);
        bool all_correct = true;
        for (std::size_t i = 0; i < prediction.size(); ++i) {
            const bool right = prediction[i] == example.y[i];
            correct += right;
            all_correct &= right;
            ++bits;
        }
        exact += all_correct;
    }
    return {double(exact) / data.size(), double(correct) / bits};
}

}  // namespace rbfl
