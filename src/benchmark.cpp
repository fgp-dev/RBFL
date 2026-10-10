#include "ann_baseline.hpp"
#include "rbfl.hpp"
#include "bfl/routed_binary.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

namespace {

using namespace rbfl;
using Clock = std::chrono::steady_clock;

volatile unsigned checksum = 0;
double measured_steps = 1;
int max_steps = 1;

struct Data {
    std::vector<Example> train, val, test, shift;
    std::mt19937 after_split;
};

Bit static_target(const Input& x) {
    const int score = -1 + x[0] + x[1] + (x[0] & x[1]) +
        (x[2] & x[3]) + (x[4] & x[5]) - x[6] - x[7] - (x[8] & x[9]);
    return score >= 1;
}

Data static_data(unsigned seed, bool validation) {
    std::vector<unsigned> domain(1024);
    std::iota(domain.begin(), domain.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(domain.begin(), domain.end(), rng);

    Data data;
    for (unsigned j = 0; j < 1024; ++j) {
        Input x{};
        for (int k = 0; k < 10; ++k) x[k] = (domain[j] >> k) & 1;
        Example example{{x}, {static_target(x)}};
        if (j < 600) data.train.push_back(example);
        else if (validation && j < 800) data.val.push_back(example);
        else data.test.push_back(example);
    }
    data.after_split = rng;
    return data;
}

Data temporal_data(unsigned seed) {
    Data data;
    std::mt19937 rng(seed * 7919 + 17);
    const auto make = [&](std::vector<Example>& output, int count, int lo,
                          int hi, bool shifted) {
        for (int j = 0; j < count; ++j) {
            Example example;
            const int length = lo + rng() % (hi - lo + 1);
            Bit parity = 0;
            for (int t = 0; t < length; ++t) {
                Input x{};
                for (auto& bit : x) bit = rng() & 1;
                if (shifted) x[0] = (rng() % 10) < 2;
                parity ^= x[0];
                example.x.push_back(x);
                example.y.push_back(parity);
            }
            output.push_back(std::move(example));
        }
    };

    make(data.train, 160, 4, 8, false);
    make(data.val, 40, 4, 8, false);
    // The final tests use a separate stream from training and validation.
    rng.seed(seed * 7919 + 1000003);
    make(data.test, 80, 9, 12, false);
    make(data.shift, 80, 9, 12, true);
    return data;
}

// Equation and training schedule of the archived 16-unit ANN benchmark.
struct MLP {
    static constexpr int H = 16;
    static constexpr int P = 193;
    std::array<double, P> w{}, m{}, v{};

    explicit MLP(unsigned seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> distribution(0, std::sqrt(2.0 / 26));
        for (int i = 0; i < 160; ++i) w[i] = distribution(rng);
        for (int i = 176; i < 192; ++i) w[i] = distribution(rng);
    }

    double logit(const Input& x, std::array<double, H>& hidden) const {
        double z = w[192];
        for (int j = 0; j < H; ++j) {
            double activation = w[160 + j];
            for (int i = 0; i < 10; ++i) activation += w[j * 10 + i] * x[i];
            hidden[j] = std::tanh(activation);
            z += w[176 + j] * hidden[j];
        }
        return z;
    }

    Bit predict(const Input& x) const {
        std::array<double, H> hidden{};
        return logit(x, hidden) >= 0;
    }

    double fit(const std::vector<Example>& data, int epochs = 500) {
        const auto start = Clock::now();
        double p1 = 1;
        double p2 = 1;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::array<double, P> gradient{};
            int count = 0;
            for (const auto& example : data) {
                for (std::size_t t = 0; t < example.x.size(); ++t) {
                    ++count;
                    std::array<double, H> hidden{};
                    const double z = logit(example.x[t], hidden);
                    const double delta = 1 / (1 + std::exp(-z)) - example.y[t];
                    gradient[192] += delta;
                    for (int j = 0; j < H; ++j) {
                        gradient[176 + j] += delta * hidden[j];
                        const double back = delta * w[176 + j] *
                            (1 - hidden[j] * hidden[j]);
                        gradient[160 + j] += back;
                        for (int i = 0; i < 10; ++i) {
                            gradient[j * 10 + i] += back * example.x[t][i];
                        }
                    }
                }
            }

            p1 *= 0.9;
            p2 *= 0.999;
            for (int i = 0; i < P; ++i) {
                const double value = gradient[i] / count;
                m[i] = 0.9 * m[i] + 0.1 * value;
                v[i] = 0.999 * v[i] + 0.001 * value * value;
                w[i] -= 0.03 * (m[i] / (1 - p1)) /
                    (std::sqrt(v[i] / (1 - p2)) + 1e-8);
            }
        }
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
};

template <class Predict>
Accuracy evaluate(const std::vector<Example>& data, Predict predict) {
    int bits = 0;
    int correct = 0;
    int exact = 0;
    for (const auto& example : data) {
        bool all_correct = true;
        for (std::size_t t = 0; t < example.x.size(); ++t) {
            const bool right = predict(example.x[t]) == example.y[t];
            all_correct &= right;
            correct += right;
            ++bits;
        }
        exact += all_correct;
    }
    return {data.empty() ? 0.0 : double(exact) / data.size(),
            bits ? double(correct) / bits : 0.0};
}

template <class Predict>
double timing(const std::vector<Example>& data, Predict predict) {
    for (int i = 0; i < 10000; ++i) checksum += predict(data[i % data.size()]);
    std::array<double, 5> nanoseconds{};
    for (int repetition = 0; repetition < 5; ++repetition) {
        const auto start = Clock::now();
        for (int i = 0; i < 100000; ++i) {
            checksum += predict(data[i % data.size()]);
        }
        nanoseconds[repetition] =
            std::chrono::duration<double, std::nano>(Clock::now() - start).count() /
            100000;
    }
    return std::accumulate(nanoseconds.begin(), nanoseconds.end(), 0.0) / 5;
}

void row(const std::string& task, const std::string& model, unsigned seed,
         Accuracy train, Accuracy validation, Accuracy test, Accuracy shifted,
         double training_ms, double ns, int parameter_bits, int model_bytes,
         int temporary_bytes, int accepted, int candidates, int inactive,
         int q_changes) {
    const auto accuracy_columns = [](Accuracy value, bool present) {
        if (present) std::cout << value.exact << ',' << value.bit;
        else std::cout << "NA,NA";
    };
    std::cout << task << ',' << model << ',' << seed << ',';
    accuracy_columns(train, true);
    std::cout << ',';
    accuracy_columns(validation, task != "historical");
    std::cout << ',';
    accuracy_columns(test, true);
    std::cout << ',';
    accuracy_columns(shifted, task == "parity");
    std::cout << ',' << (train.bit - test.bit) << ',' << training_ms << ',' << ns
              << ',' << (ns / measured_steps) << ',' << measured_steps << ','
              << parameter_bits << ',' << model_bytes << ',' << temporary_bytes
              << ',' << accepted << ',' << candidates << ',' << inactive << ','
              << q_changes << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? std::stoi(argv[1]) : 3;
    std::ofstream trace;
    if (argc > 2) {
        trace.open(argv[2]);
        trace << "task,model,seed,epoch,err_before,err_best,accepted_total\n";
    }
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "task,model,seed,train_exact,train_bit,val_exact,val_bit,"
                 "test_exact,test_bit,shift_exact,shift_bit,gap_bit,training_ms,"
                 "ns_per_sequence,ns_per_step,mean_steps,parameter_bits,"
                 "model_bytes,temporary_inference_bytes,accepted,candidates,"
                 "inactive_rewires,q_changes\n";

    for (int seed = 1; seed <= seeds; ++seed) {
        for (int task = 0; task < 3; ++task) {
            const bool temporal = task == 2;
            const Data data = temporal ? temporal_data(seed) :
                static_data(seed, task == 1);
            const std::string label = task == 0 ? "historical" :
                task == 1 ? "generalization" : "parity";

            measured_steps = 0;
            max_steps = 1;
            for (const auto& example : data.test) {
                measured_steps += example.x.size();
                max_steps = std::max(max_steps, int(example.x.size()));
            }
            measured_steps /= data.test.size();

            if (!temporal) {
                std::mt19937 rng = data.after_split;
                std::uint64_t initial = 0;
                for (int i = 0; i < 16; ++i) {
                    if (rng() & 1) initial |= 1ull << i;
                }
                bfl::RoutedBinaryModel bfl_model(initial);
                std::vector<bfl::RoutedBinarySample> samples;
                for (const auto& example : data.train) {
                    samples.push_back({example.x[0], example.y[0]});
                }
                const auto start = Clock::now();
                const auto report = bfl_model.train(samples, rng, 20);
                const double milliseconds =
                    std::chrono::duration<double, std::milli>(Clock::now() - start)
                        .count();
                const auto predict = [&](const Input& x) {
                    return bfl_model.predict(x);
                };
                row(label, "bfl_binary_global", seed,
                    evaluate(data.train, predict), evaluate(data.val, predict),
                    evaluate(data.test, predict), {}, milliseconds,
                    timing(data.test, [&](const Example& e) {
                        return predict(e.x[0]);
                    }), 2048, 256, 0, report.accepted_updates,
                    report.candidate_evaluations, 0, 0);

                MLP mlp(seed * 101 + 42);
                const double mlp_ms = mlp.fit(data.train);
                const auto mlp_predict = [&](const Input& x) {
                    return mlp.predict(x);
                };
                row(label, "ann_16", seed,
                    evaluate(data.train, mlp_predict),
                    evaluate(data.val, mlp_predict),
                    evaluate(data.test, mlp_predict), {}, mlp_ms,
                    timing(data.test, [&](const Example& e) {
                        return mlp_predict(e.x[0]);
                    }), 193 * 64, 1544, 128, 0, 0, 0, 0);
            } else {
                MLP mlp(seed * 101 + 42);
                const double mlp_ms = mlp.fit(data.train);
                const auto predict = [&](const Input& x) { return mlp.predict(x); };
                row(label, "ann_16", seed, evaluate(data.train, predict),
                    evaluate(data.val, predict), evaluate(data.test, predict),
                    evaluate(data.shift, predict), mlp_ms,
                    timing(data.test, [&](const Example& e) {
                        unsigned sum = 0;
                        for (const auto& x : e.x) sum += predict(x);
                        return sum;
                    }), 193 * 64, 1544, 128, 0, 0, 0, 0);
            }

            RNN rnn(seed * 101 + 42);
            const double rnn_ms = rnn.fit(data.train);
            const auto rnn_predict = [&](const Example& e) {
                return rnn.predict(e)[0];
            };
            row(label, "elman_rnn_8", seed, accuracy(rnn, data.train),
                accuracy(rnn, data.val), accuracy(rnn, data.test),
                accuracy(rnn, data.shift), rnn_ms, timing(data.test, rnn_predict),
                RNN::P * 64, RNN::P * 8,
                (max_steps + 1) * RNN::H * 8 + max_steps * 9, 0, 0, 0, 0);

            for (int mode = 0; mode < (temporal ? 7 : 1); ++mode) {
                Config config;
                config.trained_used = mode == 1 || mode == 6;
                config.forget = mode != 2;
                config.feedback = mode != 3 && mode != 6;
                config.trained_q0 = mode == 5;

                Model model(seed * 101 + 42, config);
                std::mt19937 rng(seed * 103 + 1234);
                std::vector<std::array<double, 4>> log;
                const auto report = train(model, data.train, rng, &log);
                const char* name = mode == 0 ? "rbfl_used_m_fixed1_feedback" :
                    mode == 1 ? "rbfl_used_m_trainable_feedback" :
                    mode == 2 ? "rbfl_used_m_fixed1_no_forget" :
                    mode == 3 ? "rbfl_used_m_fixed1_no_feedback" :
                    mode == 4 ? "rbfl_used_m_fixed1_q0_zero" :
                    mode == 5 ? "rbfl_used_m_fixed1_q0_trainable" :
                                "rbfl_used_m_trainable_no_feedback";

                for (const auto& entry : log) {
                    if (trace) {
                        trace << label << ',' << name << ',' << seed << ','
                              << entry[0] << ',' << entry[1] << ',' << entry[2]
                              << ',' << entry[3] << '\n';
                    }
                }
                const auto predict = [&](const Example& e) {
                    return model.infer(e.x).pred[0];
                };
                row(label, name, seed, accuracy(model, data.train),
                    accuracy(model, data.val), accuracy(model, data.test),
                    accuracy(model, data.shift), report.ms,
                    timing(data.test, predict), model.parameter_bits(),
                    model.storage_bytes(), sizeof(Trace) + 2 * N + max_steps,
                    report.accepted, report.candidates,
                    report.inactive_rewires, report.q_changes);
            }
        }
    }
    return checksum == UINT32_MAX;
}
