#include "rbfl.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>

namespace rbfl {
namespace {

int bit_count(unsigned value) { return __builtin_popcount(value); }

std::vector<Bit> errors(const Model& model, const std::vector<Example>& data) {
    std::vector<Bit> result;
    for (const auto& example : data) {
        const Trace trace = model.run(example);
        result.insert(result.end(), trace.change_error.begin(),
                      trace.change_error.end());
    }
    return result;
}

int count_errors(const std::vector<Bit>& values) {
    return std::accumulate(values.begin(), values.end(), 0);
}

Score score(const std::vector<Bit>& before, const std::vector<Bit>& after,
            int change_cost) {
    Score result{0, 0, change_cost};
    for (std::size_t i = 0; i < before.size(); ++i) {
        result.err += after[i];
        result.forget += !before[i] && after[i];
    }
    return result;
}

struct Candidate {
    int kind;
    int unit;
    int value;
};

}  // namespace

TrainReport train(Model& model, const std::vector<Example>& data,
                  std::mt19937& rng, std::vector<std::array<double, 4>>* log) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    TrainReport report;
    report.initial_err = count_errors(errors(model, data));

    // Propose all source alternatives. Dormant routes remain searchable.
    std::vector<Candidate> options;
    for (int i = 0; i < N; ++i) {
        for (int value = 0; value < SOURCES; ++value) {
            options.push_back({0, i, value});  // A source.
            options.push_back({1, i, value});  // B source.
            if (model.cfg.trained_used) options.push_back({2, i, value});
        }
        if (model.cfg.trained_q0) options.push_back({3, i, 1});
    }
    for (int value = 0; value < N; ++value) options.push_back({4, 0, value});
    options.push_back({5, 0, 1});  // Output inversion.

    for (int epoch = 0; epoch < model.cfg.epochs; ++epoch) {
        const auto before = errors(model, data);
        const Score baseline{count_errors(before), 0, 0};
        std::array<Bit, N> used_a{};
        std::array<Bit, N> used_b{};
        for (const auto& example : data) {
            const Trace trace = model.run(example);
            for (int i = 0; i < N; ++i) {
                used_a[i] |= trace.used_a[i];
                used_b[i] |= trace.used_b[i];
            }
        }

        std::shuffle(options.begin(), options.end(), rng);
        Score best = baseline;
        Model winner = model;
        int best_inactive = 0;

        for (const Candidate candidate : options) {
            Model proposal = model;
            int cost = 0;
            int inactive = 0;
            if (candidate.kind <= 2) {
                Bit* source = candidate.kind == 0 ? &proposal.a[candidate.unit]
                    : candidate.kind == 1 ? &proposal.b[candidate.unit]
                                          : &proposal.enable[candidate.unit];
                if (*source == candidate.value) continue;
                if (candidate.kind < 2 &&
                    !(candidate.kind == 0 ? used_a[candidate.unit]
                                           : used_b[candidate.unit])) {
                    inactive = 1;
                }
                cost = bit_count(*source ^ candidate.value);
                *source = candidate.value;
            } else if (candidate.kind == 3) {
                proposal.q0[candidate.unit] ^= 1;
                cost = 1;
            } else if (candidate.kind == 4) {
                if (proposal.output == candidate.value) continue;
                cost = bit_count(proposal.output ^ candidate.value);
                proposal.output = candidate.value;
            } else {
                proposal.invert ^= 1;
                cost = 1;
            }

            // Every candidate starts at its own H0 and replays all train data.
            const auto after = errors(proposal, data);
            ++report.candidates;
            const Score candidate_score = score(before, after, cost);
            if (less(candidate_score, best, model.cfg.forget)) {
                best = candidate_score;
                winner = proposal;
                best_inactive = inactive;
            }
        }

        const bool accepted = less(best, baseline, model.cfg.forget);
        if (accepted) {
            model = winner;
            ++report.accepted;
            report.inactive_rewires += best_inactive;
        }
        if (log) {
            log->push_back({double(epoch), double(baseline.err),
                            double(best.err), double(report.accepted)});
        }
        if (best.err == 0 || !accepted) break;
    }

    report.final_err = count_errors(errors(model, data));
    for (const auto& example : data) {
        report.q_changes += model.infer(example.x).transitions;
    }
    report.ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return report;
}

}  // namespace rbfl
