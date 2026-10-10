#include "rbfl.hpp"

#include <cassert>
#include <random>

using namespace rbfl;

int main() {
    assert(less({1, 100, 100}, {2, 0, 0}));
    assert(less({1, 0, 100}, {1, 1, 0}));
    assert(!less({1, 1, 0}, {1, 0, 100}));

    Example example{{Input{}}, {0}};
    Model model(1);
    model.output = 0;
    model.q0.fill(0);
    model.a.fill(0);
    model.b.fill(0);

    // With Q=0, the truth table returns A; with Q=1, it returns B.
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            model.a[0] = a;
            model.b[0] = b;
            assert(model.run(example).q[0] == a);
        }
    }
    model.q0[0] = 1;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            model.a[0] = a;
            model.b[0] = b;
            assert(model.run(example).q[0] == b);
        }
    }

    model.q0[0] = 0;
    model.a[0] = 1;
    model.b[0] = 1;
    example.x.push_back(Input{});
    example.y.push_back(1);
    const Trace trace = model.run(example);
    assert(trace.pred[0] == 1 && trace.pred[1] == 1);
    assert(trace.change_error[0] == 1 && trace.change_error[1] == 0);
    assert(model.infer(example.x).change_error.empty());
    assert(model.infer(example.x).transitions > 0);
    assert(trace.used_a[0] && trace.used_b[0] && trace.used_m[0]);

    model.cfg.trained_used = true;
    model.enable[0] = 0;
    const Trace held = model.run(example);
    assert(held.q[0] == 0 && !held.used_a[0] && !held.used_b[0]);

    // A unit must read the other unit's old Q, even if that unit updates first.
    Model simultaneous(2);
    simultaneous.a.fill(0);
    simultaneous.b.fill(0);
    simultaneous.q0.fill(0);
    simultaneous.q0[0] = 1;
    simultaneous.a[1] = 12;  // Old Q of unit 0.
    simultaneous.output = 1;
    const Trace simultaneous_trace = simultaneous.infer({Input{}});
    assert(simultaneous_trace.q[0] == 0);
    assert(simultaneous_trace.q[1] == 1);

    // Feedback carries one bit across time; the ablation resets it each step.
    Model toggling(3);
    toggling.a.fill(0);
    toggling.b.fill(0);
    toggling.q0.fill(0);
    toggling.a[0] = 1;
    toggling.output = 0;
    const std::vector<Input> two_steps(2);
    assert((toggling.infer(two_steps).pred == std::vector<Bit>{1, 0}));
    toggling.cfg.feedback = false;
    assert((toggling.infer(two_steps).pred == std::vector<Bit>{1, 1}));

    model.cfg.trained_used = false;
    std::mt19937 rng(4);
    const TrainReport report = train(model, {example}, rng);
    assert(report.final_err <= report.initial_err && report.candidates > 0);

    Model easy(9);
    easy.a.fill(0);
    easy.b.fill(0);
    easy.q0.fill(0);
    easy.output = 0;
    easy.invert = 0;
    const Example positive{{Input{}}, {1}};
    std::mt19937 easy_rng(5);
    const TrainReport update = train(easy, {positive}, easy_rng);
    assert(update.initial_err == 1 && update.final_err == 0 && update.accepted > 0);
    const TrainReport stable = train(easy, {positive}, easy_rng);
    assert(stable.initial_err == 0 && stable.final_err == 0 && stable.accepted == 0);
}
