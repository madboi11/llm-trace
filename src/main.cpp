#include "capture.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace ftxui;

struct ProducerState {
    TraceBuffer & buf;
    std::chrono::steady_clock::time_point last;
    bool have_last = false;

    ProducerState(TraceBuffer & b) : buf(b) {}
};

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true;
    }

    ProducerState * ps = (ProducerState *) user_data;

    auto now = std::chrono::steady_clock::now();
    double ms = 0.0;
    if (ps->have_last) {
        ms = std::chrono::duration<double, std::milli>(now - ps->last).count();
    }
    ps->last = now;
    ps->have_last = true;

    Capture c;
    c.name  = t->name;
    c.type  = classify(t->name);
    c.op    = ggml_op_desc(t);
    c.dtype = ggml_type_name(t->type);
    c.ne[0] = t->ne[0];
    c.ne[1] = t->ne[1];
    c.ne[2] = t->ne[2];
    c.ne[3] = t->ne[3];
    c.ms    = ms;

    ps->buf.push(c);
    return true;
}

static llama_token greedy(llama_context * ctx, const llama_vocab * vocab) {
    float * logits = llama_get_logits_ith(ctx, -1);
    int n = llama_vocab_n_tokens(vocab);
    llama_token best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

// Runs on its own thread. Keeps generating so the callback keeps firing,
// until the UI sets the stop flag.
static void inference_loop(llama_context * ctx, const llama_model * model,
                           ProducerState & ps, std::string prompt,
                           std::atomic<bool> & stop) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    while (!stop.load()) {
        llama_memory_clear(llama_get_memory(ctx), true);

        std::vector<llama_token> tokens = common_tokenize(ctx, prompt, add_bos, true);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
            break;
        }

        for (int i = 0; i < 128 && !stop.load(); i++) {
            llama_token tok = greedy(ctx, vocab);
            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }
            if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
                break;
            }
        }
    }
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    TraceBuffer buffer(256);
    ProducerState ps(buffer);

    params.cb_eval = trace_cb;
    params.cb_eval_user_data = &ps;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        return 1;
    }

    // silence llama/ggml logs so they don't corrupt the terminal UI
    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    std::atomic<bool> stop{false};
    std::thread producer(inference_loop, ctx, model, std::ref(ps), params.prompt, std::ref(stop));

    auto screen = ScreenInteractive::Fullscreen();

    std::atomic<bool> ui_running{true};
    std::thread refresher([&] {
        while (ui_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            screen.Post(Event::Custom);
        }
    });

    int focused = 0;

    auto panel = [&](const std::string & title, Element body, int idx) {
        auto e = window(text(" " + title + " "), body);
        if (idx == focused) {
            e = e | color(Color::GreenLight) | bold;
        }
        return e;
    };

    auto renderer = Renderer([&] {
        size_t total = buffer.total_count();

        auto topology = panel("Model Topology", text("(empty)") | dim, 0);
        auto metrics  = panel("Runtime Metrics", text("(empty)") | dim, 1);
        auto stream   = panel("Live Stream",
            vbox({
                text("captured nodes: " + std::to_string(total)),
                text("(panel content comes in M3)") | dim,
            }), 2);

        return vbox({
            topology | flex,
            hbox({ metrics | flex, stream | flex }) | flex,
            text(" Tab: switch panel    q: quit ") | inverted,
        });
    });

    auto component = CatchEvent(renderer, [&](Event e) {
        if (e == Event::Character('q') || e == Event::Escape) {
            screen.Exit();
            return true;
        }
        if (e == Event::Tab) {
            focused = (focused + 1) % 3;
            return true;
        }
        return false;
    });

    screen.Loop(component);

    ui_running.store(false);
    refresher.join();
    stop.store(true);
    producer.join();

    llama_backend_free();
    return 0;
}
