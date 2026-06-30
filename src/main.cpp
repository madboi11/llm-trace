#include "capture.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"

#include <atomic>
#include <cctype>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ftxui;

// group key for a node: drop the " (view)"/"(permuted)" decoration and the
// trailing layer index, so ffn_gate-7 and ffn_gate-12 share the role "ffn_gate".
static std::string role_of(const std::string & name) {
    std::string r = name;
    size_t paren = r.find(" (");
    if (paren != std::string::npos) {
        r = r.substr(0, paren);
    }
    while (!r.empty() && std::isdigit((unsigned char) r.back())) {
        r.pop_back();
    }
    if (!r.empty() && (r.back() == '-' || r.back() == '_')) {
        r.pop_back();
    }
    return r;
}

struct RoleGroup {
    std::string role;
    LayerType type;
    std::vector<Capture> nodes;
};

// Accumulated tree of every node seen, grouped by role. Updated by the
// inference thread, read by the UI thread, guarded by a mutex.
struct Topology {
    mutable std::mutex mtx;
    std::vector<RoleGroup> roles;
    std::unordered_map<std::string, size_t> role_pos;
    std::unordered_map<std::string, std::pair<size_t, size_t>> node_pos;

    void update(const Capture & c) {
        std::lock_guard<std::mutex> lock(mtx);

        std::string role = role_of(c.name);
        size_t ri;
        auto it = role_pos.find(role);
        if (it == role_pos.end()) {
            ri = roles.size();
            role_pos[role] = ri;
            roles.push_back(RoleGroup{role, c.type, {}});
        } else {
            ri = it->second;
        }

        auto nit = node_pos.find(c.name);
        if (nit == node_pos.end()) {
            roles[ri].nodes.push_back(c);
            node_pos[c.name] = {ri, roles[ri].nodes.size() - 1};
        } else {
            roles[nit->second.first].nodes[nit->second.second] = c;
        }
    }

    std::vector<RoleGroup> snapshot() const {
        std::lock_guard<std::mutex> lock(mtx);
        return roles;
    }
};

struct Shared {
    TraceBuffer ring;
    Topology    topo;
    Shared(size_t cap) : ring(cap) {}
};

struct ProducerState {
    Shared & sh;
    std::chrono::steady_clock::time_point last;
    bool have_last = false;
    ProducerState(Shared & s) : sh(s) {}
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

    ps->sh.ring.push(c);
    ps->sh.topo.update(c);
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

static Color color_of(LayerType t) {
    switch (t) {
        case ATTN:  return Color::Red;
        case MLP:   return Color::Cyan;
        case NORM:  return Color::Yellow;
        case EMBED: return Color::Magenta;
        default:    return Color::GrayLight;
    }
}

// one visible line in the topology tree
struct Row {
    bool   is_role;
    size_t ri;
    size_t ni;
};

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    Shared shared(256);
    ProducerState ps(shared);

    params.cb_eval = trace_cb;
    params.cb_eval_user_data = &ps;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        return 1;
    }

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

    int focused = 0;                 // 0 = topology, 1 = metrics, 2 = stream
    int cursor = 0;                  // index into the visible rows
    std::vector<char> expanded;      // per role, indexed by role order
    std::vector<RoleGroup> roles;    // last snapshot (UI thread only)
    std::vector<Row> rows;           // last visible rows (UI thread only)

    auto rebuild_rows = [&] {
        roles = shared.topo.snapshot();
        if (expanded.size() < roles.size()) {
            expanded.resize(roles.size(), 0);
        }
        rows.clear();
        for (size_t ri = 0; ri < roles.size(); ri++) {
            rows.push_back(Row{true, ri, 0});
            if (expanded[ri]) {
                for (size_t ni = 0; ni < roles[ri].nodes.size(); ni++) {
                    rows.push_back(Row{false, ri, ni});
                }
            }
        }
        if (cursor >= (int) rows.size()) {
            cursor = rows.empty() ? 0 : (int) rows.size() - 1;
        }
        if (cursor < 0) {
            cursor = 0;
        }
    };

    auto panel = [&](const std::string & title, Element body, int idx) {
        auto e = window(text(" " + title + " "), body);
        if (idx == focused) {
            e = e | color(Color::GreenLight) | bold;
        }
        return e;
    };

    auto topology_panel = [&] {
        Elements lines;
        for (size_t i = 0; i < rows.size(); i++) {
            const Row & row = rows[i];
            const RoleGroup & g = roles[row.ri];
            Element e;
            if (row.is_role) {
                std::string mark = expanded[row.ri] ? "▾ " : "▸ ";
                e = hbox({
                    text(mark),
                    text(g.role) | color(color_of(g.type)),
                    text("  (" + std::to_string(g.nodes.size()) + ")") | dim,
                });
            } else {
                e = text("    " + g.nodes[row.ni].name) | color(Color::GrayLight);
            }
            if ((int) i == cursor) {
                e = e | inverted | focus;
            }
            lines.push_back(e);
        }
        if (lines.empty()) {
            lines.push_back(text("waiting for nodes...") | dim);
        }
        return vbox(std::move(lines)) | frame | flex;
    };

    auto metrics_panel = [&] {
        if (rows.empty()) {
            return vbox({text("(no selection)") | dim});
        }
        const Row & row = rows[cursor];
        const RoleGroup & g = roles[row.ri];
        if (row.is_role) {
            double sum = 0.0;
            for (const auto & n : g.nodes) {
                sum += n.ms;
            }
            double avg = g.nodes.empty() ? 0.0 : sum / g.nodes.size();
            return vbox({
                text("role:      " + g.role),
                text(std::string("type:      ") + type_name(g.type)),
                text("instances: " + std::to_string(g.nodes.size())),
                text("avg lat:   " + std::to_string(avg) + " ms"),
            });
        }
        const Capture & c = g.nodes[row.ni];
        std::string shape = "[" + std::to_string(c.ne[0]) + ", " + std::to_string(c.ne[1]) +
                            ", " + std::to_string(c.ne[2]) + ", " + std::to_string(c.ne[3]) + "]";
        return vbox({
            text("node:    " + c.name),
            text(std::string("type:    ") + type_name(c.type)),
            text("op:      " + c.op),
            text("dtype:   " + c.dtype),
            text("shape:   " + shape),
            text("latency: " + std::to_string(c.ms) + " ms"),
        });
    };

    auto stream_panel = [&] {
        size_t total = shared.ring.total_count();
        std::vector<Capture> recent = shared.ring.snapshot();
        Elements lines;
        lines.push_back(text("captured nodes: " + std::to_string(total)));
        lines.push_back(separator());
        size_t show = recent.size() < 8 ? recent.size() : 8;
        for (size_t i = recent.size() - show; i < recent.size(); i++) {
            lines.push_back(text(recent[i].name) | dim);
        }
        return vbox(std::move(lines));
    };

    auto renderer = Renderer([&] {
        rebuild_rows();
        return vbox({
            panel("Model Topology", topology_panel(), 0) | flex,
            hbox({
                panel("Runtime Metrics", metrics_panel(), 1) | flex,
                panel("Live Stream", stream_panel(), 2) | flex,
            }) | flex,
            text(" Tab: panel   j/k: move   space: expand   q: quit ") | inverted,
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
        if (focused == 0) {
            if (e == Event::Character('j') || e == Event::ArrowDown) {
                if (cursor + 1 < (int) rows.size()) {
                    cursor++;
                }
                return true;
            }
            if (e == Event::Character('k') || e == Event::ArrowUp) {
                if (cursor > 0) {
                    cursor--;
                }
                return true;
            }
            if (e == Event::Character(' ') || e == Event::Return) {
                if (!rows.empty() && rows[cursor].is_role) {
                    size_t ri = rows[cursor].ri;
                    expanded[ri] = !expanded[ri];
                }
                return true;
            }
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
