#include "capture.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ftxui;

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

struct Topology {
    mutable std::mutex mtx;
    std::vector<RoleGroup> roles;
    std::unordered_map<std::string, size_t> role_pos;
    std::unordered_map<std::string, std::pair<size_t, size_t>> node_pos;

    void update(Capture c) {
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
            Capture & dst = roles[nit->second.first].nodes[nit->second.second];
            // keep the last good stats when this fire didn't compute them
            if (!c.stats_ok && dst.stats_ok) {
                c.stats_ok = true;
                c.mean = dst.mean;
                c.vmin = dst.vmin;
                c.vmax = dst.vmax;
                c.vstd = dst.vstd;
                c.sparsity = dst.sparsity;
                c.n_bad = dst.n_bad;
            }
            dst = c;
        }
    }

    std::vector<RoleGroup> snapshot() const {
        std::lock_guard<std::mutex> lock(mtx);
        return roles;
    }
};

// attention weights of one prompt decode for the selected layer (all heads).
// value(key j, query i, head h) = data[j + i*n_kv + h*n_kv*n_query]
struct LayerAttn {
    bool valid = false;
    int n_query = 0;
    int n_head = 0;
    int n_kv = 0;
    double ms = 0;
    std::vector<float> data;
};

struct Attention {
    mutable std::mutex mtx;
    std::vector<LayerAttn> layers;       // one per model layer
    std::vector<std::string> labels;
};

struct Shared {
    TraceBuffer ring;
    Topology    topo;
    Attention   attn;
    Shared(size_t cap) : ring(cap) {}
};

struct ProducerState {
    Shared & sh;
    std::chrono::steady_clock::time_point last;
    std::chrono::steady_clock::time_point start;
    bool have_last = false;
    size_t seq = 0;
    ProducerState(Shared & s) : sh(s) {}
};

static void fill_stats(Capture & c, const ggml_tensor * t);

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true;
    }

    ProducerState * ps = (ProducerState *) user_data;

    auto now = std::chrono::steady_clock::now();
    if (!ps->have_last) {
        ps->start = now;
    }
    double ms = 0.0;
    if (ps->have_last) {
        ms = std::chrono::duration<double, std::milli>(now - ps->last).count();
    }
    ps->last = now;
    ps->have_last = true;

    Capture c;
    c.id     = ps->seq++;
    c.t_ms   = std::chrono::duration<double, std::milli>(now - ps->start).count();
    c.name   = t->name;
    c.type   = classify(t->name);
    c.op     = ggml_op_desc(t);
    c.dtype  = ggml_type_name(t->type);
    c.device = t->buffer ? ggml_backend_buffer_name(t->buffer) : "?";
    c.ne[0]  = t->ne[0];
    c.ne[1]  = t->ne[1];
    c.ne[2]  = t->ne[2];
    c.ne[3]  = t->ne[3];
    c.ms     = ms;
    if (t->ne[1] > 1) {            // prompt decode only — keeps generation fast
        fill_stats(c, t);
    }

    ps->sh.ring.push(c);
    ps->sh.topo.update(c);

    // capture the score matrix for every layer on the prompt decode,
    // so the UI can switch layers instantly
    if (c.name.rfind("kq_soft_max-", 0) == 0 && t->ne[1] > 1) {
        int layer = std::atoi(c.name.c_str() + 12);
        std::vector<float> tmp(ggml_nelements(t));
        if (ggml_backend_buffer_is_host(t->buffer)) {
            std::memcpy(tmp.data(), t->data, ggml_nbytes(t));
        } else {
            ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
        }
        std::lock_guard<std::mutex> lock(ps->sh.attn.mtx);
        if (layer >= 0 && layer < (int) ps->sh.attn.layers.size()) {
            LayerAttn & la = ps->sh.attn.layers[layer];
            la.data    = std::move(tmp);
            la.n_kv    = t->ne[0];
            la.n_query = t->ne[1];
            la.n_head  = t->ne[2];
            la.ms      = ms;
            la.valid   = true;
        }
    }
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

        for (int i = 0; i < 32 && !stop.load(); i++) {
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

static std::string short_label(std::string s) {
    for (char & ch : s) {
        if ((unsigned char) ch < 32) {
            ch = ' ';
        }
    }
    if (s.size() > 7) {
        s = s.substr(0, 7);
    }
    while (s.size() < 7) {
        s += ' ';
    }
    return s;
}

// numeric stats over a node's output (contiguous f32/f16 only)
static void fill_stats(Capture & c, const ggml_tensor * t) {
    c.stats_ok = false;
    if (!ggml_is_contiguous(t)) {
        return;
    }
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
        return;
    }
    int64_t n = ggml_nelements(t);
    if (n <= 0) {
        return;
    }

    const void * raw;
    std::vector<char> tmp;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        raw = t->data;
    } else {
        tmp.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
        raw = tmp.data();
    }

    double sum = 0, sumsq = 0, mn = 0, mx = 0;
    int64_t zeros = 0, good = 0;
    int bad = 0;
    bool first = true;
    for (int64_t i = 0; i < n; i++) {
        float v = (t->type == GGML_TYPE_F32)
                      ? ((const float *) raw)[i]
                      : ggml_fp16_to_fp32(((const ggml_fp16_t *) raw)[i]);
        if (!std::isfinite(v)) {
            bad++;
            continue;
        }
        sum += v;
        sumsq += (double) v * v;
        if (first || v < mn) mn = v;
        if (first || v > mx) mx = v;
        first = false;
        if (std::fabs(v) < 1e-6f) zeros++;
        good++;
    }

    c.stats_ok  = true;
    c.mean      = good ? sum / good : 0;
    double var  = good ? sumsq / good - c.mean * c.mean : 0;
    c.vstd      = var > 0 ? std::sqrt(var) : 0;
    c.vmin      = mn;
    c.vmax      = mx;
    c.sparsity  = (double) zeros / n;
    c.n_bad     = bad;
}

struct Row {
    bool   is_role;
    size_t ri;
    size_t ni;
};

// snapshot of the attention struct for one render frame
struct AttnView {
    bool valid = false;
    int layer = 0, n_query = 0, n_head = 0, n_kv = 0;
    double ms = 0;
    std::vector<float> data;
    std::vector<std::string> labels;

    float at(int i, int j, int h) const {
        return data[j + i * n_kv + h * n_kv * n_query];
    }
};

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.prompt.empty()) {
        params.prompt = "The cat sat on the mat";
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    Shared shared(256);
    ProducerState ps(shared);

    params.cb_eval = trace_cb;
    params.cb_eval_user_data = &ps;
    params.warmup = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // exposes kq_soft_max

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        return 1;
    }

    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    const int n_layers = llama_model_n_layer(model);
    shared.attn.layers.resize(n_layers);

    {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        bool add_bos = llama_vocab_get_add_bos(vocab);
        std::vector<llama_token> ptoks = common_tokenize(ctx, params.prompt, add_bos, true);
        for (llama_token tk : ptoks) {
            shared.attn.labels.push_back(common_token_to_piece(ctx, tk, true));
        }
    }

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

    int focused = 0;                 // 0 topology, 1 attention, 2 metrics, 3 stream, 4 ledger
    int active_selector = 0;         // which panel drives the metrics: 0 topology, 1 attention
    int cursor = 0;
    int sel_head = 0;
    int sel_layer = 0;
    int led_cursor = 0;
    bool attn_full = false;
    std::vector<char> expanded;
    std::vector<RoleGroup> roles;
    std::vector<Row> rows;
    std::vector<Capture> flat;       // every node, flattened, for the anomaly ledger
    AttnView av;

    auto rebuild = [&] {
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

        flat.clear();
        for (const auto & g : roles) {
            for (const auto & c : g.nodes) {
                flat.push_back(c);
            }
        }
        if (led_cursor >= (int) flat.size()) {
            led_cursor = flat.empty() ? 0 : (int) flat.size() - 1;
        }
        if (led_cursor < 0) {
            led_cursor = 0;
        }

        std::lock_guard<std::mutex> lock(shared.attn.mtx);
        av.labels = shared.attn.labels;
        av.layer  = sel_layer;
        if (sel_layer >= 0 && sel_layer < (int) shared.attn.layers.size()) {
            const LayerAttn & la = shared.attn.layers[sel_layer];
            av.valid   = la.valid;
            av.n_query = la.n_query;
            av.n_head  = la.n_head;
            av.n_kv    = la.n_kv;
            av.ms      = la.ms;
            av.data    = la.data;
        } else {
            av.valid = false;
        }
        if (av.n_head > 0 && sel_head >= av.n_head) {
            sel_head = av.n_head - 1;
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

    auto topology_metrics = [&]() -> Element {
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

    auto attention_metrics = [&]() -> Element {
        if (!av.valid || av.n_query == 0 || av.n_head == 0) {
            return vbox({text("waiting for attention...") | dim});
        }
        int h = sel_head < 0 ? 0 : (sel_head >= av.n_head ? av.n_head - 1 : sel_head);

        double sum_entropy = 0, max_w = 0, self_sum = 0, sink_sum = 0;
        for (int i = 0; i < av.n_query; i++) {
            double ent = 0, rowmax = 0;
            for (int j = 0; j < av.n_query; j++) {
                float p = av.at(i, j, h);
                if (p > 0) {
                    ent -= p * std::log(p);
                }
                if (p > rowmax) {
                    rowmax = p;
                }
            }
            sum_entropy += ent;
            if (rowmax > max_w) {
                max_w = rowmax;
            }
            self_sum += av.at(i, i, h);
            sink_sum += av.at(i, 0, h);
        }
        double n = av.n_query;
        char buf[64];
        auto fixed = [&](double x) {
            std::snprintf(buf, sizeof(buf), "%.3f", x);
            return std::string(buf);
        };
        return vbox({
            text("layer:       " + std::to_string(av.layer)),
            text("head:        " + std::to_string(h) + " / " + std::to_string(av.n_head - 1)),
            text("score shape: [" + std::to_string(av.n_kv) + ", " +
                 std::to_string(av.n_query) + ", " + std::to_string(av.n_head) + "]"),
            text("node lat:    " + fixed(av.ms) + " ms"),
            separator(),
            text("avg entropy: " + fixed(sum_entropy / n) + " nats"),
            text("peak weight: " + fixed(max_w)),
            text("self-attn:   " + fixed(self_sum / n)),
            text("sink (tok0): " + fixed(sink_sum / n)),
        });
    };

    auto metrics_panel = [&]() -> Element {
        if (active_selector == 1) {
            return attention_metrics();
        }
        return topology_metrics();
    };

    auto attention_panel = [&] {
        if (!av.valid || av.n_query == 0 || av.n_head == 0) {
            return vbox({text("waiting for prompt-decode attention...") | dim});
        }
        int h = sel_head < 0 ? 0 : (sel_head >= av.n_head ? av.n_head - 1 : sel_head);

        Elements out;
        out.push_back(text("layer " + std::to_string(av.layer) + "/" + std::to_string(n_layers - 1) +
                           "   head " + std::to_string(h) + "/" + std::to_string(av.n_head - 1) +
                           "   seq " + std::to_string(av.n_query)));
        out.push_back(text("←/→ head   ↑/↓ layer") | dim);

        Elements col_hdr;
        col_hdr.push_back(text("       "));
        for (int j = 0; j < av.n_query; j++) {
            char b[8];
            std::snprintf(b, sizeof(b), "%2d", j % 100);
            col_hdr.push_back(text(b) | dim);
        }
        out.push_back(hbox(std::move(col_hdr)));

        for (int i = 0; i < av.n_query; i++) {
            Elements cells;
            std::string lab = i < (int) av.labels.size() ? av.labels[i] : std::to_string(i);
            cells.push_back(text(short_label(lab)) | dim);
            for (int j = 0; j < av.n_query; j++) {
                float w = av.at(i, j, h);
                if (w < 0) w = 0;
                if (w > 1) w = 1;
                int v = (int) (30 + 225 * w);
                cells.push_back(text("  ") | bgcolor(Color::RGB(v, v, v)));
            }
            out.push_back(hbox(std::move(cells)));
        }
        return vbox(std::move(out)) | frame | flex;
    };

    auto stream_panel = [&] {
        size_t total = shared.ring.total_count();
        std::vector<Capture> recent = shared.ring.snapshot();
        Elements lines;
        lines.push_back(text("captured nodes: " + std::to_string(total)));
        lines.push_back(text("   id     t(ms)  type   dev   node") | dim);
        lines.push_back(separator());
        size_t show = recent.size() < 14 ? recent.size() : 14;
        for (size_t i = recent.size() - show; i < recent.size(); i++) {
            const Capture & c = recent[i];
            char buf[192];
            std::snprintf(buf, sizeof(buf), "%6zu %8.2f  %-5s %-4s %-22.22s",
                          c.id, c.t_ms, type_name(c.type), c.device.c_str(), c.name.c_str());
            lines.push_back(text(buf) | color(color_of(c.type)));
        }
        return vbox(std::move(lines)) | frame | flex;
    };

    auto ledger_panel = [&] {
        Elements lines;
        lines.push_back(text("  node                      mean       min       max       std   spars  flag") | dim);
        lines.push_back(separator());
        for (size_t i = 0; i < flat.size(); i++) {
            const Capture & c = flat[i];
            std::string nm = c.name.size() > 24 ? c.name.substr(0, 24) : c.name;
            char buf[224];
            Color col = Color::Default;
            if (!c.stats_ok) {
                std::snprintf(buf, sizeof(buf), "  %-24s  (no stats)", nm.c_str());
                col = Color::GrayDark;
            } else {
                const char * flag = "";
                if (c.n_bad > 0) {
                    flag = "NaN/Inf";
                    col = Color::Red;
                } else if (std::fabs(c.vmax) > 1000 || std::fabs(c.vmin) > 1000) {
                    flag = "large";
                    col = Color::Yellow;
                }
                std::snprintf(buf, sizeof(buf), "  %-24s %9.3f %9.3f %9.3f %9.3f %4.0f%%  %s",
                              nm.c_str(), c.mean, c.vmin, c.vmax, c.vstd, c.sparsity * 100, flag);
            }
            Element e = text(buf) | color(col);
            if ((int) i == led_cursor && focused == 4) {
                e = e | inverted | focus;
            }
            lines.push_back(e);
        }
        if (flat.size() < 3) {
            lines.push_back(text("collecting...") | dim);
        }
        return vbox(std::move(lines)) | frame | flex;
    };

    auto renderer = Renderer([&] {
        rebuild();
        Element title = hbox({
            text(" llm-trace ") | bold | bgcolor(Color::Blue) | color(Color::White),
            text(" live ggml forward-pass tracer ") | dim,
            filler(),
            text("Attn ")  | color(Color::Red),
            text("MLP ")   | color(Color::Cyan),
            text("Norm ")  | color(Color::Yellow),
            text("Embed ") | color(Color::Magenta),
            text("Other ") | color(Color::GrayLight),
        });
        Element status =
            text(" Tab: panel   j/k: move   space: expand   ←→ head  ↑↓ layer   f: fullscreen attn   q: quit ") | inverted;

        if (attn_full) {
            return vbox({
                title,
                panel("Attention  (f / Esc to exit)", attention_panel(), 1) | flex,
                status,
            });
        }

        return vbox({
            title,
            hbox({
                panel("Model Topology", topology_panel(), 0) | flex,
                panel("Attention", attention_panel(), 1) | flex,
            }) | flex,
            hbox({
                panel("Runtime Metrics", metrics_panel(), 2) | flex,
                panel("Live Stream", stream_panel(), 3) | flex,
            }) | size(HEIGHT, EQUAL, 9),
            panel("Numerical Anomaly Ledger", ledger_panel(), 4) | size(HEIGHT, EQUAL, 8),
            status,
        });
    });

    auto component = CatchEvent(renderer, [&](Event e) {
        if (e == Event::Character('q')) {
            screen.Exit();
            return true;
        }
        if (e == Event::Escape) {
            if (attn_full) {
                attn_full = false;
            } else {
                screen.Exit();
            }
            return true;
        }
        if (e == Event::Character('f')) {
            attn_full = !attn_full;
            if (attn_full) {
                focused = 1;
                active_selector = 1;
            }
            return true;
        }
        if (e == Event::Tab) {
            if (!attn_full) {
                focused = (focused + 1) % 5;
                if (focused == 0 || focused == 1) {
                    active_selector = focused;
                }
            }
            return true;
        }
        if (focused == 0) {
            if (e == Event::Character('j') || e == Event::ArrowDown) {
                if (cursor + 1 < (int) rows.size()) cursor++;
                return true;
            }
            if (e == Event::Character('k') || e == Event::ArrowUp) {
                if (cursor > 0) cursor--;
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
        if (focused == 1) {
            if (e == Event::ArrowLeft) {
                if (sel_head > 0) sel_head--;
                return true;
            }
            if (e == Event::ArrowRight) {
                if (av.n_head > 0 && sel_head + 1 < av.n_head) sel_head++;
                return true;
            }
            if (e == Event::ArrowUp) {
                if (sel_layer > 0) sel_layer--;
                return true;
            }
            if (e == Event::ArrowDown) {
                if (sel_layer + 1 < n_layers) sel_layer++;
                return true;
            }
        }
        if (focused == 4) {
            if (e == Event::Character('j') || e == Event::ArrowDown) {
                if (led_cursor + 1 < (int) flat.size()) led_cursor++;
                return true;
            }
            if (e == Event::Character('k') || e == Event::ArrowUp) {
                if (led_cursor > 0) led_cursor--;
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
