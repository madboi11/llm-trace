#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <chrono>
#include <clocale>
#include <string>
#include <vector>

enum LayerType { ATTN, MLP, NORM, EMBED, OTHER };

static LayerType classify(const std::string & name) {
    if (name.find("ffn") != std::string::npos)  return MLP;
    if (name.find("attn") != std::string::npos) return ATTN;
    if (name.find("norm") != std::string::npos) return NORM;
    if (name.find("embd") != std::string::npos) return EMBED;
    return OTHER;
}

static const char * type_name(LayerType t) {
    switch (t) {
        case ATTN:  return "Attn";
        case MLP:   return "MLP";
        case NORM:  return "Norm";
        case EMBED: return "Embed";
        default:    return "Other";
    }
}

struct Capture {
    std::string name;
    LayerType   type;
    std::string op;
    std::string dtype;
    int64_t     ne[4];
    double      ms;
};

struct RingBuffer {
    std::vector<Capture> buf;
    size_t head = 0;
    size_t count = 0;

    RingBuffer(size_t capacity) : buf(capacity) {}

    void push(const Capture & c) {
        buf[head] = c;
        head = (head + 1) % buf.size();
        if (count < buf.size()) {
            count++;
        }
    }

    size_t size() const {
        return count;
    }

    const Capture & at(size_t i) const {
        size_t start = (count == buf.size()) ? head : 0;
        return buf[(start + i) % buf.size()];
    }
};

struct TraceState {
    RingBuffer ring;
    std::chrono::steady_clock::time_point last;
    bool have_last = false;

    TraceState(size_t capacity) : ring(capacity) {}
};

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true;
    }

    TraceState * state = (TraceState *) user_data;

    auto now = std::chrono::steady_clock::now();
    double ms = 0.0;
    if (state->have_last) {
        ms = std::chrono::duration<double, std::milli>(now - state->last).count();
    }
    state->last = now;
    state->have_last = true;

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

    state->ring.push(c);

    LOG("%-20s | %-5s | %-10s | %-5s | %7.3f ms | [%lld, %lld, %lld, %lld]\n",
        c.name.c_str(),
        type_name(c.type),
        c.op.c_str(),
        c.dtype.c_str(),
        c.ms,
        (long long) c.ne[0],
        (long long) c.ne[1],
        (long long) c.ne[2],
        (long long) c.ne[3]);

    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("no input tokens, use -p to give a prompt\n");
        return false;
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("decode failed\n");
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    TraceState state(64);

    params.cb_eval = trace_cb;
    params.cb_eval_user_data = &state;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to load model\n");
        return 1;
    }

    LOG("\n");
    LOG("%-20s | %-5s | %-10s | %-5s | %-10s | shape\n", "name", "type", "op", "dtype", "latency");
    LOG("-----------------------------------------------------------------------------------\n");

    if (!run(ctx, params)) {
        return 1;
    }

    LOG("\nring buffer kept the last %zu of all traced nodes (capacity %zu):\n",
        state.ring.size(), state.ring.buf.size());
    for (size_t i = 0; i < state.ring.size(); i++) {
        const Capture & c = state.ring.at(i);
        LOG("  %-20s | %-5s | %-10s | %7.3f ms\n",
            c.name.c_str(), type_name(c.type), c.op.c_str(), c.ms);
    }

    llama_backend_free();
    return 0;
}
