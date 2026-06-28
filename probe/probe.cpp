#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

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

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true;
    }

    LayerType type = classify(t->name);
    LOG("%-20s | %-5s | %-10s | %-5s | [%lld, %lld, %lld, %lld]\n",
        t->name,
        type_name(type),
        ggml_op_desc(t),
        ggml_type_name(t->type),
        (long long) t->ne[0],
        (long long) t->ne[1],
        (long long) t->ne[2],
        (long long) t->ne[3]);

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

    params.cb_eval = trace_cb;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to load model\n");
        return 1;
    }

    LOG("\n");
    LOG("%-20s | %-5s | %-10s | %-5s | shape\n", "name", "type", "op", "dtype");
    LOG("-------------------------------------------------------------------------\n");

    if (!run(ctx, params)) {
        return 1;
    }

    llama_backend_free();
    return 0;
}
