#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

enum LayerType { ATTN, MLP, NORM, EMBED, OTHER };

inline LayerType classify(const std::string & name) {
    if (name.find("ffn") != std::string::npos)  return MLP;
    if (name.find("attn") != std::string::npos) return ATTN;
    if (name.find("norm") != std::string::npos) return NORM;
    if (name.find("embd") != std::string::npos) return EMBED;
    return OTHER;
}

inline const char * type_name(LayerType t) {
    switch (t) {
        case ATTN:  return "Attn";
        case MLP:   return "MLP";
        case NORM:  return "Norm";
        case EMBED: return "Embed";
        default:    return "Other";
    }
}

struct Capture {
    size_t      id;
    double      t_ms;     // time since trace start
    std::string name;
    LayerType   type;
    std::string op;
    std::string dtype;
    std::string device;
    int64_t     ne[4];
    double      ms;       // latency since previous node

    bool        stats_ok = false;
    double      mean = 0, vmin = 0, vmax = 0, vstd = 0, sparsity = 0;
    int         n_bad = 0;   // count of NaN/Inf values
};

// Shared between the inference thread (push) and the UI thread (snapshot).
// Fixed-size ring buffer, guarded by a mutex.
struct TraceBuffer {
    std::vector<Capture> buf;
    size_t head = 0;
    size_t count = 0;
    mutable std::mutex mtx;
    std::atomic<size_t> total{0};

    TraceBuffer(size_t capacity) : buf(capacity) {}

    void push(const Capture & c) {
        std::lock_guard<std::mutex> lock(mtx);
        buf[head] = c;
        head = (head + 1) % buf.size();
        if (count < buf.size()) {
            count++;
        }
        total.fetch_add(1, std::memory_order_relaxed);
    }

    std::vector<Capture> snapshot() const {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<Capture> out;
        out.reserve(count);
        size_t start = (count == buf.size()) ? head : 0;
        for (size_t i = 0; i < count; i++) {
            out.push_back(buf[(start + i) % buf.size()]);
        }
        return out;
    }

    size_t total_count() const {
        return total.load(std::memory_order_relaxed);
    }
};
