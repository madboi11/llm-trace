# llm-trace

A lightweight, non-invasive telemetry and diagnostic TUI for **local transformer
models**. It hooks into a running `llama.cpp` / `ggml` inference engine, captures
real-time intermediate state as tokens flow through the network, and renders it in an
interactive terminal UI.

> *btop / lazygit, but for watching a transformer's forward pass.*

<!-- TODO: add a screenshot or GIF here (see "Demo" below) -->

## How it works (the core idea)

This tool does **not** reimplement inference. It instruments an existing engine
(`llama.cpp`) through **ggml's per-node evaluation callback**.

`llama.cpp` exposes two fields on the context params: `cb_eval` and
`cb_eval_user_data`. The callback is invoked **once per tensor node** while the
compute graph runs, and is handed the live `ggml_tensor` — its name, op, type, and
shape (`ne[]`). That single hook is the whole capture layer.

```
llama.cpp decode  ──fires per node──►  eval callback  ──►  ring buffer + topology
   (producer thread)                   (capture layer)      (shared, mutex-guarded)
                                                                     │
                                                              FTXUI panels  ◄── UI thread
```

Three details that make this work, each handled in `src/main.cpp`:

1. **The `ask` two-phase protocol.** The callback fires twice per node: first with
   `ask == true` (the scheduler asking "do you want this node's data?" — we return
   `true`), then with `ask == false` (data is ready — we do the real work).
2. **Host copy for tensor data.** A tensor may live on a non-CPU backend, so reading
   its values requires `ggml_backend_tensor_get` after checking
   `ggml_backend_buffer_is_host`. This is what makes the attention heatmap possible.
3. **Producer / consumer threading.** Inference runs on one thread (firing callbacks);
   the TUI renders on another. The shared state — a fixed-size ring buffer and a
   topology model — is guarded by a mutex.

### Attention heatmap & flash-attention

Flash-attention fuses the softmax and never materializes a `[seq × seq]` score
tensor, so there is nothing to draw. The app forces flash-attention **off**
(`flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED`), which switches `llama.cpp` to
the explicit `softmax(QK·T)` path and exposes the `kq_soft_max-<layer>` tensor. That
tensor is copied to host and rendered as a per-head heatmap — the causal mask shows up
as a clean lower-triangular pattern.

## Panels

| Panel | What it shows |
|-------|---------------|
| **Model Topology** | Every graph node, grouped by role (`ffn_gate`, `attn_norm`, …), expandable to its per-layer instances. Colored by layer type. |
| **Attention** | `[seq × seq]` attention heatmap for a selected layer/head, with token labels. |
| **Runtime Metrics** | Context-aware: the selected node's shape/dtype/op/latency, or attention stats (entropy, peak weight, self-attention, attention-sink). |
| **Live Stream** | Scrolling feed of captured nodes — id, timestamp, type, compute device. |

### Keybindings

| Key | Action |
|-----|--------|
| `Tab` | cycle panel focus |
| `j` / `k` | move selection (Topology) |
| `space` / `Enter` | expand / collapse a role (Topology) |
| `←` / `→` | change attention head (Attention) |
| `↑` / `↓` | change attention layer (Attention) |
| `q` / `Esc` | quit |

## Build (WSL2 / Ubuntu, CPU-only)

```bash
# system dependencies (one time)
sudo apt update
sudo apt install -y build-essential cmake ninja-build pkg-config libcurl4-openssl-dev

# clone with submodules (llama.cpp + FTXUI are pinned submodules)
git clone --recurse-submodules <repo-url>
cd llm-trace

# configure + build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces two binaries in `build/`:
- `llm-trace` — the interactive TUI.
- `llm-trace-probe` — a minimal stdout probe that prints each node's metadata (the
  starting point for understanding the eval callback).

## Usage

Download a small GGUF model, then run:

```bash
./build/llm-trace -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "The cat sat on the mat" -ngl 0
```

A longer prompt gives a larger attention matrix. `-ngl 0` keeps it CPU-only.

## Demo

<!-- TODO: record a short GIF (e.g. `asciinema` / `terminalizer`) or add a screenshot,
     then embed it at the top of this README. -->

## Stack

- **Backend:** llama.cpp / ggml (pinned submodule, CPU-only build)
- **TUI:** FTXUI v7 (pinned submodule)
- **Test model:** Qwen2.5-0.5B-Instruct (GGUF)
- **Language:** C++17

## What I learned

<!-- TODO (personalize this in your own words): the ggml eval-callback hook and the
     ask two-phase protocol, reading typed data out of ggml buffers, the
     single-producer/single-consumer ring buffer + threading, building one CMake tree
     against two third-party libraries, and why flash-attention hides the score matrix. -->
