# Reference sources for Qwen3.5-397B-A17B implementation

Fetched from llama.cpp / ggml repos (via webfetch; sandbox has no direct network).
Kept here for offline reference.

| File | Source | Notes |
|------|--------|-------|
| `qwen35moe-pr.patch` | llama.cpp PR adding Qwen3.5 MoE (delta-net) | Full patch: convert_hf_to_gguf.py, llama-graph.cpp, llama-arch, hparams |
| `llama-graph.cpp` | merged llama.cpp `src/llama-graph.cpp` | Contains `build_moe_ffn`, `llm_graph_context` helpers |
| `llama-model.cpp` | merged llama.cpp `src/llama-model.cpp` | Contains `llama_model_rope_type`, per-arch graph builders |
| `extraction-notes.md` | subagent extraction of graph/model sources | Indexed by line numbers |
| `ggml-kernels.cpp` | ggml backend kernels | `gated_delta_net`, `rope_multi` (IMRoPE) reference |
| `gguf-py-constants.py` | gguf-py `gguf/constants.py` | GGUF quant type enum / names for dequant mapping |
| `llama-arch.cpp` | llama.cpp `src/llama-arch.cpp` | Arch names + tensor name mapping |

## Key facts confirmed from these sources

- Recurrent linear-attention layers: `blk.*.attn_linear_*`; gated delta-net state math from `build_delta_net_unified*`.
- `rope_multi`: partial IMRoPE with `rope_sections` [11, 11, 10, 0] for q/k.
- MoE gating: gate proj -> softmax -> top-k -> renormalize top-k weights -> `sum w_e * down_e(silu(up_e*x) * gate_e(x))`.
