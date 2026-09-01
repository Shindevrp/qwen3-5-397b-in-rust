# Progress & Next Steps — Qwen3.5-397B Rust Engine

> Status snapshot: **2026-09-01**
> This document records where the project stands, what was accomplished, and
> the concrete next steps — so work can resume cleanly after a ~3-month pause.

## 1. Objective

Get **correct generation** from the real `Qwen3.5-397B-A17B` GGUF.
Acceptance prompt: `"The capital of France is"` should produce `ĠParis`
(leading space + "Paris").

- Model: `/home/shinde/models/qwen35-397b/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00007.gguf`
  (7-shard Q4_K_M, ~240 GB, mmapped)
- Tokenizer: `/home/shinde/models/qwen35-397b/tokenizer.json`
- Repo: `/home/shinde/qwen3-5-397b-in-rust`

Run command:
```
./target/release/run <gguf> <tokenizer> "The capital of France is" --memory-bounded --kv-q8 --n-predict 1 --argmax
```

## 2. Current status: NOT achieving objective (flat logits)

### What works
- Builds clean (`cargo build --release`).
- All unit/regression tests pass: **106 lib tests + 5 crossval tests, 0 failed**.
- Correct architecture confirmed against llama.cpp reference:
  - dedicated `ssm_beta` / `ssm_alpha` projections
  - correct decay gate formula (see below)
  - correct `ba_dim`
  - gated RMS per-head norm with `silu(z)`
  - shared-expert gating (`σ(gate)`) matches reference
- Recurrent (delta-net) path is now **architecturally correct** per
  `qwen35moe-pr.patch` (llama.cpp `qwen3-5.cpp`).

### What is broken
- **End-to-end output is wrong**: prompt yields token 220 (`Ġ`, space) repeated,
  NOT `ĠParis`. Logits are **near-flat** (top-5 spread ≈ 3; top5 roughly
  220:22.02, 11:20.01, 318:19.98, 17:19.26, 16:18.83).
- Improvement already achieved: before fixing the SSM, output was a garbage
  loop (`947→622→947→54087`); after the fix it became a stable `Ġ` space —
  proving the SSM fix path is exercised and changed behavior, but it is not
  sufficient to get coherent logits.

## 3. Root cause confirmed & fixed (delta-net SSM decay)

The confirmed, FIXED root cause was the **decay gate** in the delta-net layer.

Exact reference formula (llama.cpp `qwen3-5.cpp` `build_layer_attn_linear`):
```
alpha_biased = W_alpha @ x + ssm_dt.bias
gate         = softplus(alpha_biased) * ssm_a        # ssm_a stored as log(A), A in (0,1]
state       *= exp(gate)                              # applied in delta_net_autoregressive
```

- **Was (wrong)**: `decay_gate = -exp(ssm_a) * softplus(alpha+bias)`
  — this killed the state (decay range gave wrong scale).
- **Now (correct)**: `decay_gate = ssm_a * softplus(alpha+bias)`.
  The AR kernel applies `state *= exp(decay_gate)`.

Other recalled fixes in this area (all verified against reference, all in the
recalled `kernels.rs`/`pipeline.rs`/`synth.rs`/`config.rs`/`crossval_tests.rs`):
- `ssm_beta` / `ssm_alpha` are separate projections `[n_embd, n_heads_v]`
  (the GGUF stores them separately — confirmed `ssm_beta.weight` /
  `ssm_alpha.weight` present, no combined `ssm_beta_alpha`).
- `attn_gate` (z) is used ONLY for the gated norm, not for beta/alpha.
- `ba_dim = s_v * n_heads_v = v_size` (asserted).
- `n_heads_v = ssm_time_step_rank`.
- Gated norm: `rms_norm_per_head(attn_out, ssm_norm_w, s_v, eps)[i] * silu(z[i])`.
- Decay formula fixed in **both** the f32 path and the quantized (q) path.
- Steamline/consistency: f32 and q paths now agree.

All four `delta_net_layer_forward_q` call sites in `pipeline.rs` updated to pass
`(ssm_beta_w)`, `(ssm_alpha_w)`, and `ssm_norm_w`.

### Note on state RMS
Earlier diagnostics: `state_norm=93.45` over 1,048,576 state elements
(RMS ≈ 0.09). Whether the recurrent state should have larger magnitude is an
open question to re-examine when resuming.

## 4. Where to resume: the remaining flat-logits hunt

The delta-net is correct, layer norms are healthy, and the delta-net residual
(22.62 vs input 0.59) is strong — yet final logits are near-flat. Investigate
these, in order:

1. **Full-attention layers** (`layer_idx % 4 == 0`, every 4th, 15 of 60).
   Verify the `wq` split (Q + sigmoid gate), QK-norm, IMRoPE, GQA scaling, and
   the sigmoid-gated output (`cur = attn * σ(gate)` then `× wo`) against
   `build_layer_attn` in the patch.
2. **MoE FFN path** — the model is the **MoE variant**
   (`Qwen3.5-397B-A17B`, expert_count 512, n_expert_used 10). Verify routing,
   top-k weight normalization, per-expert SwiGLU, and shared-expert weighting
   against `qwen3-5moe.cpp` `build_layer_ffn` / `build_moe_ffn`. Check whether
   expert gate/up handling and the router are correct (see `moe_ffn_stream`).
3. **LM head / output scaling** — check `output_norm` + `output_weight`
   (`n_vocab = n_elements()/n_embd`), and whether logits are being scaled or
   flattened. Suspects: output-weight content, norm scaling, or f32 precision
   loss in some path.
4. **Precision**: a global 32-bit (f32) degradation in some path (e.g. MoE,
   cache q8, or large accumulators) is unconfirmed but possible.
5. **Reference to fetch/verify against**:
   - Dense: `https://github.com/ggml-org/llama.cpp/blob/6c8dcaa7/src/models/qwen35.cpp`
   - MoE: `https://github.com/ggml-org/llama.cpp/blob/fc2b0053/src/models/qwen35moe.cpp`
   - Local copy of the patch: `ref/qwen35moe-pr.patch` (already in repo).

A direct numerical comparison is a strong tool: run the same 5-token prompt
through llama.cpp and compare hidden norms / logits layer-by-layer to localize
the divergence.

## 5. Verification commands

```
cargo build --release
cargo test --release        # expect 106 lib + 5 crossval passing
cargo test --release -- --nocapture   # to see any eprintln diagnostics
```

Real run (memory-bound; keep n_predict small, generous timeouts):
```
./target/release/run /home/shinde/models/qwen35-397b/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00007.gguf \
  /home/shinde/models/qwen35-397b/tokenizer.json \
  "The capital of France is" --memory-bounded --kv-q8 --n-predict 1 --argmax
```
Constraints: no NaN sanitization; keep `n_vocab = n_elements()/n_embd`;
keep Phase 31 per-tensor MoE streaming; environment is RAM-limited.

## 6. Key files

- `src/model/kernels.rs` — core fix site (f32 + q delta-net forward,
  `delta_net_autoregressive` applies `exp(gate)`, `rms_norm_per_head`,
  `full_layer_forward_q`, `moe_ffn_stream`, `shared_expert_ffn`)
- `src/model/pipeline.rs` — loader + 4 delta-net call sites, full-attn layers
- `src/model/config.rs` — config parsing / validation
- `src/model/synth.rs` — synthetic GGUF for tests (real shapes + `ssm_beta`/
  `ssm_alpha`/`ssm_norm`/`ssm_dt`)
- `src/model/crossval_tests.rs` — fixtures, thresholds, debug eprintlns
- `src/bin/run.rs`, `src/bin/kern_check.rs` — CLI / kernel checker
- `src/gguf/` — GGUF reader
- `tests/golden/spec.txt` — golden test configs
- References (in repo): `ref/qwen35moe-pr.patch`, `ref/llama-arch.cpp`,
  `ref/llama-graph.cpp`; docs in `docs/`.

## 7. Important notes for resuming (3-month gap)

- Rebuild before running; `target/` may be stale.
- Check `git status` / `git log` for any uncommitted work.
- The `already-fixed` list above (Section 3) must NOT be re-broken — the f32
  and q paths must stay in agreement.
- The run is memory-hungry; keep `--n-predict` small and use long timeouts.
- Do NOT remove `n_vocab = n_elements()/n_embd`. Do NOT add NaN sanitization.
