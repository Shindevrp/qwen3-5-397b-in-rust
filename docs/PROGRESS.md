# Progress & Next Steps — Qwen3.5-397B Rust Engine

> Status snapshot: **2026-09-01 (rev 2)** — flat-logits resolved; content-loss
> hunt in progress, paused.
> This document records where the project stands, what was accomplished, and
> the concrete next steps — so work can resume cleanly after a pause.

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

## 2. Current status: NOT achieving objective (peaked-but-wrong logits)

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
- The **near-flat logits problem is gone** (2026-09-01 session): logits are now
  peaked and prompt-dependent. Details in § 3a below.

### What is broken
- **End-to-end output is still wrong**: argmax is low-information structural
  junk instead of content words. `"The capital of France is"` → top5
  `Ġ`(220), `,`(11), `Ġ(`(318), `2`(17), `Ċ`(198); `"Hello world"` →
  `0ER 2` (digits/letters); `"1+1="` → `Ġ`. Hidden RMS grows 0.7→7.3 across
  layers, logit spread 29–48 (top logit 15–27) — signal is strong but the
  *direction/content* is corrupted or lost.
- History: before the SSM fix the run was a garbage loop (`947→622→947→54087`);
  after the decay-gate fix the logits became peaked but the argmax moved to a
  stable `Ġ` space; it is still not `ĠParis`.

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

## 4. Where to resume: the remaining content-loss hunt

The delta-net math is verified, quantization decode is verified, yet the output
is peaked-but-content-free. The investigation below is what was done in the
**2026-09-01 session** and where it stood when paused.

### 4a. Ruled OUT (verified against real model + authoritative refs)

1. **Layer routing**: `is_recurrent(layer) = !(layer+1).is_multiple_of(4)`;
   full-attn layers are `il % 4 == 3` (3,7,…59). Loader tensor names/layout
   match GGUF. NOT the bug.
2. **Config dims vs GGUF**: head_k=head_v=256, n_head=32, n_kv_head=2,
   ssm_state_size=128, ssm_group_count=16, ssm_time_step_rank=64,
   ssm_inner_size=8192, conv_dim=12288, key_dim=2048, value_dim=8192,
   ba_dim=8192, n_q_full=16384, n_kv_size=512 — all match real tensor shapes.
3. **Quantization decode**: our `dequantize_q4_k/q5_k/q6_k` match
   `ref/ggml-quants.c` (authoritative ggml source) line-for-line.
4. **SIMD gemv**: `gemv_parallel` == naive dequant-dot on **real model bytes**
   (Q4_K/Q5_K/Q6_K) within Q-noise (rms_err ~0.002, ref_rms ~0.4). Verified via
   `src/bin/gemvcheck.rs` on `attn_qkv/attn_gate/ssm_beta/ssm_alpha/attn_q/k/v/
   attn_output/ffn_gate_inp`.
5. **Delta-net AR math** (`delta_net_autoregressive`): q/k L2-norm, q scaled by
   `1/sqrt(S_v)`, GDA gate broadcast over both state dims, `k_state =
   state^T k_norm`, `v_diff = v - k_state`, outer-product state update,
   `out = state^T q_norm` — verified against `build_delta_net_unified_autoregressive`.
6. **Gated norm**: `rms_norm_per_head(attn_out, ssm_norm_w, s_v) * silu(z)`
   matches `build_norm_gated`. (Note: `attn_gate`/`z` used ONLY for gated norm.)
7. **Recurrent state is NOT collapsing**: state RMS 0.09→0.26 across layers,
   decay `exp(g)` range [0,1] healthy; attn_out RMS 0.017→0.09.
8. **FFN bypass** (`QWEN35_NOFNN=1`, attention-only): still codes to
   structural junk (top5 `Ġ`,`2`,digits) — so the FFN/MoE is not the *sole*
   culprit; the attention/residual content itself is degraded. **Do not
   conclude MoE is clean** — FFN is essential for gen and the test only showed
   attention alone is also wrong-colored.

### 4b. Still suspicious / not yet verified (resume here, in order)

1. **conv1d window & kernel orientation** — the one block NOT yet diffed
   against ggml. ggml `ggml_ssm_conv` semantics (state prepend + which kernel
   tap lands on the newest token, SiLU after conv) must be confirmed against the
   real op (fetch `ggml/src/ggml-cpu.c` `ggml_compute_forward_ssm_conv_f32` or
   the branch/PR that added it). An off-by-one here corrupts the whole
   recurrent signal while everything else "looks right".
2. **Full-attention layers** (15 of 60, layers 3,7,…59): `wq` = Q + sigmoid
   gate split, QK-norm, IMRoPE (`rope_sections`), GQA 16:1 scaling, sigmoid-
   gated `attn*σ(gate)` then `×wo` — diff `full_layer_forward_q` against
   `build_layer_attn` in the patch.
3. **MoE FFN path** (512 experts / 10 used, streaming per-expert Q4_K from
   shards): routing / top-k renormalization (`norm_w`), per-expert SwiGLU,
   shared-expert `σ(gate_imp)` weighting, and **expert weight byte layout
   across shards** (the MoE tensors are the only ones not covered by
   `gemvcheck`-style real-byte verification).
4. **LM head / output**: `output.weight` Q6_K [4096, 248320] is loaded and the
   argmax path structurally matches llama's `llm_output`; the "embedding domain"
   test (`src/bin/lmtest.rs` — raw embedding → lm_head, no layers) produces
   unrelated argmax, i.e. the embedding↔output correlation is low; whether that
   is expected for an untied head is unconfirmed.
5. **Missing reference**: llama.cpp is NOT installed on this machine (no
   `llama-cli`), so a full layer-by-layer numeric diff vs llama.cpp was not
   possible. Building/running llama.cpp on this 397B model is infeasible locally
   (23 GB RAM). Strongest future tool: a side-by-side hidden-state dump from a
   machine that can run llama.cpp.

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
- `src/bin/probe.rs`, `src/bin/gemvcheck.rs`, `src/bin/lmtest.rs`,
  `src/bin/dumpsmall.rs` — diagnostics (see § 7a)
- `src/gguf/` — GGUF reader
- `tests/golden/spec.txt` — golden test configs
- References (in repo): `ref/qwen35moe-pr.patch`, `ref/llama-arch.cpp`,
  `ref/llama-graph.cpp`; docs in `docs/`.

## 7. Important notes for resuming

- Rebuild before running; `target/` may be stale.
- Check `git status` / `git log` for any uncommitted work.
- The `already-fixed` list above (Section 3) must NOT be re-broken — the f32
  and q paths must stay in agreement.
- The run is memory-hungry; keep `--n-predict` small and use long timeouts.
- Do NOT remove `n_vocab = n_elements()/n_embd`. Do NOT add NaN sanitization.

### 7a. Temporary instrumentation left in the tree (clean up or keep deliberately)

These are **uncommitted** and were added for this session's diagnostics.
Decide their fate before the next commit:

- `src/bin/probe.rs` — prints real config + tensor shapes/dims (keep, harmless).
- `src/bin/gemvcheck.rs` — SIMD-gemv vs dequant-dot on real bytes (keep, useful).
- `src/bin/lmtest.rs` — raw-embedding→lm_head argmax test (keep, useful).
- `src/bin/dumpsmall.rs` — dumps small f32 tensors (remove or keep).
- Env-gated debug prints in `kernels.rs` / `pipeline.rs` gated behind
  `QWEN35_DBG=1` (per-layer RMS / cosine-vs-embedding / state RMS / decay / logit
  top5). Currently dormant unless the env var is set — decide to keep or strip.
- `QWEN35_NOFNN=1` env bypass in `delta_net_layer_forward_q` and
  `full_layer_forward_q` (skips the FFN entirely) — used for the attention-only
  test. **Remove before any release commit** unless deliberately retained.

Debug command used this session:
```
QWEN35_DBG=1 ./target/release/run <gguf> <tok> "The capital of France is" \
  --memory-bounded --kv-q8 --n-predict 0|1 --argmax    # per-layer + logit stats
QWEN35_NOFNN=1 ... --n-predict 1 --argmax             # attention-only test
./target/release/gemvcheck <shard1.gguf>              # real-byte quant check
./target/release/lmtest <shard1.gguf>                 # embedding-domain test
```

### 7b. Unresolved questions carried forward

- Whether the recurrent state magnitude (state RMS ≈ 0.09–0.26 over S_v²·H_v
  elements) is correct — unresolved.
- Whether the embedding↔output-weight correlation being low (`lmtest`) is
  expected for an untied head — unconfirmed.
- ggml `ggml_ssm_conv` (conv window/kernel orientation) not yet diffed — see § 4b.
