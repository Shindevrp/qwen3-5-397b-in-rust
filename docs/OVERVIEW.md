# System Overview

## End-to-end flow

```mermaid
sequenceDiagram
    participant U as CLI
    participant F as fetch.rs
    participant HF as HuggingFace
    participant G as GGUF loader
    participant M as ModelWeights mmap
    participant P as Pipeline
    participant T as Tokenizer
    participant S as Sampler

    U->>F: fetch repo
    F->>HF: download shards + tokenizer.json
    HF-->>F: bytes
    U->>G: open shard 00001
    G->>M: mmap + shard discovery
    U->>T: load tokenizer.json
    U->>P: prefill prompt
    P->>M: embed + 60 layers
    loop decode
        P->>M: delta-net / flash attn / MoE streaming
        P->>S: logits → sample
        S-->>U: token
    end
```

## Core components

- **gguf/**: v3 parser, metadata, multi-shard, memmap2
- **model/config.rs**: typed config + validation
- **model/quant.rs**: Q8/Q4K/Q5K/Q6K codecs
- **model/kernels.rs**: norms, GEMV/GEMM, IMRoPE, flash attn, delta-net, MoE
- **model/loader.rs**: shard assembly → ModelWeights
- **model/pipeline.rs**: prefill/decode, paged KV, scheduler, verify_draft
- **tokenizer/**: tokenizer.json loader

## Invariants

1. Byte-identical output regardless of RAM.
2. Config reader refuses to guess missing fields.
3. Tokenizer is byte-level BPE, deterministic across platforms.
4. All kernels have scalar reference + SIMD parity.

## Validation pyramid

```mermaid
flowchart TB
    T1[Golden references NumPy] --> T2[Path parity opt vs naive]
    T2 --> T3[End-to-end synthetic models]
```

106 tests, no checkpoint required.
