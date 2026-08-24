# Documentation

Clear, formula-driven explanation of how Qwen3.5-397B-A17B runs CPU-only in safe Rust with streaming MoE and hybrid attention.

## Contents

- [Memory & Streaming Model](MEMORY_MODEL.md) – mmap, OS page cache as expert cache, memory ladder
- [Model Architecture](MODEL_ARCHITECTURE.md) – 60 layers, 45 delta-net + 15 full attention, Top-10 MoE
- [Formulas](FORMULAS.md) – RMSNorm, DeltaNet recurrence, IMRoPE, MoE routing, SwiGLU
- [System Overview](OVERVIEW.md) – end-to-end data flow and invariants

All diagrams are Mermaid and render in GitHub.
