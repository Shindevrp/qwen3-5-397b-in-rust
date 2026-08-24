# Model Architecture

Qwen3.5-397B-A17B has 60 layers, 4096 hidden, 512 experts per MoE layer, top-10 routing.

## Layer mix

```mermaid
flowchart TB
    subgraph L0["Layer 0 dense"]
        D0[RMSNorm → SwiGLU]
    end
    subgraph L1["Layers 1-59"]
        direction TB
        A[Attn Norm RMSNorm]
        B[DeltaNet if layer%4≠0<br/>Full Attn if layer%4=0]
        C[Post Norm]
        D[MoE Top-10 + shared]
    end
    A --> B --> C --> D
```

- **45 delta-net layers**: layer_idx % 4 != 0
- **15 full attention layers**: layer_idx % 4 == 0

## DeltaNet recurrence

Fixed size state per head: 128×128.

```mermaid
flowchart TB
    x --> Norm[RMSNorm]
    Norm --> Proj[Wqkv GEMV]
    Proj --> Gate[β sigmoid, α decay]
    Gate --> Conv[conv1d k=4 + SiLU]
    Conv --> Rec[DeltaNet recurrence S_t]
    Rec --> Out[Gated RMSNorm]
```

State update is O(1) per token.

## Full attention

```mermaid
flowchart TB
    x --> Norm[RMSNorm]
    Norm --> QKV[Wq, Wk, Wv]
    QKV --> RoPE[IMRoPE sections [11,11,10,0]]
    RoPE --> Flash[Flash Attention GQA 16:1]
    Flash --> Gate[sigmoid × Wo]
```

Paged KV cache with optional Q8 quantization.

## MoE

```mermaid
flowchart LR
    x --> Router[Softmax → top-10]
    Router --> Experts[Stream 10 experts Q4_K]
    Experts --> Shared[Shared expert resident]
    Experts --> Sum[Weighted sum + shared]
```

Routing:
- Gate logits → top-k
- Combine weights = softmax over top-k
- Shared expert added unweighted

## Invariants

- `full_attention_interval = 4`
- `expert_count = 512`, `expert_used_count = 10`
- `rope_sections = [11,11,10,0]`
