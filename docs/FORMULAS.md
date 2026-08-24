# Formulas

## RMSNorm

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

## DeltaNet recurrence

Conv + gating:
$$
h_t = \text{SiLU}(\text{Conv1d}(x_t))
$$
$$
g_t = \sigma(W_g h_t), \quad \alpha_t = \text{sigmoid}(W_\alpha h_t)
$$
$$
S_t = \alpha_t S_{t-1} + h_t k_t^\top
$$
$$
o_t = S_t q_t
$$

State $S_t \in \mathbb{R}^{128 \times 128}$ per head, fixed size.

## IMRoPE

Rotary position embedding with interleaved dimensions.
Sections $[11,11,10,0]$ define frequency bands.

## MoE routing

Gate logits:
$$
g_i = W_g x
$$

Top-k selection:
$$
\mathcal{T} = \text{topk}(g_i, k=10)
$$

Combine weights:
$$
w_i = \frac{\exp(g_i)}{\sum_{j\in\mathcal{T}} \exp(g_j)} \;\; \text{for } i\in\mathcal{T}
$$

Output:
$$
y = \sum_{i\in\mathcal{T}} w_i \, \text{Expert}_i(x) + \sigma(\text{gate}) \cdot \text{Shared}(x)
$$

## SwiGLU

$$
\text{SwiGLU}(x) = \text{SiLU}(x W_g) \odot (x W_u) W_d
$$

## Quantization

Q4_K block quantization:
- Block size 32
- Scale per block
- 4 bits per weight

Dequant:
$$
w = \text{code} \times \text{scale}
$$

## Sampling

Temperature:
$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Top-k / top-p applied after temperature.
