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

Decay gate (matches `qwen3-5.cpp` `build_layer_attn_linear`):
$$
\alpha_t = \log(A) \cdot \text{softplus}(W_\alpha h_t + b)
$$
where $A \in (0,1]$ is the learned per-head `ssm_a` (stored as $\log A$), and
$b$ is `ssm_dt.bias`. The state is scaled by $\exp(\alpha_t)$ each step:
$$
S_t = e^{\alpha_t} S_{t-1} + h_t k_t^\top
$$
$$
o_t = S_t q_t
$$

State $S_t \in \mathbb{R}^{128 \times 128}$ per head, fixed size.

## IMRoPE

Rotary position embedding with interleaved dimensions.
Sections $[11,11,10,0]$ define frequency bands.

RoPE rotates query and key vectors:

$$
R_{\theta,m} = \begin{pmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{pmatrix}
$$

$$
\theta_i = 10000^{-2i/d}
$$

For interleaved RoPE, dimensions are interleaved per section.

IMRoPE applies RoPE only to the first $q k_{rope}$ dimensions, leaving $q k_{nope}$ unchanged.

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

## Flash Attention

Online softmax to avoid materializing full $N \times N$ matrix:

$$
m_i = \max(m_{i-1}, \max_j q_i k_j^\top)
$$
$$
\ell_i = \exp(m_{i-1} - m_i) \ell_{i-1} + \sum_j \exp(q_i k_j^\top - m_i)
$$
$$
o_i = \frac{\exp(m_{i-1} - m_i) \ell_{i-1} o_{i-1} + \sum_j \exp(q_i k_j^\top - m_i) v_j}{\ell_i}
$$

Memory $O(N)$ instead of $O(N^2)$.

## Sampling

Temperature:
$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Top-k / top-p applied after temperature.

Repeat penalty:
$$
z_i \leftarrow z_i / \text{penalty} \quad \text{if token already generated}
$$

Top-p nucleus:
$$
\sum_{i \in \text{top-p}} p_i \ge p
$$

Greedy decoding is deterministic and byte-identical across runs.
