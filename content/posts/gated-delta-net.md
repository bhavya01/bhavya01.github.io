+++
title = "From Linear Attention to Gated DeltaNet"
date = 2026-07-20
draft = true
tags = ["attention", "linear-attention", "transformers", "sequence-models"]
description = "A background on linear attention, and how the delta rule and gating mechanisms lead to Gated Delta Net."
+++

## Background: Standard Dot Product Attention and Its Cost

The Transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))
computes attention as a softmax over the pairwise dot products of queries and keys:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

For a sequence of length `n` and head dimension `d`, `Q K^T` is an `n x n` matrix — every
token computes a similarity score against every other token. This is the source of both
the model's representational power and its cost. Because every pair of positions gets a
direct, content-dependent interaction, the model can route information between any two
tokens in a single layer regardless of how far apart they are, which is a big part of why
Transformers are so good at in-context learning and long-range dependencies. But that same
`n x n` matrix means the compute and memory needed for one attention layer scale as
`O(n^2 * d)` and `O(n^2)` respectively — quadratic in sequence length. Double the context
window and you quadruple the cost of every attention layer.

**Inference and the KV cache.** Training and prefill process a whole sequence at once, but
autoregressive decoding generates one token at a time. Naively, generating token `n+1`
would require recomputing the keys and values for all `n` previous tokens from scratch.
Instead, every serving system caches the per-layer, per-head key and value vectors as they
are produced, so each decode step only has to compute the new token's own K/V and attend
over the cached history. This is the **KV cache**, and it's what makes decoding roughly
linear in sequence length instead of quadratic per step. The catch is that the cache itself
is not free: its size is

```
KV cache size = 2 x num_layers x num_kv_heads x head_dim x seq_len x batch_size x bytes_per_element
              = 2 x num_layers x d_model x seq_len x batch_size x bytes_per_element   (for MHA)
```

That size grows **linearly with sequence length**, and it has to live in accelerator
memory for the entire duration of a request. Unlike the model weights, which are fixed, the
KV cache competes with weights and activations for the same HBM budget and grows with every
token generated and with every concurrent request served (see
[Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102)
for a detailed treatment of this bottleneck).

**A more realistic estimate.** Modern large models don't use plain multi-head attention —
they use grouped-query attention (GQA,
[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)), where many query heads share a
much smaller number of key/value heads, precisely to shrink the KV cache. Take a backbone
with `128` layers, `d_model = 16384`, `128` query heads (`head_dim = 128`), but only `8`
KV heads — an 16:1 GQA ratio similar to Llama 3's largest models — served in bf16:

```
KV cache per token per layer = 2 x num_kv_heads x head_dim x bytes
                              = 2 x 8 x 128 x 2 bytes = 4 KiB
KV cache per token (all 128 layers) = 4 KiB x 128 = 512 KiB
```

Even with GQA cutting the per-token cost by 16x relative to plain MHA, the cache still
grows linearly and gets large fast for a single sequence:

| Context length | KV cache (batch = 1) |
|---|---|
| 8K tokens   | ~4 GB   |
| 32K tokens  | ~16 GB  |
| 128K tokens | ~64 GB  |
| 1M tokens   | ~512 GB |

Numbers like these are why the KV cache, not the model weights, is usually the binding
constraint when serving long-context requests at scale. It has to sit in the same HBM as
the weights and activations, it grows with every token generated, and it multiplies with
every concurrent request a server handles — a handful of 128K-context requests can
already outweigh the memory the weights themselves occupy. Worse, in a naive
implementation each request's cache is allocated as one contiguous block sized for the
maximum sequence length, so memory gets fragmented and wasted whenever a request finishes
early or sequences vary in length, capping how many requests can be batched together.

**PagedAttention** ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180), the algorithm
behind vLLM) targets exactly this allocation problem. A naive serving system reserves one
contiguous chunk of memory per request sized for the maximum sequence length it might
ever reach, up front — so a request that ends up short still ties up memory it never
uses, and that memory is wasted for the request's entire lifetime. PagedAttention instead
allocates the KV cache in small fixed-size blocks on demand, one block at a time as the
sequence actually grows, and uses a per-sequence block table to keep track of which
physical blocks belong to which sequence. Nothing is reserved ahead of time, so no memory
is wasted on requests that finish early, which lets the server pack many more sequences
into a batch. What it does *not* do is change the fact that total cache size still grows
linearly with sequence length; it just removes the wasted upfront reservation.

**FlashAttention and its limits.** A lot of engineering effort has gone into making the
`O(n^2)` compute cheaper to execute rather than avoiding it. FlashAttention
([Dao et al., 2022](https://arxiv.org/abs/2205.14135); FlashAttention-2,
[Dao, 2023](https://arxiv.org/abs/2307.08691)) never materializes the full `n x n`
attention matrix in slow HBM. Instead it tiles Q, K, and V into blocks that fit in fast
on-chip SRAM, computes attention block-by-block, and maintains a running (online) softmax
so the final result is exact despite never seeing the whole score matrix at once. This is
an IO-aware optimization: it doesn't change the `O(n^2 * d)` FLOP count, but it removes the
`O(n^2)` memory traffic to HBM that dominated wall-clock time, giving large real-world
speedups and enabling longer context windows during training and prefill. What
FlashAttention does **not** do is shrink the KV cache. The cache is a decode-time,
per-request storage cost — it exists regardless of how efficiently a single attention
matmul is computed, so even with FlashAttention the memory needed to serve long sequences
still grows linearly and unboundedly with sequence length. Solving *that* problem requires
rethinking the recurrence itself, which is where linear attention comes in.

## Linear Attention

<!-- Cite: Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2020) -->

## Intuition: What Linear Attention Actually Does

## The Delta Update Rule

<!-- Cite: Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" (2024) -->

## The Gated Delta Update Rule

<!-- Cite: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule" (2024) -->

## Putting It Together: Gated DeltaNet

## What's Next
