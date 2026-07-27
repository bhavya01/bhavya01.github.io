+++
title = "From Linear Attention to Gated DeltaNet"
date = 2026-07-20
draft = true
tags = ["attention", "linear-attention", "transformers", "sequence-models"]
description = "A background on linear attention, and how the delta rule and gating mechanisms lead to Gated Delta Net."
math = true
+++

## Background: Standard Dot Product Attention and Its Cost

The Transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))
computes attention as a softmax over the pairwise dot products of queries and keys:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

For a sequence of length \(n\) and head dimension \(d\), \(QK^\top\) is an \(n \times n\) matrix.
Every token computes a similarity score against every other token. This is the source of
both the model's representational power and its cost. Because every pair of positions gets
a direct, content-dependent interaction, the model can route information between any two
tokens in a single layer regardless of how far apart they are, which is a big part of why
Transformers are so good at in-context learning and long-range dependencies. But that same
\(n \times n\) matrix means the compute and memory needed for one attention layer scale quadratically in sequence length as
\(O(n^2 d)\) and \(O(n^2)\) respectively.

**Inference and the KV cache.** Training and prefill process a whole sequence at once, but
autoregressive decoding generates one token at a time. Naively, generating token `n+1`
would require recomputing the keys and values for all `n` previous tokens from scratch.
Instead, every serving system caches the per-layer, per-head key and value vectors as they
are produced, so each decode step only has to compute the new token's own K/V and attend
over the cached history. This is the **KV cache**, and it's what makes decoding roughly
linear in sequence length instead of quadratic per step. The catch is that the cache itself
is not free: its size is

$$
\begin{aligned}
\text{KV cache size} &= 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot n \cdot B \cdot \text{bytes}\\[4pt]
&= 2 \cdot L \cdot d_{model} \cdot n \cdot B \cdot \text{bytes} \qquad \text{(for plain MHA)}
\end{aligned}
$$

where \(L\) is the number of layers, \(H_{kv}\) the number of KV heads, \(d_{head}\) the
per-head dimension, \(n\) the sequence length, and \(B\) the batch size. That size grows
**linearly with sequence length**, and it has to live in accelerator
memory for the entire duration of a request. Unlike the model weights, which are fixed, the
KV cache competes with weights and activations for the same HBM budget and grows with every
token generated and with every concurrent request served (see
[Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102)
for a detailed treatment of this bottleneck).

**A more realistic estimate.** Modern large models use grouped-query attention (GQA,
[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)), where many query heads share a
much smaller number of key/value heads, precisely to shrink the KV cache. Take a backbone
with `128` layers, `d_model = 16384`, `128` query heads (`head_dim = 128`), but only `8`
KV heads. A 16:1 GQA ratio similar to Llama 3's largest models served in bf16:

$$
\begin{aligned}
\text{KV cache per token per layer} &= 2 \cdot H_{kv} \cdot d_{head} \cdot \text{bytes} \\
&= 2 \cdot 8 \cdot 128 \cdot 2\ \text{bytes} = 4\ \text{KiB}\\[4pt]
\text{KV cache per token (all 128 layers)} &= 4\ \text{KiB} \times 128 = 512\ \text{KiB}
\end{aligned}
$$

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
every concurrent request a server handles. A handful of 128K-context requests can
already outweigh the memory the weights themselves occupy.

**PagedAttention** ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180), the algorithm
behind vLLM) targets exactly this allocation problem. A naive serving system reserves one
contiguous chunk of memory per request sized for the maximum sequence length it might
ever reach, up front. So, a request that ends early ties up memory it never
uses, and that memory is wasted for the request's entire lifetime. PagedAttention instead
allocates the KV cache in small fixed-size blocks on demand, one block at a time as the
sequence actually grows, and uses a per-sequence block table to keep track of which
physical blocks belong to which sequence. Nothing is reserved ahead of time, so no memory
is wasted on requests that finish early, which lets the server pack many more sequences
into a batch. It does *not* change the fact that total cache size still grows
linearly with sequence length; it just removes the wasted upfront reservation.

**FlashAttention and its limits.** A lot of engineering effort has gone into making the
\(O(n^2)\) compute cheaper to execute rather than avoiding it. FlashAttention
([Dao et al., 2022](https://arxiv.org/abs/2205.14135); FlashAttention-2,
[Dao, 2023](https://arxiv.org/abs/2307.08691)) never materializes the full \(n \times n\)
attention matrix in slow HBM. Instead it tiles \(Q\), \(K\), and \(V\) into blocks that fit in
fast on-chip SRAM, computes attention block-by-block, and maintains a running (online)
softmax so the final result is exact despite never seeing the whole score matrix at once.
This is an IO-aware optimization: it doesn't change the \(O(n^2 d)\) FLOP count, but it
removes the \(O(n^2)\) memory traffic to HBM that dominated wall-clock time, giving large
real-world speedups and enabling longer context windows during training and prefill. What
FlashAttention does **not** do is shrink the KV cache. The cache is a decode-time,
per-request storage cost. It exists regardless of how efficiently a single attention
matmul is computed, so even with FlashAttention the memory needed to serve long sequences
still grows linearly and unboundedly with sequence length. Solving *that* problem requires
rethinking the recurrence itself, which is where linear attention comes in.

## Linear Attention

The quadratic cost of standard attention and the linearly-growing KV cache both trace
back to the same culprit: the softmax. Softmax couples every query to every key through a
nonlinear normalization, so there's no way to reorder the computation to avoid forming the
full \(n \times n\) score matrix. Linear attention
([Katharopoulos et al., 2020](https://arxiv.org/abs/2006.16236)) removes that nonlinearity
and, with it, the quadratic cost.

**Dropping the softmax.** Write standard attention's un-normalized numerator for query \(i\)
as a sum over all keys/values:

$$
o_i = \sum_j \mathrm{sim}(q_i, k_j)\, v_j
$$

where \(\mathrm{sim}(q_i, k_j) = \exp\!\left(q_i \cdot k_j / \sqrt{d}\right)\) in standard
attention. Linear attention replaces this similarity with a kernel
\(\mathrm{sim}(q, k) = \phi(q) \cdot \phi(k)\) for some (elementwise, positive) feature map
\(\phi\). The original paper uses \(\phi(x) = \mathrm{elu}(x) + 1\), and later work often
just uses \(\phi(x) = x\) or a simple nonlinearity. The key property is that this similarity
function *factorizes*, so it can be pulled apart:

$$
o_i = \sum_j \big(\phi(q_i) \cdot \phi(k_j)\big)\, v_j = \phi(q_i) \cdot \sum_j \phi(k_j)\, v_j^\top
$$

*For the rest of this article, we'll assume \(\phi(x) = x\) to keep the equations
uncluttered.*

**Why this matters: matmul associativity.** In matrix form, standard attention computes
\((QK^\top)V\). Form the \(n \times n\) matrix \(QK^\top\) first, then multiply by \(V\). Because
\(QK^\top\) is just an ordinary matrix product with no softmax in the way, matrix
multiplication is associative and we're free to instead compute
\(Q\big(K^\top V\big)\). The second grouping never materializes an
\(n \times n\) matrix at all: \(K^\top V\) is a \(d \times d\) matrix (summed over all \(n\)
tokens), and \(Q\) times that is \(O(n d^2)\), **linear** in sequence length instead of
quadratic.

**The recurrent, constant-memory form.** Let's take a closer look at what linear attention
is doing by looking at how we process tokens one at a time during decode. In the product
\(Q(K^\top V)\), the \(d \times d\) matrix \(S = K^\top V\) is just a sum of rank-1 outer products,
\(S = \sum_j k_j\, v_j^\top\), which means it can be built up one token at a time:

$$
\begin{aligned}
S_t &= S_{t-1} + k_t\, v_t^\top \qquad \text{(state update, }S\text{ is }d \times d\text{)}\\[4pt]
o_t &= q_t^\top S_t \qquad\qquad\quad\ \text{(readout)}
\end{aligned}
$$

This is exactly the recurrence of an RNN, which is why the paper is titled "Transformers
are RNNs". A linear attention layer can run autoregressively by carrying a single fixed-size
state matrix \(S\) forward, rather than a cache of every past key and value. \(S\) has a fixed
\(d \times d\) shape no matter how long the sequence gets, so decoding cost per step is
constant and memory for the recurrent state doesn't grow with sequence length at all.

Here's that recurrence running step by step on a toy example (\(d = 3\), 4 tokens):

<div class="lka-anim" id="lka-anim">
  <div class="lka-row">
    <div class="lka-tokens" id="lka-tokens"></div>
  </div>

  <div class="lka-stage">
    <div class="lka-col">
      <div class="lka-label">\(k_t\)</div>
      <div class="lka-vec lka-vcol lka-role-k" id="lka-k"></div>
    </div>
    <div class="lka-op">⊗</div>
    <div class="lka-col">
      <div class="lka-label">\(v_t^\top\)</div>
      <div class="lka-vec lka-vrow lka-role-v" id="lka-v"></div>
    </div>
    <div class="lka-op">→</div>
    <div class="lka-col">
      <div class="lka-label">\(k_t v_t^\top\)</div>
      <div class="lka-grid" id="lka-outer"></div>
    </div>
    <div class="lka-op lka-plus">+</div>
    <div class="lka-col">
      <div class="lka-label">\(S_{t-1}\)</div>
      <div class="lka-grid" id="lka-prev"></div>
    </div>
    <div class="lka-op">=</div>
    <div class="lka-col">
      <div class="lka-label">\(S_t\)</div>
      <div class="lka-grid" id="lka-state"></div>
    </div>
  </div>

  <div class="lka-readout">
    <span class="lka-label">readout&nbsp;</span>
    <span>\(q_t^\top = \)</span>
    <span class="lka-vec lka-vrow lka-role-q" id="lka-q"></span>
    <span>\(\ o_t = q_t^\top S_t = \)</span>
    <span class="lka-vec lka-vrow lka-role-o" id="lka-o"></span>
  </div>

  <div class="lka-controls">
    <button id="lka-step">Step</button>
    <button id="lka-play">Play</button>
    <button id="lka-reset">Reset</button>
    <span class="lka-caption" id="lka-caption">Click Step to process the first token.</span>
  </div>
</div>

<style>
.lka-anim {
  --lka-k:       #2a78d6;
  --lka-v:       #eb6834;
  --lka-outer:   #1baf7a;
  --lka-s:       #4a3aa7;
  --lka-q:       #e87ba4;
  --lka-o:       #008300;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--entry);
  padding: 20px;
  margin: 24px 0;
  font-size: 0.9rem;
  overflow-x: auto;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .lka-anim {
    --lka-k:     #3987e5;
    --lka-v:     #d95926;
    --lka-outer: #199e70;
    --lka-s:     #9085e9;
    --lka-q:     #d55181;
    --lka-o:     #0ca30c;
  }
}
:root[data-theme="dark"] .lka-anim {
  --lka-k:     #3987e5;
  --lka-v:     #d95926;
  --lka-outer: #199e70;
  --lka-s:     #9085e9;
  --lka-q:     #d55181;
  --lka-o:     #0ca30c;
}
.lka-tokens { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.lka-token {
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--secondary);
  font-family: monospace;
  transition: background 0.2s, color 0.2s, border-color 0.2s;
}
.lka-token.lka-active {
  background: var(--lka-k);
  color: #fff;
  border-color: var(--lka-k);
}
.lka-token.lka-done {
  color: var(--content);
  border-color: var(--tertiary);
}
.lka-stage {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: center;
  min-width: 560px;
}
.lka-col { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.lka-label { color: var(--secondary); font-size: 0.8rem; }
.lka-op { font-size: 1.3rem; color: var(--secondary); }
.lka-vec { display: flex; gap: 4px; }
.lka-vcol { flex-direction: column; }
.lka-cell {
  width: 40px; height: 40px;
  display: flex; align-items: center; justify-content: center;
  font-family: monospace; font-size: 0.8rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--code-bg);
  color: var(--content);
}
.lka-grid {
  display: grid;
  grid-template-columns: repeat(3, 40px);
  grid-template-rows: repeat(3, 40px);
  gap: 3px;
}
.lka-grid .lka-cell { transition: background-color 0.4s, color 0.4s, border-color 0.4s; }
.lka-role-k .lka-cell { border-color: var(--lka-k); color: var(--lka-k); font-weight: 600; }
.lka-role-v .lka-cell { border-color: var(--lka-v); color: var(--lka-v); font-weight: 600; }
.lka-role-q .lka-cell { border-color: var(--lka-q); color: var(--lka-q); font-weight: 600; }
.lka-role-o .lka-cell { border-color: var(--lka-o); color: var(--lka-o); font-weight: 600; }
.lka-readout {
  margin-top: 18px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.lka-controls {
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.lka-controls button {
  border: 1px solid var(--border);
  background: var(--code-bg);
  color: var(--content);
  border-radius: 6px;
  padding: 6px 14px;
  cursor: pointer;
  font-size: 0.85rem;
}
.lka-controls button:hover { border-color: var(--lka-k); }
.lka-caption { color: var(--secondary); font-size: 0.82rem; }
</style>

<script>
(function () {
  const K = [[1, 0.3, 0], [0.2, 1, 0.4], [0, 0.5, 1], [0.6, 0.1, 0.2]];
  const V = [[0.8, 0, 0.2], [0.1, 0.9, 0], [0.3, 0.2, 0.7], [0.5, 0.4, 0.1]];
  const Q = [[0.7, 0.5, 0.9], [0.4, 0.8, 0.2], [0.9, 0.1, 0.5], [0.3, 0.6, 0.7]];
  const n = K.length, d = 3;

  let t = 0;
  let S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let playing = false, timer = null;

  const el = (id) => document.getElementById(id);
  const fmt = (x) => x.toFixed(2);

  function renderTokens() {
    const c = el('lka-tokens');
    c.innerHTML = '';
    for (let i = 0; i < n; i++) {
      const s = document.createElement('span');
      s.className = 'lka-token' + (i === t ? ' lka-active' : i < t ? ' lka-done' : '');
      s.textContent = 'token ' + (i + 1);
      c.appendChild(s);
    }
  }

  function renderVec(id, vec, dir) {
    const c = el(id);
    c.classList.toggle('lka-vcol', dir === 'col');
    c.classList.toggle('lka-vrow', dir !== 'col');
    c.innerHTML = '';
    vec.forEach((x) => {
      const cell = document.createElement('div');
      cell.className = 'lka-cell';
      cell.textContent = fmt(x);
      c.appendChild(cell);
    });
  }

  function renderGrid(id, mat, hueVar) {
    const c = el(id);
    c.innerHTML = '';
    for (let r = 0; r < d; r++) {
      for (let col = 0; col < d; col++) {
        const cell = document.createElement('div');
        cell.className = 'lka-cell';
        const val = mat[r][col];
        cell.textContent = fmt(val);
        const alpha = Math.min(Math.abs(val) / 2.5, 1);
        cell.style.background = `color-mix(in srgb, var(${hueVar}) ${Math.round(alpha * 55)}%, var(--code-bg))`;
        cell.style.borderColor = `color-mix(in srgb, var(${hueVar}) ${Math.round(alpha * 70 + 15)}%, var(--border))`;
        c.appendChild(cell);
      }
    }
  }

  function dot(vec, mat) {
    const out = [0, 0, 0];
    for (let col = 0; col < d; col++) {
      let s = 0;
      for (let r = 0; r < d; r++) s += vec[r] * mat[r][col];
      out[col] = s;
    }
    return out;
  }

  function render() {
    renderTokens();
    if (t < n) {
      renderVec('lka-k', K[t], 'col');
      renderVec('lka-v', V[t], 'row');
      const outerMat = [];
      for (let r = 0; r < d; r++) {
        const row = [];
        for (let c = 0; c < d; c++) row.push(K[t][r] * V[t][c]);
        outerMat.push(row);
      }
      renderGrid('lka-outer', outerMat, '--lka-outer');
    } else {
      el('lka-k').innerHTML = '';
      el('lka-v').innerHTML = '';
      renderGrid('lka-outer', [[0,0,0],[0,0,0],[0,0,0]], '--lka-outer');
    }
    renderGrid('lka-prev', S.map(row => row.slice()), '--lka-s');
    const Snext = t < n ? addOuter(S, K[t], V[t]) : S;
    renderGrid('lka-state', Snext, '--lka-s');
    const qIdx = Math.min(t, n - 1);
    renderVec('lka-q', Q[qIdx], 'row');
    renderVec('lka-o', dot(Q[qIdx], Snext), 'row');
    el('lka-caption').textContent =
      t >= n
        ? `Done. Processed all ${n} tokens with a fixed ${d}x${d} state, no growing cache.`
        : `Step ${t + 1} of ${n}`;
  }

  function addOuter(mat, k, v) {
    const out = [];
    for (let r = 0; r < d; r++) {
      const row = [];
      for (let c = 0; c < d; c++) row.push(mat[r][c] + k[r] * v[c]);
      out.push(row);
    }
    return out;
  }

  function step() {
    if (t >= n) return;
    S = addOuter(S, K[t], V[t]);
    t += 1;
    render();
    if (t >= n) stop();
  }

  function reset() {
    stop();
    t = 0;
    S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    render();
  }

  function stop() {
    playing = false;
    el('lka-play').textContent = 'Play';
    if (timer) { clearInterval(timer); timer = null; }
  }

  el('lka-step').addEventListener('click', step);
  el('lka-reset').addEventListener('click', reset);
  el('lka-play').addEventListener('click', function () {
    if (playing) { stop(); return; }
    playing = true;
    el('lka-play').textContent = 'Pause';
    timer = setInterval(() => {
      if (t >= n) { stop(); return; }
      step();
    }, 1400);
  });

  render();
})();
</script>


## Intuition: What Linear Attention Actually Does

**\(S\) as the network's short term memory.** Before going further, it's worth being precise about what the recurrence state
\(S\) represents.  \(S = \sum_j k_j v_j^\top\) is a sum of outer
products, one per token seen so far. Multiply a unit vector \(x\) through it and the sum unpacks
into a similarity-weighted combination of every value ever written:

$$
x^\top S = \sum_j (x \cdot k_j)\, v_j^\top
$$

If \(x\) happens to line up closely with some earlier key \(k_j\) (and not with the others),
this returns something close to that key's stored value \(v_j^\top\): a genuine lookup. If
\(x\) is orthogonal to everything written so far, it returns close to zero: nothing is
stored along that direction yet. This is what makes \(S\)
resemble the short term memory of the network. It contains everything the network
remembers so far, all superimposed into one fixed-size matrix.


**Fast weight programming.** The recurrence from the previous section:
\(S_t = S_{t-1} + k_t v_t^\top\), \(o_t = q_t^\top S_t\) - is an instance of an old idea
called *fast weight programming*
([Schmidhuber, 1992](https://ieeexplore.ieee.org/document/6796337); see also
[Schlag, Irie & Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers"
(2021)](https://arxiv.org/abs/2102.11174), which makes the connection to linear attention
explicit). 

A normal linear layer applies a weight matrix that was fixed at training time
by gradient descent. Those are *slow* weights, updated once every training step, the same
for every input at inference. \(S\) is different: it's a weight matrix that gets rewritten
*at every token, during inference itself*. Each step performs an outer-product ("Hebbian")
write \(k_t v_t^\top\) into \(S\), associating key \(k_t\) with value \(v_t\), and then a query
\(q_t\) *reads* from that same matrix via an ordinary matrix-vector product \(q_t^\top S_t\). \(S\) is the fast weight
matrix being written, and \(q_t^\top S_t\) is that fast weight matrix being executed.

**The trade-off: compression.** Standard attention keeps every past key and value around
individually. \(n\) separate vectors, so a query can retrieve whichever one it's most
similar to with no interference from the rest. Linear attention throws that away and keeps
only the running sum \(S = \sum_j k_j v_j^\top\), a single \(d \times d\) matrix whose size
never grows. The KV cache size problem now disappears because
the information that used to occupy \(O(n)\) memory is now being folded into a memory of
fixed capacity, \(O(d^2)\). A
query \(q_t\) reading from \(S_t\) doesn't get back one clean value; it gets a weighted blend
of every value ever written, weighted by how much each past key resembles \(q_t\), including
keys that are only accidentally similar. Early tokens are especially at risk: their
contribution to \(S\) never gets removed or refreshed, it just keeps getting summed over and
diluted by everything written after it. Standard attention's softmax gives it a sharp,
content-addressed lookup over an ever-growing exact memory; linear attention trades that
sharpness for a fixed-size memory that must approximate, and slowly forgets by dilution
rather than by any deliberate mechanism.

That last part is the real problem: vanilla linear attention has no way to *decide* what to
keep and what to overwrite. The next two sections fix that, first by
letting each token's write selectively erase what's already in \(S\) (the delta rule), and
then by adding an explicit forget gate on top of that (the gated delta rule).

## The Delta Update Rule

Vanilla linear attention only ever adds to \(S\): \(S_t = S_{t-1} + k_t v_t^\top\). Nothing is
ever selectively removed, so two tokens that write similar keys interfere in \(S\) forever. The delta rule (DeltaNet,
[Yang, Wang, Zhang, Shen & Kim, 2024](https://arxiv.org/abs/2406.06484), building on
[Schlag, Irie & Schmidhuber, 2021](https://arxiv.org/abs/2102.11174)) fixes this with one
extra step: before writing \(v_t\), read out whatever value \(S_{t-1}\) currently associates
with \(k_t\), and partially update it with the *difference* of new and old value, controlled by parameter \(\beta_t\)

$$
\begin{aligned}
\bar v_t &= S_{t-1}^\top k_t \qquad \text{(look up $k_t$ in the existing memory)}\\[4pt]
S_t &= S_{t-1} + \beta_t\, k_t\, (v_t - \bar v_t)^\top
\end{aligned}
$$

where \(\beta_t \in (0, 1)\) is a per-token, learned scalar controlling how aggressively to
overwrite, typically a sigmoid of a linear projection of the input, so the model decides
token by token how much to trust the new value over what's already stored. There are two
different ways to understand why this works.

**View 1: selectively erase, then write.** Expand the update above:

$$
S_t = S_{t-1} + \beta_t k_t v_t^\top - \beta_t k_t k_t^\top S_{t-1}
    = S_{t-1}\big(I - \beta_t k_t k_t^\top\big) + \beta_t k_t v_t^\top
$$

The term \(I - \beta_t k_t k_t^\top\) acts on \(S_{t-1}\) *before* anything new is added: it
removes a \(\beta_t\)-fraction of whatever \(S_{t-1}\) currently outputs along the direction
\(k_t\), and only then is the fresh association \(\beta_t k_t v_t^\top\) written in. So
instead of the new outer product piling on top of the old one, as in plain linear attention,
the delta rule first *makes room* for it. Write the same key twice with \(\beta_t = 1\) and
the second write fully overwrites the first rather than blending with it. Overwrite instead
of interference is exactly the "forget" operation the
[intuition](#intuition-what-linear-attention-actually-does) section found missing from
vanilla linear attention.

**View 2: one step of online gradient descent.** The same update falls out of a completely
different starting point. Treat \(S\) as the parameters of a tiny linear regressor whose job
is to predict \(v_t\) from \(k_t\), and define a per-token squared-error loss:

$$
\mathcal{L}_t(S) = \tfrac12 \big\lVert S^\top k_t - v_t \big\rVert^2
$$

Its gradient is \(\nabla_S \mathcal{L}_t = k_t (S^\top k_t - v_t)^\top = k_t (\bar v_t -
v_t)^\top\), so one step of gradient descent with learning rate \(\beta_t\) gives

$$
S_t = S_{t-1} - \beta_t \nabla_S \mathcal{L}_t = S_{t-1} + \beta_t\, k_t\, (v_t - \bar v_t)^\top
$$

This is identical to the update above. This is, literally, the classical delta rule / Widrow-Hoff
LMS rule from adaptive filtering: adjust weights in proportion to the *prediction error*
\((v_t - \bar v_t)\), not just the raw correlation between input and target. Seen this way,
vanilla linear attention's Hebbian update \(S_t = S_{t-1} + k_t v_t^\top\) is what falls out
of the same setup with no error-correction term at all: it always writes the full \(v_t\)
regardless of what \(S\) already predicts, so it can never fix a stale or wrong association,
only bury it under new ones. The delta rule instead turns \(S\) into memory that is actually
*trained*, one token at a time, rather than memory that only ever accumulates.

Both views describe the exact same arithmetic; Either way, \(S\) keeps the same fixed \(d \times d\) size and
\(O(d^2)\) per-token cost as plain linear attention during decode, but it can now edit, not just append.
What it still can't do is decide to forget an association wholesale, independent of whether
the current token happens to write a similar key; that's what the explicit gate in the next
section adds.

## The Gated Delta Update Rule

**What the delta rule still can't do.** The delta rule's erase term,
\(I - \beta_t k_t k_t^\top\), only removes information along the direction of the *current*
key \(k_t\). That's precise: it edits exactly the association a new write would otherwise
collide with, but it's also narrow. If the model wants to wipe the slate more broadly (say,
at a topic or document boundary, where everything written so far has simply gone stale,
regardless of whether any of it resembles the current token's key) the delta rule has no way
to express that. It can only ever erase what the current key happens to point at.

**Adding a global forget gate.** Gated DeltaNet
([Yang, Kautz & Hatamizadeh, 2024](https://arxiv.org/abs/2412.06464)) fixes this by borrowing
the other half of the story from gated linear attention and Mamba2
([Dao & Gu, 2024](https://arxiv.org/abs/2405.21060)): a data-dependent scalar decay
\(\alpha_t \in (0, 1)\), produced by the model from the current token, that uniformly shrinks
*all* of \(S_{t-1}\), every stored association, not just the ones near \(k_t\), before the
delta-rule write happens:

$$
S_t = S_{t-1}\big(\alpha_t (I - \beta_t k_t k_t^\top)\big) + \beta_t k_t v_t^\top
$$

\(\alpha_t\) and \(\beta_t\) are two independent, learned, per-token gates with different
jobs. \(\alpha_t\) is content-*independent* forgetting: it can decay the entire state toward
zero regardless of what's currently being written, giving the model a cheap way to say "start
fresh" or gradually fade out old context. \(\beta_t\) is content-*addressed* editing: it
overwrites whatever is specifically associated with \(k_t\), leaving everything else alone.
Vanilla linear attention has neither: it can only append. The plain delta rule has the
second but not the first: it can correct, but can't wholesale forget. Gated DeltaNet has both.

## Putting It Together: Gated DeltaNet

Everything so far has been about a single recurrence, \(S_t = S_{t-1}\big(\alpha_t(I -
\beta_t k_t k_t^\top)\big) + \beta_t k_t v_t^\top\), acting on one head's worth of keys,
values, and queries. Turning that into an actual layer you can drop into a Transformer needs
a few more pieces, all fairly standard in this line of work.

<div class="gdn-arch">
  <div class="gdn-title">Gated DeltaNet layer &mdash; one head, one token</div>

  <div class="gdn-box gdn-x">\(x_t\)</div>
  <div class="gdn-arrow">projections</div>

  <div class="gdn-row">
    <div class="gdn-box gdn-q">\(q_t\)</div>
    <div class="gdn-box gdn-k">\(k_t\)</div>
    <div class="gdn-box gdn-v">\(v_t\)</div>
    <div class="gdn-gaterow">
      <div class="gdn-box gdn-alpha">\(\alpha_t\)</div>
      <div class="gdn-box gdn-beta">\(\beta_t\)</div>
    </div>
  </div>
  <div class="gdn-caption">short causal conv on \(q,k,v\) &nbsp;&middot;&nbsp; \(L_2\)-normalize \(q_t, k_t\) &nbsp;&middot;&nbsp; \(\alpha_t,\beta_t\) are scalar gates from \(x_t\)</div>

  <div class="gdn-arrow">↓</div>

  <div class="gdn-recurrence">
    <div class="gdn-state-row">
      <div class="gdn-box gdn-s">\(S_{t-1}\)</div>
      <div class="gdn-update">
        \(S_t = S_{t-1}\big(\alpha_t(I - \beta_t k_t k_t^\top)\big) + \beta_t k_t v_t^\top\)
      </div>
      <div class="gdn-box gdn-s">\(S_t\)</div>
    </div>
    <div class="gdn-loopback">↺ carried forward as \(S_{t-1}\) for the next token</div>
  </div>

  <div class="gdn-arrow">readout \(o_t = q_t^\top S_t\)</div>
  <div class="gdn-box gdn-o">\(o_t\)</div>

  <div class="gdn-arrow">× SiLU output gate\((x_t)\)</div>
  <div class="gdn-arrow">concat \(H\) heads → output projection</div>

  <div class="gdn-box gdn-y">\(y_t\)</div>
  <div class="gdn-arrow gdn-residual">+ \(x_t\) (residual)</div>
</div>

<style>
.gdn-arch {
  --gdn-x:     #898781;
  --gdn-q:     #e87ba4;
  --gdn-k:     #2a78d6;
  --gdn-v:     #eb6834;
  --gdn-alpha: #eda100;
  --gdn-beta:  #e34948;
  --gdn-s:     #4a3aa7;
  --gdn-o:     #008300;
  --gdn-y:     #898781;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--entry);
  padding: 24px 20px;
  margin: 24px 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  font-size: 0.88rem;
  overflow-x: auto;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .gdn-arch {
    --gdn-q:     #d55181;
    --gdn-k:     #3987e5;
    --gdn-v:     #d95926;
    --gdn-alpha: #c98500;
    --gdn-beta:  #e66767;
    --gdn-s:     #9085e9;
    --gdn-o:     #0ca30c;
  }
}
:root[data-theme="dark"] .gdn-arch {
  --gdn-q:     #d55181;
  --gdn-k:     #3987e5;
  --gdn-v:     #d95926;
  --gdn-alpha: #c98500;
  --gdn-beta:  #e66767;
  --gdn-s:     #9085e9;
  --gdn-o:     #0ca30c;
}
.gdn-title {
  color: var(--secondary);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 10px;
}
.gdn-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  flex-wrap: wrap;
}
.gdn-gaterow {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: 18px;
  padding-left: 18px;
  border-left: 1px dashed var(--border);
}
.gdn-caption {
  color: var(--secondary);
  font-size: 0.75rem;
  text-align: center;
  max-width: 640px;
  margin: 4px 0 2px;
}
.gdn-box {
  padding: 8px 14px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--code-bg);
  color: var(--content);
  font-family: monospace;
  white-space: nowrap;
}
.gdn-x, .gdn-y { border-color: var(--tertiary); }
.gdn-q     { border-color: var(--gdn-q); color: var(--gdn-q); font-weight: 600; }
.gdn-k     { border-color: var(--gdn-k); color: var(--gdn-k); font-weight: 600; }
.gdn-v     { border-color: var(--gdn-v); color: var(--gdn-v); font-weight: 600; }
.gdn-alpha { border-color: var(--gdn-alpha); color: var(--gdn-alpha); font-weight: 600; }
.gdn-beta  { border-color: var(--gdn-beta); color: var(--gdn-beta); font-weight: 600; }
.gdn-s     { border-color: var(--gdn-s); color: var(--gdn-s); font-weight: 600; }
.gdn-o     { border-color: var(--gdn-o); color: var(--gdn-o); font-weight: 600; }
.gdn-arrow {
  color: var(--secondary);
  font-size: 0.78rem;
  padding: 2px 0;
}
.gdn-residual { margin-top: 2px; }
.gdn-recurrence {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  margin: 6px 0;
  padding: 14px 18px;
  border: 1px dashed var(--gdn-s);
  border-radius: 8px;
}
.gdn-state-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}
.gdn-update {
  font-size: 0.82rem;
  color: var(--content);
  text-align: center;
}
.gdn-loopback {
  color: var(--gdn-s);
  font-size: 0.75rem;
}
</style>


**Multiple heads, each with its own memory.** Just like standard multi-head attention,
Gated DeltaNet splits \(q_t, k_t, v_t\) into \(H\) heads of dimension \(d_{head}\) and runs
the recurrence independently per head, each maintaining its own \(d_{head} \times d_{head}\)
state matrix \(S^{(h)}\). Smaller, more numerous memories turn out to matter for
expressiveness: a single \(d \times d\) matrix has only so much capacity before superposition
starts to hurt (recall the [compression trade-off](#intuition-what-linear-attention-actually-does)
from earlier), and splitting into heads is the same fix multi-head attention already uses for
a different reason: many smaller, independent subspaces instead of one large shared one.

**A short convolution before the recurrence.** Before \(q_t, k_t, v_t\) enter the recurrence,
Gated DeltaNet (following GLA, [Yang, Wang, Shen, Panda & Kim,
2023](https://arxiv.org/abs/2312.06635), and Mamba) passes them through a short causal
depthwise convolution, a kernel width of about 4 tokens. The recurrence itself has no
built-in notion of local order beyond what \(S\) accumulates; a few positions of local
convolutional context ahead of time gives each token's key and value a cheap, local n-gram
signal to work with, which empirically matters a lot for tasks like copying and induction
that hinge on nearby token patterns.

**Normalization for a stable recurrence.** \(q_t\) and \(k_t\) are \(L_2\)-normalized before
use. This isn't cosmetic: the erase term \(I - \beta_t k_t k_t^\top\) is only guaranteed to be
a contraction (i.e., not blow up the state over thousands of steps) when \(k_t\) has unit
norm and \(\beta_t \in (0, 1)\). Without normalization, a recurrence run over a long sequence
has every opportunity to drift; keeping \(k_t\) on the unit sphere is what keeps \(S\) from
exploding or decaying pathologically over long contexts.

**An output gate.** After the readout \(o_t = q_t^\top S_t\), the layer applies one more
gate: an elementwise SiLU/sigmoid gate computed from the input, multiplied into \(o_t\)
before the final output projection, the same device GLA and Mamba2 use. It lets the layer
suppress its output on tokens where reading from memory isn't useful, independent of what
\(\alpha_t\) and \(\beta_t\) decided to do to \(S\) itself.

**Where this lands.** Gated DeltaNet consistently beats both of the mechanisms it combines: Mamba2 (gating alone) and plain DeltaNet
(the delta rule alone), on language modeling
perplexity, in-context retrieval, length extrapolation, and long-context benchmarks. It still
inherits linear attention's core trade-off from the [intuition](#intuition-what-linear-attention-actually-does)
section: a fixed-size state is a compressed, lossy summary of the past, and no amount of
gating or delta-rule editing changes that asymptotically. What gating and the delta rule buy
back is *control* over that compression: deciding, every token, what's worth keeping, which
in practice closes most of the gap to full attention while keeping the KV-cache-free,
constant-memory decoding that motivated this whole detour away from softmax attention in the
first place.

## What's Next

Everything above describes Gated DeltaNet token by token, the way it actually runs during decoding. Computing \(S_t\) requires \(S_{t-1}\), which requires
\(S_{t-2}\), and so on. Training the model sequentially on each 
individual token is painfully slow. Both DeltaNet and Gated
DeltaNet get around this with a chunkwise-parallel form: split the sequence into fixed-size chunks, and within each chunk, use a matrix identity (a "WY representation") to express the chunk's entire sequence of erase-and-write
updates as a handful of dense matmuls instead of a token-by-token loop, leaving only the state handoff *between* chunks sequential.

The chunkwise algorithm is a pretty neat idea: it turns an inherently sequential recurrence
into a handful of matmuls without changing what it computes, which is what makes Gated
DeltaNet practical to train on GPUs and TPUs in the first place. It's also hard to grasp by
just reading existing implementations, since the kernels are written for throughput, not for
following the derivation step by step. The next post will break down how the WY
representation makes this possible, one step at a time.
