+++
title = "Inside mini-nccl: Building a PyTorch Distributed Backend from Scratch"
date = 2026-04-14
draft = false
tags = ["pytorch", "distributed", "cuda"]
description = "A deep dive into how collective communication operations work, illustrated by building a custom torch.distributed backend from scratch."
+++

I work on making it easier to program TPUs with PyTorch during my day job. Even though
chip providers pack huge HBM capacity into today's chips, performant distributed programs
are still necessary to train and run frontier large language models. Since I work
mostly on TPUs, I was curious what abstractions other hardwares provide to program their
distributed backends. What better way to find out than building one yourself.

I created [mini-nccl](https://github.com/bhavya01/mini-nccl), a from-scratch
implementation of a PyTorch distributed backend for GPUs attached to a single node. The
goal is not to build something production-ready, but to make the concepts concrete: what
a collective *is*, how the `torch.distributed` plugin interface works, and how
peer-to-peer GPU memory transfers actually happen. By the end you should be able to read
a `torch.distributed.all_reduce` call and trace exactly what happens between Python and
the GPU silicon. Each collective has its own `.cu` file, and the examples directory has
a runnable script for every operation — if you have two or more GPUs, you can run them
today.

---

## The torch.distributed Abstraction Layer

Before looking at any collective, it helps to understand how PyTorch structures the
distributed stack.

```
  ┌─────────────────────────────────────────────────┐
  │            User code / DDP / FSDP               │
  │         dist.all_reduce(tensor, ...)            │
  └───────────────────┬─────────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────────┐
  │            torch.distributed (Python)           │
  │   Dispatches to the active ProcessGroup backend │
  └───────────────────┬─────────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────────┐
  │            ProcessGroup Backend (C++)           │
  │   Interface: allreduce(), broadcast(), ...      │
  │   mini-nccl implements this interface           │
  └───────────────────┬─────────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────────┐
  │            CUDA Runtime / Hardware              │
  │   cudaMemcpyPeer, cudaIpcGetMemHandle, ...      │
  └─────────────────────────────────────────────────┘
```

Three key abstractions:

**`ProcessGroup` / `Backend`** — The C++ base class that every backend must implement.
It declares virtual methods for each collective: `allreduce`, `broadcast`, `gather`,
and so on. `MiniNcclBackend` in this project subclasses `c10d::Backend` and overrides
each of these.

**`Work`** — The handle returned by every collective call. It lets the caller
ask whether the operation is done (`isCompleted()`), block until it finishes (`wait()`),
or attach a callback via a `Future`. This is how async overlap between compute and
communication is expressed.

**`Store`** — A simple key-value store (think: a distributed dict) shared by all ranks
in a process group. It's used for rendezvous and coordination during collectives.
In the examples, a `FileStore` backed by a temp file plays this role.

---

## Plugging in the Backend

Mini-nccl registers itself with PyTorch using a C++ constructor attribute — a mechanism
that runs a function the moment the shared library is loaded into the process.

```cpp
// backend.h
static void RegisterMiniNcclBackend() __attribute__((constructor))
{
    py::object module = py::module::import("torch.distributed");
    py::object register_backend = module.attr("Backend").attr("register_backend");
    register_backend("mini_nccl",
                     py::cpp_function(createMiniNcclBackend),
                     py::arg("extended_api") = false,
                     py::arg("devices") = "cuda");
}
```

`__attribute__((constructor))` is a GCC/Clang directive: the annotated function runs
automatically when the `.so` is dlopen'd. So all user code needs is:

```python
import mini_nccl  # this line triggers the library load and registration

dist.init_process_group(backend="mini_nccl", ...)
```

After that, every `dist.all_reduce(...)` call will dispatch to
`MiniNcclBackend::allreduce`.

The factory function `createMiniNcclBackend` receives the `Store`, `rank`, and
`world_size` from PyTorch and constructs the backend:

```cpp
MiniNcclBackend::MiniNcclBackend(int rank, int size,
                                 c10::intrusive_ptr<::c10d::Store> store)
    : Backend(rank, size), store_(std::move(store)) {}
```

The backend holds on to the store and a monotonically increasing `seq_` counter.
Every collective call increments `seq_` and uses it to namespace the store keys it
needs, so concurrent or back-to-back collectives don't collide.

---

## The Store: Coordination Without a Controller

The Store is the backbone of how mini-nccl's collectives coordinate. Its interface is
minimal:

- `store->set(key, bytes)` — publish a value
- `store->get(key)` — retrieve a value (blocks until the key exists)
- `store->wait(keys)` — block until all listed keys are present

There's no master node managing traffic. Each rank publishes what it has (e.g., a
CUDA IPC handle) and then waits for peers to do the same. This produces a
decentralized barrier out of simple key-value operations.

The `seq_` counter makes each collective invocation unique:

```cpp
// Each call gets a fresh sequence number.
seq_.fetch_add(1, std::memory_order_relaxed)

// Keys look like: "ar_42_handle_0", "ar_42_handle_1", ...
const std::string pfx = "ar_" + std::to_string(seq) + "_";
store->set(pfx + "handle_" + std::to_string(rank), my_bytes);
```

If two all-reduces happen concurrently, they get different `seq` values and therefore
different key prefixes. No collision.

---

## CUDA IPC: How GPUs Share Memory Without Copying Through the Host

Mini-nccl's collectives all come down to one question: how does rank A read rank B's
GPU tensor without going through CPU memory?

The answer is **CUDA IPC** (Inter-Process Communication). Here's the mechanism in steps:

**Step 1: Get a handle.**
Any GPU buffer allocated with `cudaMalloc` can be converted into a serializable token
called a `cudaIpcMemHandle_t`. This handle encodes everything needed to open a
mapping to that buffer from another process.

```cpp
HandlePayload my_payload;
my_payload.device_id = my_device;
cudaIpcGetMemHandle(&my_payload.ipc_handle, tensor.data_ptr());
```

**Step 2: Share the handle via the store.**
The `HandlePayload` struct (IPC handle + device ID) is serialized to bytes and written
to the store under a unique key.

```cpp
std::vector<uint8_t> my_bytes(sizeof(HandlePayload));
std::memcpy(my_bytes.data(), &my_payload, sizeof(HandlePayload));
store->set(pfx + "handle_" + std::to_string(rank), my_bytes);
```

**Step 3: Open the handle from a peer process.**
On the other side, a peer reads the bytes from the store, deserializes the payload,
and calls `cudaIpcOpenMemHandle` to get a raw pointer to the remote buffer.

```cpp
void *remote_ptr = nullptr;
cudaIpcOpenMemHandle(&remote_ptr,
                     remote_payload.ipc_handle,
                     cudaIpcMemLazyEnablePeerAccess);
```

**Step 4: Copy peer-to-peer.**
`cudaMemcpyPeer` transfers data between two GPUs directly — no bounce through host
RAM — as long as the GPUs are in a rack topology that supports peer access (NVLink,
PCIe peer-to-peer, etc.).

```cpp
cudaMemcpyPeer(dst_ptr, dst_device, remote_ptr, src_device, nbytes);
```

**Step 5: Close the handle.**
IPC handles must be closed after use. Critically, they must be opened *and* closed
on the device the handle was created on.

```cpp
cudaIpcCloseMemHandle(remote_ptr);
```

One important detail: `cudaIpcGetMemHandle` requires a **cudaMalloc base pointer**.
PyTorch's caching allocator sub-allocates multiple tensors within a single 2 MB block,
so multiple tensors can share the same base pointer and produce the same handle.
For collectives like scatter and all-to-all that create per-rank chunks, mini-nccl
bypasses PyTorch's allocator and calls raw `cudaMalloc` for each chunk to guarantee
distinct handles.

---

## The Three-Phase Protocol

Every collective in mini-nccl follows the same three-phase structure. Understanding this
pattern once means you understand all nine collectives:

```
  ┌──────────────────────────────────────────────────────┐
  │  Phase 1 — Publish                                   │
  │  Each rank serializes its CUDA IPC handle(s)         │
  │  and writes them to the store.                       │
  │  All ranks wait until every handle is present.       │
  └──────────────────────────┬───────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────┐
  │  Phase 2 — Transfer                                  │
  │  Each rank opens the peer handles it needs,          │
  │  pulls data via cudaMemcpyPeer,                      │
  │  and applies any reduction locally.                  │
  └──────────────────────────┬───────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────┐
  │  Phase 3 — Completion Barrier                        │
  │  Each rank writes a "done" signal to the store.      │
  │  All ranks wait until every "done" is present.       │
  │  This ensures no rank frees its memory before        │
  │  every peer has finished reading from it.            │
  └──────────────────────────────────────────────────────┘
```

The completion barrier in Phase 3 is subtle but critical. If rank 0 finishes its
all-reduce and immediately frees or reuses the tensor, but rank 1 still has rank 0's
IPC handle open and is in the middle of copying from it, the result is a
use-after-free on the GPU. Phase 3 prevents this by making every rank wait for the
group's collective acknowledgment before proceeding.

---

## The Collectives

### Broadcast

**What it does:** The root rank sends its tensor to all other ranks. Every rank ends
up holding a copy of the root's data.

```
  Before:                        After:
  Rank 0 (root): [42, 42]        Rank 0: [42, 42]
  Rank 1:        [ 0,  0]  ────► Rank 1: [42, 42]
  Rank 2:        [ 0,  0]        Rank 2: [42, 42]
```

**Implementation:** Only the root publishes an IPC handle (Phase 1). Non-root ranks
wait for it, open it, and copy the data into their local tensor (Phase 2). Phase 3
ensures root doesn't discard the tensor until all readers are done.

```cpp
// Phase 1: root publishes its handle; all ranks wait for it.
if (rank == root) {
    cudaIpcGetMemHandle(&payload.ipc_handle, tensor.data_ptr());
    store->set(pfx + "handle", bytes);
}
store->wait({pfx + "handle"});

// Phase 2: non-root ranks copy root's tensor.
if (rank != root) {
    cudaIpcOpenMemHandle(&remote_ptr, remote_payload.ipc_handle, ...);
    cudaMemcpyPeer(tensor.data_ptr(), my_device,
                   remote_ptr, remote_payload.device_id, nbytes);
    cudaIpcCloseMemHandle(remote_ptr);
}
```

**PyTorch usage:**
```python
tensor = torch.full((8,), 42.0, device="cuda") if rank == 0 \
         else torch.zeros(8, device="cuda")
dist.broadcast(tensor, src=0)
# Every rank now holds [42.0, 42.0, ..., 42.0]
```

---

### Reduce

**What it does:** All ranks contribute their tensor; only the root ends up with the
combined (e.g. summed) result.

```
  Before:                        After:
  Rank 0 (root): [1, 2]         Rank 0 (root): [6, 8]
  Rank 1:        [2, 3]   ────► Rank 1:        [2, 3]  (unchanged)
  Rank 2:        [3, 3]         Rank 2:        [3, 3]  (unchanged)
```

**Implementation:** Every rank publishes its IPC handle (Phase 1). Only the root
performs work in Phase 2: it opens each peer's handle, pulls the data via P2P copy
into a scratch buffer, and reduces it into its own tensor.

```cpp
// Phase 2: only root does the reduction work.
if (rank == root) {
    at::Tensor scratch = at::empty_like(tensor);
    for (int r = 0; r < world_size; ++r) {
        if (r == root) continue;
        cudaIpcOpenMemHandle(&remote_ptr, ...);
        cudaMemcpyPeer(scratch.data_ptr(), my_device, remote_ptr, ...);
        apply_reduce(tensor, scratch, reduce_op);  // tensor += scratch (for SUM)
        cudaIpcCloseMemHandle(remote_ptr);
    }
}
```

Non-root ranks do nothing in Phase 2; they just participate in the Phase 3 barrier
so root knows it's safe to proceed.

---

### All-Reduce

**What it does:** Like reduce, but every rank ends up with the result — not just the root.

```
  Before:                        After:
  Rank 0: [1, 2]                 Rank 0: [6, 8]
  Rank 1: [2, 3]   ──────────►   Rank 1: [6, 8]
  Rank 2: [3, 3]                 Rank 2: [6, 8]
```

This is the workhorse of DDP. After an all-reduce on gradients, every rank holds the
same summed gradient and can run the optimizer in lockstep.

**Implementation:** Every rank publishes its IPC handle (Phase 1). In Phase 2, every
rank opens *every peer's* handle, copies into scratch, and reduces into its own tensor.
The reduction is applied locally on each rank — there's no single "reducer" node.

```cpp
// Phase 2: every rank reduces all peers into itself.
at::Tensor scratch = at::empty_like(tensor);
for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    cudaIpcOpenMemHandle(&remote_ptr, ...);
    cudaMemcpyPeer(scratch.data_ptr(), my_device, remote_ptr, ...);
    apply_reduce(tensor, scratch, reduce_op);
    cudaIpcCloseMemHandle(remote_ptr);
}
if (reduce_op == ReduceOp::AVG)
    tensor.div_(world_size);
```

The redundancy is intentional: on an all-to-all topology, all-reduce is trivially
parallelized because every pair of GPUs can communicate directly. Production NCCL
uses ring or tree algorithms to reduce the total data in flight, but the end result
is the same.

**PyTorch usage:**
```python
tensor = torch.full((8,), float(rank + 1), device="cuda")
dist.all_reduce(tensor)  # default op: SUM
# Every rank now holds the sum of all ranks' tensors.
```

---

### Scatter

**What it does:** The root holds N chunks and sends a different chunk to each rank.

```
  Before:                                  After:
  Rank 0 (root): [[A], [B], [C]]          Rank 0: [A]
  Rank 1:        [  ?  ]           ─────► Rank 1: [B]
  Rank 2:        [  ?  ]                  Rank 2: [C]
```

**Implementation:** The root copies each input chunk into a separate
`cudaMalloc` staging buffer (raw allocation, not through PyTorch's caching
allocator) and publishes a separate IPC handle keyed by destination rank. Each
non-root rank fetches only its own handle and copies its chunk.

```
  Root (rank 0) staging:
  staging[0] ──IPC handle 0──► Rank 0 reads own chunk locally
  staging[1] ──IPC handle 1──► Rank 1 opens and pulls chunk B
  staging[2] ──IPC handle 2──► Rank 2 opens and pulls chunk C
```

The raw `cudaMalloc` is necessary here because each chunk must have a distinct base
pointer for `cudaIpcGetMemHandle` to produce a distinct handle.

---

### Gather

**What it does:** The inverse of scatter. Each rank contributes a tensor; the root
collects all of them.

```
  Before:                            After:
  Rank 0 (root): [A]                Rank 0: [[A], [B], [C]]
  Rank 1:        [B]          ────► Rank 1: [B]  (unchanged)
  Rank 2:        [C]                Rank 2: [C]  (unchanged)
```

**Implementation:** Every rank publishes its IPC handle (Phase 1). Only the root
does work in Phase 2: it opens each peer's handle and copies the data into the
corresponding output slot.

```cpp
// Phase 2: root pulls each rank's input into its output slots.
if (rank == root) {
    cudaMemcpy(outputTensors[root].data_ptr(), inputTensor.data_ptr(), ...); // own copy
    for (int r = 0; r < world_size; ++r) {
        if (r == root) continue;
        cudaIpcOpenMemHandle(&remote_ptr, ...);
        cudaMemcpyPeer(outputTensors[r].data_ptr(), my_device, remote_ptr, ...);
        cudaIpcCloseMemHandle(remote_ptr);
    }
}
```

---

### All-Gather

**What it does:** Every rank contributes a tensor; every rank ends up with the full
concatenated tensor from all ranks.

```
  Before:                            After:
  Rank 0: [A]                       Rank 0: [[A], [B], [C]]
  Rank 1: [B]          ──────────►  Rank 1: [[A], [B], [C]]
  Rank 2: [C]                       Rank 2: [[A], [B], [C]]
```

This is used heavily in FSDP: after sharding parameters across ranks, each rank runs
an all-gather before the forward pass to reconstruct the full layer. After the
backward pass, the reconstructed copy is discarded and only the shard is kept.

**Implementation:** Every rank publishes its IPC handle. In Phase 2, every rank reads
from every peer's handle and writes into the corresponding output slot.

```
  Rank 0 output:  [own A] [pull B from rank 1] [pull C from rank 2]
  Rank 1 output:  [pull A from rank 0] [own B] [pull C from rank 2]
  Rank 2 output:  [pull A from rank 0] [pull B from rank 1] [own C]
```

The own copy uses `cudaMemcpy` (same device); peer copies use `cudaMemcpyPeer`.

---

### Reduce-Scatter

**What it does:** Each rank provides N chunks; rank r ends up with the reduction of
chunk r from all ranks. Reduce-scatter is the complement of all-gather.

```
  Before (each rank has 3 chunks):
  Rank 0: [A0, A1, A2]
  Rank 1: [B0, B1, B2]
  Rank 2: [C0, C1, C2]

  After (rank r holds the reduction of chunk r):
  Rank 0: [A0+B0+C0]
  Rank 1: [A1+B1+C1]
  Rank 2: [A2+B2+C2]
```

The combination **reduce-scatter → all-gather** is equivalent to **all-reduce**. FSDP
exploits this: gradient all-reduce can be decomposed into reduce-scatter (each rank
accumulates its shard) followed by all-gather (each rank reconstructs the full
gradient for the optimizer step).

**Implementation:** Every rank publishes IPC handles for all N of its input chunks
(`handle_rank_chunk`). In Phase 2, rank r reads `handle_r_chunk_r` from every peer —
that is, every peer's chunk at index r — reduces them all, and stores the result in
the output tensor.

```cpp
// Phase 2: rank `rank` reduces chunk `rank` from every peer.
cudaMemcpy(outputTensor.data_ptr(), inputTensors[rank].data_ptr(), ...); // seed from self
at::Tensor scratch = at::empty_like(outputTensor);
for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    // Fetch rank r's chunk at index `rank`.
    auto remote_bytes = store->get(pfx + "handle_" + to_string(r) + "_" + to_string(rank));
    cudaIpcOpenMemHandle(&remote_ptr, ...);
    cudaMemcpyPeer(scratch.data_ptr(), my_device, remote_ptr, ...);
    apply_reduce(outputTensor, scratch, reduce_op);
    cudaIpcCloseMemHandle(remote_ptr);
}
```

---

### All-to-All

**What it does:** A generalization of scatter/gather — every rank sends a different
chunk to every other rank simultaneously.

```
  Before:
  Rank 0 sends: [to_0, to_1, to_2]
  Rank 1 sends: [to_0, to_1, to_2]
  Rank 2 sends: [to_0, to_1, to_2]

  After (rank r receives chunk r from every rank):
  Rank 0 output: [rank0→0, rank1→0, rank2→0]
  Rank 1 output: [rank0→1, rank1→1, rank2→1]
  Rank 2 output: [rank0→2, rank1→2, rank2→2]
```

All-to-all is used in expert parallelism (Mixture-of-Experts models), where tokens
are routed to different expert GPUs, processed, and then routed back to their origin.

**Implementation:** Every rank splits its input into N chunks, copies each into a
raw-`cudaMalloc` staging buffer, and publishes an IPC handle keyed by
`handle_rank_destrank`. In Phase 2, rank s fetches `handle_r_s` from every r —
that is, rank r's chunk destined for rank s — and assembles its output.

```
  Store keys after Phase 1 (4 ranks, for clarity):
  handle_0_0  handle_0_1  handle_0_2  handle_0_3   ← rank 0's chunks for each dest
  handle_1_0  handle_1_1  handle_1_2  handle_1_3   ← rank 1's chunks
  ...

  Phase 2, rank 2:
    reads handle_0_2 → output[0]  (rank 0's chunk for rank 2)
    reads handle_1_2 → output[1]
    copies own chunk 2 locally   → output[2]
    reads handle_3_2 → output[3]
```

---

### Barrier

**What it does:** Synchronizes all ranks. No rank proceeds past the barrier until
every rank has reached it. No data is transferred.

**Implementation:** The barrier is the simplest collective — it's the same pattern
as Phase 3 of every other collective, elevated to a standalone operation:

```cpp
const std::string pfx = "barrier_" + to_string(seq) + "_";
const std::vector<uint8_t> signal = {1};
store->set(pfx + to_string(rank), signal);  // "I'm here"

std::vector<std::string> keys;
for (int r = 0; r < size_; ++r)
    keys.push_back(pfx + to_string(r));
store->wait(keys);  // wait until everyone is here
```

Each rank writes a signal keyed by its rank number, then blocks on `store->wait`
until all signals are present. The store's `wait` is a blocking poll: the calling
thread stalls until all listed keys exist.

---

## Putting It Together: The Full Backend Dispatch

When user code calls `dist.all_reduce(tensor)`, here's the complete call stack:

```
  Python: dist.all_reduce(tensor)
      │
      ▼
  torch/distributed/distributed_c10d.py
      calls: default_pg.allreduce([tensor], opts)
      │
      ▼
  c10d::ProcessGroup::allreduce (C++ virtual dispatch)
      │
      ▼
  MiniNcclBackend::allreduce (our implementation)
      for each tensor:
          mini_nccl::cuda_allreduce(store_, rank_, size_, seq_++, tensor, reduceOp)
      wraps result in MiniNcclWork with a completed Future
      │
      ▼
  cuda_allreduce:
      Phase 1: cudaIpcGetMemHandle → store->set → store->wait
      Phase 2: for each peer: store->get → cudaIpcOpenMemHandle →
               cudaMemcpyPeer → apply_reduce → cudaIpcCloseMemHandle
      Phase 3: store->set(done) → store->wait(all done)
```

The `Work` object returned by `allreduce` has `isCompleted()` returning true
immediately because mini-nccl's collectives are synchronous — they block until done.
A production backend would return a `Work` backed by CUDA events and let the CPU
race ahead while the GPU runs the collective asynchronously.

---

## How Real NCCL Differs

Mini-nccl is deliberately simplified to make the concepts legible. Here's what a
production-grade NCCL does differently:

### Ring and tree algorithms

Mini-nccl's all-reduce has every rank read from every other rank: O(N) IPC opens and
N P2P copies per rank. Real NCCL uses a ring all-reduce where each rank only exchanges
data with its two neighbors, achieving O(1) neighbors per rank with roughly 2(N-1)/N
bandwidth efficiency. Tree reductions are used for small messages where latency matters
more than bandwidth.

### Topology awareness

`nvidia-smi topo -m` reveals that NVLink connections between GPUs are not all equal —
GPUs in the same NVSwitch fabric talk much faster than those bridged via PCIe. NCCL
inspects the topology at init time and builds a communication graph that routes
reductions along high-bandwidth links.

### Kernel fusion and pipelining

Mini-nccl issues separate CUDA API calls for each transfer. NCCL fuses the copy and
reduction into a single custom CUDA kernel, avoids round-trips through the CPU between
each step, and pipelines multiple chunks through the ring simultaneously
(double-buffering) to hide latency.

### CUDA streams and async execution

Every collective in mini-nccl is synchronous on the CPU — `store->wait` blocks the
calling thread. NCCL submits all work to CUDA streams and returns immediately. The CPU
and GPU overlap: while the GPU runs an all-reduce, the CPU is already dispatching the
next batch of forward-pass kernels.

### NVLink / SHARP / InfiniBand

For multi-node clusters, NCCL uses RDMA over InfiniBand for inter-node transfers, and
in some configurations can offload the reduction to in-network compute (NVIDIA SHARP).
Mini-nccl is single-rack, single-hop, CUDA-only.

---

## Benchmarks

I benchmarked three collectives: all-gather, all-reduce and all-to-all against PyTorch's production NCCL backend on a
single node with **2 L4 GPUs** (`world_size=2`), float32 tensors, aggregated over 10
runs of 20 timed iterations each (5 warmup). (If I have spare time, I might try H100 next.) The full tables, including algorithm
bandwidth, live in [`benchmarks/results.md`](https://github.com/bhavya01/mini-nccl)
in the repo.

The headline result is a **fixed latency floor**. Every mini-nccl collective takes
roughly 10 ms regardless of message size until the payload gets large enough to
dominate that overhead. NCCL, by contrast, is sub-millisecond for small messages.
This is the cost of the three-phase protocol: a synchronous store-based barrier plus
CUDA IPC handle open/close on every single call, none of it overlapped with compute.

NCCL latency speedup over mini-nccl, per collective:

| per-rank tensor | all_gather | all_reduce | all_to_all |
|---|---|---|---|
| 1 MiB   | 29.5× | 36.7× | 64.8× |
| 16 MiB  | 2.0×  | 2.8×  | 5.8×  |
| 64 MiB  | 0.9×  | 1.3×  | 2.1×  |
| 256 MiB | 0.8×  | 1.0×  | 1.4×  |

Two things stand out. First, the gap collapses as tensors grow — once a single
transfer takes tens of milliseconds, the fixed 10 ms barrier stops mattering. For
all_gather and all_reduce at 256 MiB, mini-nccl actually *edges out* NCCL (speedup
< 1.0×), plateauing around 4.5 GB/s of algorithm bandwidth versus NCCL's ~3.3–4.3
GB/s.


Second, all_to_all stays NCCL-favored at every size (1.4× even at 256 MiB).
Unlike all-reduce, all-to-all doesn't degenerate at N=2 the same way, and
mini-nccl's bandwidth tops out around 3.2 GB/s against NCCL's 4.4 GB/s — the
clearest signal here of what fused, pipelined kernels buy you.

The takeaway: mini-nccl's headline weakness is **per-call overhead**, not raw
copy throughput. Closing the small-message gap means attacking the synchronous
barrier and IPC churn — persistent handles, CUDA-stream-backed async `Work`
objects — long before topology-aware ring algorithms would pay off.

---

## Conclusion

Starting from `dist.all_reduce(tensor)` and ending at `cudaMemcpyPeer`, the path goes
through:

1. **Backend registration** — a shared library constructor function plugs the backend
   into PyTorch's dispatch mechanism.
2. **The `ProcessGroup` interface** — a small set of virtual methods that abstract over
   collective semantics.
3. **The Store** — a decentralized key-value rendezvous layer that lets ranks coordinate
   without a central controller.
4. **CUDA IPC** — serializable handles that let processes open GPU memory mappings
   across process boundaries.
5. **The three-phase protocol** — publish, transfer, barrier — repeated for every
   collective.

None of these pieces are mysterious on their own. Together they form the plumbing that
lets a gradient computed on GPU 127 become part of the update applied on GPU 0 in a
single synchronous step.

The benchmarks below show the cost of that simplicity: mini-nccl pays a flat
per-collective overhead that makes it 30–65× slower than NCCL on small tensors, yet
its raw peer-to-peer bandwidth stays competitive on large ones.

