# LLM Inference Optimization from Scratch

A ground-up exploration of the optimization techniques that power production LLM inference systems, implemented from first principles in PyTorch and then deployed to real hardware to verify that the concepts hold under production load.

---

## Philosophy

Most LLM inference content either stops at theory or jumps straight to "just use vLLM". This project takes a third path: **understand every technique by building it from scratch, then deploy it and measure the results against real-world targets**.

The notebook (`LLM_inference_from_scratch.ipynb`) builds each optimization layer by layer, from a naive attention loop up to paged memory management. The deployment project (`llm-qwen-deployment/`) takes those same techniques such as KV caching, dynamic batching, INT4 quantization and wraps them in a production FastAPI server, and runs a 1-hour load test against a real throughput target (>10,000 requests/hr).

The gap between "I understand how batching works" and "I can build a server that serves 10,000+ requests in an hour" is what this project tries to close.

---

## Repository Structure

```
LLM-Inference-Optimisation/
├── LLM_inference_from_scratch.ipynb   # Optimization phases 1–7 (theory + benchmarks)
└── llm-qwen-deployment/               # Production deployment project
    ├── custom_server.py               # FastAPI server: dynamic batching + INT4 quantization
    ├── locustfile.py                  # Load test client (100 users, 2 pods)
    ├── llm_as_eval.py                 # LLM-as-a-judge quality evaluation
    ├── generate_prompts.py            # 1,500 synthetic story prompts
    └── results/                       # All benchmark CSVs and eval scores
```

---

## Optimization Phases (Notebook)

### Phase 1 : KV Caching
**The foundation of efficient autoregressive decoding**

- Implements single-head and multi-layer attention with KV cache support
- Compares naive recomputation vs. cached approach across sequence lengths
- **Key Result:** Up to **10x speedup** for long sequences by avoiding redundant key/value computations

### Phase 2 : Peak GPU Utilization through Batching
**Maximizing hardware throughput**

- Explores batch scaling to saturate GPU compute (targeting peak FLOPS)
- Benchmarks custom models and GPT-2 with varying batch sizes
- **Key Insight:** Batch processing amortizes memory bandwidth costs and dramatically improves tokens/sec

### Phase 3 : Sliding Window Attention
**Memory-bounded long-context handling**

- Implements attention with configurable window sizes (32, 64, 128, 256 tokens)
- Analyzes memory/speed trade-offs as sequence length grows
- **Key Trade-off:** O(window_size) memory for KV cache vs. loss of long-range dependencies

### Phase 4 : Flash Attention
**Kernel-level memory optimization**

- Integrates PyTorch's `scaled_dot_product_attention` with Flash/memory-efficient backends
- Compares FP32 naive attention vs. FP16 fused kernels
- **Hardware Note:** Memory-efficient attention on Turing GPUs (T4); Flash Attention v2 requires Ampere+ (A100, RTX 3090)
- **Why both Phase 3 & 4?** Sliding window caps *context length*; Flash Attention optimizes *computation*  orthogonal optimizations used together in production

### Phase 5 : Naive Paged Attention (vLLM-style)
**Dynamic memory allocation for batched inference**

- Naive, single request implementation to grasp the core ideas like block-based KV, fixed pages. A concurrent query scheduler is planned for a future update.
- Custom block based KV cache with fixed-size pages
- Enables memory sharing and reuse across sequences (prefix sharing, beam search)
- **Key Result:** ~4x memory savings vs. naive per-sequence allocation for variable-length batches

### Phase 6 : Speculative Decoding
**Latency reduction via draft models**

Empirically evaluated and rejected for this for our use case (see [llm-qwen-deployment/README.md](llm-qwen-deployment/README.md) "Why Batching Over Speculative Decoding"). HuggingFace's `generate()` enforces `batch_size=1` for speculative decoding, making it incompatible with batched serving. Dynamic batching with BS=8 delivered a **3.7× latency improvement** over speculative decoding at batch=1.

### Phase 7 : Production Deployment
**End-to-end serving pipeline — complete**

Applied KV caching (Phase 1), dynamic batching (Phase 2), and INT4 quantization to build a FastAPI inference server deployed across 2 RunPod GPU pods. See `llm-qwen-deployment/` for the full implementation and results.

**Production result: 13,558 requests served in one hour — 136% of the 10,000 req/hr target. Both GPUs’ utilization remained constantly above 90%.**

---

## Production Deployment Results

| Stage | Config | Throughput | Median Latency | P95 Latency |
|---|---|---|---|---|
| Smoke test: spec. decoding | BS=1, 10 users | ~324 req/hr | 100 s | 120 s |
| Smoke test: dynamic batching | BS=8, 10 users | ~1,440 req/hr | 27 s | 32 s |
| Production run 1 | BS=8, 20 users, 1 hr | 4,090 req/hr | 37.6 s | 75.9 s |
| **Production run 2** | **BS=64, 100 users, 1 hr** | **13,558 req/hr** | **24.9 s** | **30.0 s** |

Hardware: 2× RTX 2000 Ada (RunPod), model: Qwen/Qwen2.5-1.5B-Instruct (INT4 NF4).

Increasing `MAX_BATCH_SIZE` to 64 and the concurrent user count to 100 kept the GPUs continuously fed and achieved 13,558 req/hr with tight latency (P95 30.0 s).

For the full analysis — including the batch engine architecture, GPU utilization, and the LLM-as-a-judge quality evaluation, take a peek at [`llm-qwen-deployment/README.md`](llm-qwen-deployment/README.md).

---

## What's Next

The current server uses a static batch window: collect requests for up to `BATCH_TIMEOUT` seconds, then fire a fixed batch. The next step is replacing this with a **custom token-level scheduler** that evicts completed sequences mid-batch and immediately admits new ones, the core mechanism behind continuous batching. This requires stepping outside `model.generate()` and running the decode loop manually, one step at a time, with direct control over the KV cache.

---

## Sample Benchmark Results (Notebook)

| Phase | Result |
|---|---|
| Phase 1 (KV Cache) | 400 tokens: 2.5 s cached vs. 25.3 s naive  **10.1× speedup** |
| Phase 3 (Sliding Window) | 2048 tokens: 32 MB (w=128) vs. 512 MB (full)  **16× memory reduction** |
| Phase 5 (Paged Attention) | 5 variable-length sequences: 4.2 MB paged vs. 16.8 MB naive  **4× savings** |
| Phase 7 (Deployment) | 2 pods, 1 hour: **13,558 requests served** at median 24.9 s latency |

---

## Getting Started

```bash
git clone <repo-url>
cd LLM-Inference-Optimisation

# Notebook (phases 1–6)
pip install torch torchvision transformers matplotlib
jupyter notebook LLM_inference_from_scratch.ipynb

# Deployment project (phase 7)
cd llm-qwen-deployment
pip install -r requirements.txt
cp .env.example .env   # fill in pod URLs
```

**GPU recommended:** Performance benefits are only observable with a CUDA-enabled GPU.
