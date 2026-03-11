# Qwen Moral Story Serving Pipeline

High-throughput inference deployment for a sample use case such as a small children's moral story generation service using `Qwen/Qwen2.5-1.5B-Instruct`. The goal is to serve as many queries as possible within one hour on commodity GPU hardware, using a custom-built serving stack built from scratch in Python.

---

## Goal

Extend the optimizations explored in `LLM_inference_from_scratch.ipynb` KV caching, dynamic batching, INT4 quantization into a production grade serving pipeline and quantify how much throughput can be extracted from a small LLM on a single-class GPU without any inference framework.

**Target: >10,000 requests served within one hour.**

---

## Architecture

```
Locust (100 users, local machine)
        │
        ├─── round-robin ───┐
        │                   │
        ▼                   ▼
   Pod 1 (RunPod)      Pod 2 (RunPod)
   RTX 2000 Ada        RTX 2000 Ada
   ┌─────────────┐     ┌─────────────┐
   │ custom_     │     │ custom_     │
   │ server.py   │     │ server.py   │
   │ BS=64       │     │ BS=64       │
   └─────────────┘     └─────────────┘
        │                   │
        └────────┬───────────┘
                 ▼
         request_metrics_*.csv
         (per-request latency, tokens/sec)
```

Both pods serve independently this is **data parallelism**: each pod holds a full model copy and handles independent requests. Locust distributes traffic in a strict round-robin across both pod URLs, guaranteeing an exact 50/50 split.

---

## Serving Pipeline (`custom_server.py`)

Built from scratch, extending the notebook experiments. Served via FastAPI + uvicorn.

| Optimization | Implementation |
|---|---|
| **Dynamic batching** | Async request queue (`asyncio.Queue`) feeds a background engine thread that drains up to `MAX_BATCH_SIZE` requests and runs a single `model.generate()` call for the entire batch. GPU matrix multiplications are amortised across N sequences. |
| INT4 NF4 Quantization | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` reduces model to ~750 MB VRAM |
| Flash Attention (SDPA) | `attn_implementation="sdpa"` |
| Warmup pass | One dummy forward pass on startup to pre-compile CUDA kernels before the first real request |
| Per-request metrics | CSV log with `queue_wait_s`, `inference_s`, `total_s`, `prompt_tokens`, `completion_tokens`, `batch_size` |

### How the batch engine works

```
asyncio.Queue (FastAPI coroutines)
          │  relay
          ▼
    threading.Queue (bridge)
          │
    engine thread (background):
      1. Block until first request arrives
      2. Wait up to BATCH_TIMEOUT (100 ms) for more
      3. Drain up to MAX_BATCH_SIZE requests
      4. model.generate(**padded_batch)  ← single GPU call for all N
      5. Resolve each request's asyncio.Future with its slice of output
      6. Repeat
```

This amortises the fixed cost of loading model weights across N sequences per decode step, which is the primary lever for throughput on memory-bandwidth-bound hardware.

---

## Why Batching Over Speculative Decoding

During development I attempted to combine speculative decoding with batching to maximise both per-token speed and GPU utilisation. Three independent obstacles made this impractical:

### 1. The Ragged Tensor Problem

In speculative decoding the draft model proposes K candidate tokens per sequence, and the main model accepts a *different* number per sequence (anywhere from 0 to K). After the accept/reject step every sequence in the batch has a different length. Feeding them back into the next forward pass requires:

- Re-padding every sequence to a new common length each iteration
- Trimming each sequence's KV cache to a different position
- Rebuilding attention masks from scratch every step

This creates a **ragged tensor** problem: the batch dimension can no longer be represented as a single rectangular tensor without per-step dynamic padding. The overhead of constantly re-padding and trimming eliminates most of the throughput gain that batching provides.

### 2. HuggingFace `generate()` Limitation

HuggingFace's built-in speculative decoding (`model.generate(assistant_model=...)`) hardcodes `batch_size=1`. It is a convenience wrapper, not the underlying algorithm. Calling it with a batch raises `ValueError: assisted generate is only supported for batch_size = 1`.

### 3. Qwen2.5 HybridCache

Bypassing `generate()` and calling `model.forward()` directly requires manual KV cache manipulation (slicing, padding, trimming per layer). Qwen2.5 uses a mix of sliding-window and full-attention layers, so transformers returns a `HybridCache` object instead of a plain tuple of `(key, value)` tensors. `HybridCache` does not expose per-layer tensors via subscript access (`cache[layer][0]`), making custom KV surgery infeasible without rewriting the cache class itself.

### The Tradeoff

Our smoke tests confirmed the decision empirically:

| Configuration | Avg Latency | Throughput | Failure Rate |
|---|---|---|---|
| Speculative decoding, batch=1 | ~100 s | 0.09 req/s | 1.92% |
| Dynamic batching, batch=8 | ~27 s | 0.34–0.80 req/s | 0% |

On 2 pods serving thousands of queries, **GPU utilisation matters more than per-token latency reduction**. Batching amortises matrix multiplications across N sequences in a single forward pass; speculative decoding at batch=1 leaves the GPU idle between requests.

---

## Experimental Results

### Smoke Test 1 : Speculative Decoding (batch=1)

**Config:** `MAX_BATCH_SIZE=1`, `assistant_model=Qwen2.5-0.5B-Instruct`, 10 Locust users, single pod

| Metric | Result |
|---|---|
| Total requests | 104 |
| Failed requests | 2 (1.92%) —> ReadTimeout after 120 s |
| Avg response time | ~99,550 ms (~100 s) |
| Median (P50) latency | ~102,000 ms |
| Throughput | 0.09 req/s |

**Verdict:** Unusably slow. With `batch_size=1` enforced by speculative decoding, each of the 10 concurrent users had to wait for all previous requests to complete, requests queued serially. GPU was barely utilised.

---

### Smoke Test 2 : Dynamic Batching (BS=8)

**Config:** `MAX_BATCH_SIZE=8`, speculative decoding disabled, 10 Locust users, single pod

| Metric | Result |
|---|---|
| Total requests | 109 |
| Failed requests | 0 (0%) |
| Avg response time | ~27,250 ms (~27 s) |
| Throughput | 0.34–0.80 req/s |

| Metric | Spec Decoding (batch=1) | Dynamic Batching (batch=8) | Improvement |
|---|---|---|---|
| Avg latency | ~99,550 ms | ~27,250 ms | **3.7× faster** |
| Failure rate | 1.92% | 0% | **Zero failures** |

**Key takeaway:** Dynamic batching with BS=8 delivered a **3.7× latency reduction** and eliminated all failures. The GPU processes 8 requests simultaneously per forward pass instead of 1.

---

### Production Run 1 : Scaling Baseline (BS=8, 20 users, 1 hour)

**Config:** `MAX_BATCH_SIZE=8`, 20 Locust users, 2 pods, 1 hour

| Metric | Result |
|---|---|
| Total requests served | 4,087 |
| Successful | 4,080 (99.83%) |
| Throughput | **4,090 req/hr** |
| Median (P50) latency | 37.6 s |
| P95 latency | 75.9 s |
| P99 latency | 82.2 s |
| Median tokens/sec | 5.4 |

**Observation:** With only 20 users, requests arrived too slowly to fill BS=8 batches consistently. The queue drained between batches, leaving the GPU idle for significant stretches. P95 latency of 75.9 s indicates head-of-line blocking when batches did accumulate.

---

### Production Run 2 : Optimized (BS=64, 100 users, 1 hour)

**Config:** `MAX_BATCH_SIZE=64`, `BATCH_TIMEOUT=0.1s`, 100 Locust users, 2× RTX 2000 Ada pods, 1 hour

| Metric | Result |
|---|---|
| Total requests served | 13,522 |
| Successful | **13,513 (99.93%)** |
| Throughput | **13,558 req/hr** ✓ 136% of target |
| Median (P50) latency | 24.9 s |
| P90 latency | 29.6 s |
| P95 latency | 30.0 s |
| P99 latency | 30.8 s |
| Median tokens/sec | 7.9 |
| Per-pod split | 50% / 50% (exact round-robin) |
| Failures | 9 (0.07%) —> HTTP 404/502 |

**Result: 13,558 requests served in one hour, exceeding the 10,000 target by 36%.**

The latency distribution is tight where P50 through P99 span about 6 s only because the queue stays deep enough to fire full batches without head-of-line blocking. **Both GPUs’ utilization remained constantly above 90%** throughout the run, confirming that the batch engine keeps the hardware saturated.

---

## LLM-as-a-Judge Quality Evaluation

To verify that throughput gains did not come at the cost of output quality, 30 held-out prompts from `eval_data.json` were passed to the server and the responses scored by a judge LLM on four dimensions (0.0–1.0 each).

| Dimension | Score | What it measures |
|---|---|---|
| Moral Clarity | **0.85** | Is the intended lesson clear and identifiable? |
| Age-Appropriateness | **0.97** | Is language and tone suitable for children aged 4–10? |
| Narrative Coherence | **0.72** | Does the story have a complete beginning, middle, and end? |
| Relevance | **0.94** | Does the story match the requested character, moral, and setting? |
| **Overall** | **0.87** | Mean across all four dimensions |

**Observations:**

- Age-appropriateness (0.97) and relevance (0.94) are consistently strong, the model reliably generates on-topic, child-friendly content.
- Narrative coherence (0.72) is the primary weakness. The judge frequently noted abrupt endings, which is an artifact of the model hitting `max_tokens=350` mid-sentence. Increasing the token budget or post-processing to trim at a sentence boundary would raise this score.
- Moral clarity (0.85) is solid but occasionally suffers when the model states the moral didactically rather than demonstrating it through the narrative.

---

## Inference Parallelism

This deployment uses **Data Parallelism (DP)**: each pod holds a complete copy of the model and handles independent requests. Locust distributes traffic in strict round-robin across both pod URLs.

Alternative strategies (not implemented here):

- **Tensor Parallelism (TP)**: splits individual weight matrices across multiple GPUs on the same machine which reduces per-GPU memory, enables larger models.
- **Pipeline Parallelism (PP)**: assigns different transformer layers to different GPUs, useful for very deep models that don't fit in a single GPU's memory.

For a 1.5 B-parameter model on 20 GB VRAM, DP across 2 pods is the right strategy: the model fits comfortably on one GPU, so parallelizing the workload (not the model) is optimal.

---

## Future Work

### Custom Request Scheduler (Continuous Batching)

The current batch engine uses a static window: it collects requests for up to `BATCH_TIMEOUT` seconds, then fires a fixed batch. This means sequences that finish early hold up the batch until all sequences complete which is a form of head-of-line blocking at the batch level.

The next step is to replace this with a **custom token-level scheduler** that:

1. Maintains a pool of active sequences being decoded step-by-step.
2. After each decode step, evicts sequences that have generated their `<eos>` token and immediately admits new sequences from the waiting queue into the freed KV-cache slots.
3. Runs the forward pass over a dynamically sized, always-full set of active sequences.

This is the architectural mechanism behind continuous batching, the core scheduling innovation that allows production-grade inference systems to avoid head-of-line blocking entirely. Implementing it requires stepping outside `model.generate()` and running the decode loop manually, one token at a time, with full control over the KV cache.

The primary engineering challenges:

- **Manual decode loop**: call `model.forward()` directly, manage the KV cache explicitly across steps.
- **Sequence lifecycle management**: track which positions in the batch tensor belong to which request, and handle variable-length eviction and admission without reordering the entire batch.
- **Padding-free computation** (stretch goal): pack sequences without padding using custom attention masks or block-sparse kernels to eliminate computation on padding tokens.

This is the implementation path toward the kind of throughput efficiency that purpose-built inference engines achieve through their schedulers and it is the natural next phase of this project.

### PagedAttention-Style KV Memory

The current implementation allocates a contiguous KV cache for the entire maximum sequence length at batch creation time. A block-based allocator would allow KV memory to grow on demand, reducing VRAM waste for short sequences and enabling larger effective batch sizes.

---

## Repository Structure

```
llm-qwen-deployment/
├── custom_server.py       # FastAPI server with dynamic batching + INT4 quantization
├── locustfile.py          # Load test client (round-robin across pods)
├── llm_as_eval.py         # LLM-as-a-judge quality scorer
├── generate_prompts.py    # 1,500 synthetic children's story prompts
├── eval_data.json         # 30 held-out prompts + reference stories for quality eval
├── requirements.txt       # Dependencies
├── .env.example           # Environment variable template
└── results/
    ├── request_metrics_custom_2pod_1h.csv           # Production Run 1 per-request metrics
    ├── request_metrics_rtx_2000_ada_2pod_bs64.csv   # Production Run 2 per-request metrics (RTX 2000 Ada)
    ├── locust_custom_2pod_1h_stats.csv             # Locust aggregate stats (Run 1)
    ├── locust_custom_2pod_rtx_2000_ada_bs64_stats.csv  # Locust aggregate stats (Run 2)
    └── eval_results.json                            # LLM-as-a-judge scores (30 prompts)
```

---

## Setup

### 1. Local machine

```bash
cd llm-qwen-deployment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in POD_1_URL, POD_2_URL
```

### 2. RunPod pods

Create two pods using **RTX 2000 Ada** with RunPod's PyTorch image (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` or similar). SSH into each and install dependencies:

```bash
pip install fastapi "uvicorn[standard]" transformers accelerate bitsandbytes torch
```

Upload `custom_server.py` to each pod:

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 \
    llm-qwen-deployment/custom_server.py \
    root@<POD_IP>:/root/custom_server.py
```

Start the server on each pod:

```bash
MAX_BATCH_SIZE=64 BATCH_TIMEOUT=0.1 PORT=8000 \
    nohup python custom_server.py > server.log 2>&1 &
```

Verify health:

```bash
curl https://<POD_ID>-8000.proxy.runpod.net/health
```

### 3. Run load test (local machine)

```bash
cd llm-qwen-deployment

RUN_TAG=custom_2pod_bs64 \
POD_1_URL=https://<POD1>-8000.proxy.runpod.net \
POD_2_URL=https://<POD2>-8000.proxy.runpod.net \
locust -f locustfile.py --headless \
    --users 100 \
    --spawn-rate 5 \
    --run-time 1h \
    --host https://<POD1>-8000.proxy.runpod.net \
    --csv results/locust_custom_2pod_bs64
```

### 4. Quality evaluation (after load test)

```bash
python llm_as_eval.py
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
POD_1_URL=https://<pod1-id>-8000.proxy.runpod.net
POD_2_URL=https://<pod2-id>-8000.proxy.runpod.net
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
JUDGE_API_KEY=...
JUDGE_API_BASE=https://api.openai.com/v1
JUDGE_MODEL=gpt-4o-mini
```
