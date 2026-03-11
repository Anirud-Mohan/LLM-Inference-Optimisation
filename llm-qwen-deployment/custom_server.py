import asyncio
import csv
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue

import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import uvicorn

# ── Compatibility patch ───────────────────────────────────────────────────────
# Some torch builds (e.g. 2.10) are missing nn.Module.set_submodule, which
# transformers calls during BitsAndBytes INT4 weight replacement.
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        parts = target.split(".")
        parent = self
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)
    nn.Module.set_submodule = _set_submodule

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID       = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
BATCH_TIMEOUT  = float(os.getenv("BATCH_TIMEOUT", "0.05"))
PORT           = int(os.getenv("PORT", "8888"))
METRICS_FILE   = os.getenv("METRICS_FILE", "/root/request_metrics.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("batch_server")

# ── Globals ───────────────────────────────────────────────────────────────────
_tokenizer = None
_model     = None


# ── Request descriptor ────────────────────────────────────────────────────────
@dataclass
class _Request:
    messages:   list[dict]
    max_tokens: int
    future:     asyncio.Future
    loop:       asyncio.AbstractEventLoop
    t_arrived:  float = field(default_factory=time.perf_counter)


# ── Queues ────────────────────────────────────────────────────────────────────
_async_queue: asyncio.Queue | None = None
_bridge: Queue = Queue()


# ── Metrics writer ────────────────────────────────────────────────────────────
_metrics_lock  = threading.Lock()
_metrics_first = True


def _write_metric(row: dict) -> None:
    global _metrics_first
    with _metrics_lock:
        write_header = _metrics_first
        _metrics_first = False
        with open(METRICS_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model() -> None:
    global _tokenizer, _model

    log.info("Loading tokenizer from %s …", MODEL_ID)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "left"

    log.info("Loading model %s (INT4 NF4, SDPA) …", MODEL_ID)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_cfg,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()

    n_params = sum(p.numel() for p in _model.parameters()) / 1e6
    log.info("Model ready on %s — %.0fM params", DEVICE, n_params)

    # Warmup: run one dummy forward pass so CUDA kernels are compiled and cached
    # before the first real request arrives. Skips this cost from request latency.
    log.info("Running warmup pass …")
    with torch.no_grad():
        dummy = _tokenizer("warmup", return_tensors="pt").to(DEVICE)
        _model.generate(**dummy, max_new_tokens=4, do_sample=False, pad_token_id=_tokenizer.pad_token_id)
    log.info("Warmup done — server ready to serve requests.")


# ── Batch inference ───────────────────────────────────────────────────────────
@torch.no_grad()
def _run_batch(batch: list[_Request]) -> None:
    """Run a single model.generate() call for the whole batch."""
    assert _tokenizer and _model

    prompts = [
        _tokenizer.apply_chat_template(
            r.messages, tokenize=False, add_generation_prompt=True
        )
        for r in batch
    ]
    max_new = max(r.max_tokens for r in batch)

    enc = _tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    t0 = time.perf_counter()
    out_ids = _model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=_tokenizer.pad_token_id,
    )
    inference_s = time.perf_counter() - t0

    prompt_len = enc["input_ids"].shape[1]
    now = time.perf_counter()

    for i, req in enumerate(batch):
        new_ids = out_ids[i, prompt_len:]
        text    = _tokenizer.decode(new_ids, skip_special_tokens=True)
        n_comp  = int((new_ids != _tokenizer.pad_token_id).sum())

        result = {
            "text":              text,
            "prompt_tokens":     prompt_len,
            "completion_tokens": n_comp,
            "batch_size":        len(batch),
            "inference_s":       round(inference_s, 4),
        }

        _write_metric({
            "timestamp":         round(now, 3),
            "queue_wait_s":      round(t0 - req.t_arrived, 4),
            "inference_s":       round(inference_s, 4),
            "total_s":           round(now - req.t_arrived, 4),
            "prompt_tokens":     prompt_len,
            "completion_tokens": n_comp,
            "batch_size":        len(batch),
        })

        req.loop.call_soon_threadsafe(
            lambda r=req, res=result: (
                None if r.future.done() else r.future.set_result(res)
            )
        )


# ── Batch engine (background thread) ─────────────────────────────────────────
def _engine_loop() -> None:
    """Continuously drain _bridge in batches of up to MAX_BATCH_SIZE."""
    log.info("Engine loop started — max_batch=%d  timeout=%.3fs", MAX_BATCH_SIZE, BATCH_TIMEOUT)

    while True:
        batch: list[_Request] = []

        try:
            batch.append(_bridge.get(timeout=1.0))
        except Empty:
            continue

        deadline = time.perf_counter() + BATCH_TIMEOUT
        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(_bridge.get(timeout=remaining))
            except Empty:
                break

        try:
            _run_batch(batch)
        except Exception as exc:
            for req in batch:
                req.loop.call_soon_threadsafe(
                    lambda r=req, e=exc: (
                        None if r.future.done() else r.future.set_exception(e)
                    )
                )


# ── Async relay (asyncio Queue → thread-safe bridge) ─────────────────────────
async def _relay_loop() -> None:
    while True:
        req = await _async_queue.get()
        _bridge.put(req)


# ── FastAPI application ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _async_queue
    _load_model()
    _async_queue = asyncio.Queue()
    threading.Thread(target=_engine_loop, daemon=True, name="engine").start()
    asyncio.create_task(_relay_loop())
    log.info("Server ready — max_batch=%d  port=%d", MAX_BATCH_SIZE, PORT)
    yield


app = FastAPI(title="Dynamic Batching LLM Server", lifespan=lifespan)


class _Message(BaseModel):
    role:    str
    content: str


class _ChatRequest(BaseModel):
    messages:   list[_Message]
    max_tokens: int = MAX_NEW_TOKENS


@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "model":       MODEL_ID,
        "batch_size":  MAX_BATCH_SIZE,
        "device":      DEVICE,
        "queue_depth": _bridge.qsize(),
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: _ChatRequest):
    loop   = asyncio.get_event_loop()
    future = loop.create_future()

    await _async_queue.put(_Request(
        messages=[m.model_dump() for m in req.messages],
        max_tokens=req.max_tokens,
        future=future,
        loop=loop,
    ))

    try:
        result = await asyncio.wait_for(future, timeout=300.0)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "request timed out"})

    return {
        "id": f"chatcmpl-batch",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
        },
        "custom": {
            "batch_size":  result["batch_size"],
            "inference_s": result["inference_s"],
        },
    }


# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
