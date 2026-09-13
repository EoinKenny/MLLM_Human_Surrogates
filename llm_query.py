"""One provider-independent LLM call for every experiment.

``call_llm`` always returns one raw response string per prompt and appends a
JSONL audit record containing the prompt, the byte-for-byte visible model
generation, a best-effort split of visible reasoning/final answer, and any
separate reasoning content the provider exposes.  It cannot record private
model activations or reasoning that a provider does not return.
"""
from __future__ import annotations

import atexit, base64, gc, hashlib, json, mimetypes, os, re, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_id: str
    dtype: str = "auto"
    revision: Optional[str] = None
    processor_id: Optional[str] = None
    processor_revision: Optional[str] = None

MODEL_REGISTRY = {
    "qwen3-vl-32b": ModelSpec(
        "vllm", "Qwen/Qwen3-VL-32B-Instruct-FP8", "auto",
        "4bf2c2f39c37c0fede78bede4056e1f18cdf81092"),
    "gemma3-27b-it": ModelSpec(
        "vllm", "pytorch/gemma-3-27b-it-FP8", "auto",
        "eeae0c843ad5ee3f1c9533d28aa4ae7fe1724fc9"),
    "mistral-small-3.2-24b": ModelSpec(
        "vllm", "stelterlab/Mistral-Small-3.2-24B-Instruct-2506-FP8", "auto",
        "3ee34ae98ba449de9df91e34cdd26e597d074aea"),
    "qwen3-vl-8b": ModelSpec("huggingface", "Qwen/Qwen3-VL-8B-Instruct", "float16"),
    "llama3.2-vision-11b": ModelSpec("huggingface", "meta-llama/Llama-3.2-11B-Vision-Instruct", "float16"),
    "gemma3-12b-it": ModelSpec("huggingface", "google/gemma-3-12b-it", "bfloat16"),
    "ministral-3-14b": ModelSpec("huggingface", "mistralai/Ministral-3-14B-Instruct-2512-BF16", "bfloat16"),
    "gpt-5": ModelSpec("openai", "gpt-5-2025-08-07"),
    "gpt-5-mini": ModelSpec("openai", "gpt-5-mini-2025-08-07"),
    "gpt-4o": ModelSpec("openai", "gpt-4o-2024-11-20"),
    "gpt-4o-mini": ModelSpec("openai", "gpt-4o-mini-2024-07-18"),
    "o4-mini": ModelSpec("openai", "o4-mini-2025-04-16"),
    "claude3.5-haiku": ModelSpec("bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0"),
    "claude3.5-sonnet": ModelSpec("bedrock", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    "claude3.7-sonnet": ModelSpec("bedrock", "anthropic.claude-3-7-sonnet-20250219-v1:0"),
    "claude4.5-sonnet": ModelSpec("bedrock", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
}
_custom_alias = os.getenv("LLM_CUSTOM_MODEL_ALIAS")
_custom_id = os.getenv("LLM_CUSTOM_MODEL_ID")
_custom_provider = os.getenv("LLM_CUSTOM_MODEL_PROVIDER")
if _custom_alias and _custom_id and _custom_provider:
    MODEL_REGISTRY[_custom_alias] = ModelSpec(_custom_provider, _custom_id)
MODEL_IDS = {name: spec.model_id for name, spec in MODEL_REGISTRY.items()}
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen3-vl-32b")
DEFAULT_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful assistant who imitates human users in explainable AI user testing.")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "10000"))
DEFAULT_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
DEFAULT_SEED = int(os.getenv("LLM_SEED", "42"))
TRACE_DIR = Path(os.getenv("LLM_TRACE_DIR", Path(__file__).resolve().parent / "data" / "llm_traces"))
_MODEL_LOCK, _TRACE_LOCK = threading.RLock(), threading.Lock()
_MODEL = _PROCESSOR = None
_LOADED_ID = None
_VLLM = _VLLM_PROCESSOR = None
_VLLM_LOADED_ID = None
_PAUSE_FILE = Path(__file__).resolve().parent / ".gpu_keepalive_pause"

def _resolve(name):
    if name in MODEL_REGISTRY: return MODEL_REGISTRY[name]
    if "/" in name: return ModelSpec("huggingface", name)
    raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODEL_REGISTRY)}")

def _image(path):
    if not path: return None, None
    p = Path(path).expanduser().resolve()
    if not p.is_file(): raise FileNotFoundError(f"Image not found: {p}")
    return p.read_bytes(), mimetypes.guess_type(p.name)[0] or "image/png"

def _split(text, reasoning=None):
    """Best-effort split while always preserving ``text`` separately.

    ``reasoning`` is provider-returned reasoning (or a reasoning summary).  If
    absent, recognise the explicit formats used by this repository.  The raw
    response remains the canonical complete visible generation.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    if reasoning:
        return reasoning.strip(), cleaned

    match = re.search(r"<think>(.*?)</think>", text, re.I | re.S)
    if match:
        return match.group(1).strip() or None, cleaned

    section = re.search(
        r"(?:^|\n)\s*REASONING\s*:\s*(.*?)(?=\n\s*CONFIGURATION\s*:)",
        text, re.I | re.S)
    if section:
        final = re.split(r"\n\s*CONFIGURATION\s*:\s*", text,
                         maxsplit=1, flags=re.I | re.S)[-1].strip()
        return section.group(1).strip() or None, final

    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    if answer:
        prefix = text[:answer.start()].strip()
        return prefix or None, answer.group(1).strip()

    # The actionable-recommendation prompt requests a JSON reasoning field.
    try:
        start, end = text.find("{"), text.rfind("}")
        payload = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
        visible = payload.get("reasoning") if isinstance(payload, dict) else None
        if isinstance(visible, str) and visible.strip():
            return visible.strip(), text.strip()
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return None, cleaned

def _request_seed(base_seed: int, name: str, system: str, prompt: str,
                  occurrence: int = 0) -> int:
    """Stable per-prompt seed, independent of batching and resume boundaries."""
    payload = f"{base_seed}\0{name}\0{system}\0{prompt}\0{occurrence}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _save_trace(name, spec, prompt, system, temperature, seed, image_path, raw,
                reasoning, final, elapsed, trace_metadata=None,
                provider_reasoning=None):
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = dict(trace_metadata or {})
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    run_id = metadata.pop("run_id", None) or os.getenv("LLM_RUN_ID")
    study = metadata.pop("study", None) or os.getenv("LLM_STUDY")
    leg = metadata.pop("leg", None) or os.getenv("LLM_LEG")
    record = {"schema_version": 2, "request_id": str(uuid.uuid4()),
      "timestamp_utc": datetime.now(timezone.utc).isoformat(), "model_name": name,
      "provider": spec.provider, "model_id": spec.model_id, "model_revision": spec.revision,
      "processor_id": spec.processor_id or spec.model_id,
      "processor_revision": spec.processor_revision or spec.revision,
      "run_id": run_id, "study": _canonical_study(study), "leg": leg,
      "instance_id": metadata.pop("instance_id", prompt_hash),
      "condition": metadata.pop("condition", None),
      "attempt": metadata.pop("attempt", 0),
      "temperature": temperature, "seed": seed,
      "generation_parameters": {"temperature": temperature, "seed": seed, "top_p": DEFAULT_TOP_P, "max_new_tokens": metadata.get("max_new_tokens")},
      "prompt_version": metadata.get("prompt_version"),
      "parser_result": _parse_response(_canonical_study(study), raw)[0],
      "parser_error": _parse_response(_canonical_study(study), raw)[1],
      "system_prompt": system, "prompt": prompt,
      "prompt_sha256": prompt_hash,
      "image_path": str(Path(image_path).expanduser().resolve()) if image_path else None,
      "image_metadata": _image_metadata(image_path),
      # raw_response is the authoritative, complete visible generation.
      "raw_response": raw, "raw_response_sha256": response_hash,
      "visible_reasoning": reasoning, "reasoning": reasoning,
      "reasoning_available": reasoning is not None, "final_answer": final,
      "provider_reasoning": provider_reasoning,
      "provider_reasoning_available": provider_reasoning is not None,
      "metadata": metadata,
      "latency_seconds": round(elapsed, 6)}
    target = TRACE_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}.jsonl"
    with _TRACE_LOCK, target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())

_ATTEMPTS = {}
_ATTEMPT_FILES = set()

def _canonical_study(study):
    return {"sys_eng":"system_engagement", "recourse":"actionable_recommendation",
       "model_improv":"model_improvement", "reliance_tabular":"adult", "reliance_nlp":"bios",
       "knowledge_extraction":"teaching"}.get(study, study)

def _logical_seed_key(metadata):
    return json.dumps([metadata.get("study"),metadata.get("instance_id"),metadata.get("condition"),metadata.get("attempt")],sort_keys=True)

def _allocate_attempts(name, prompts, metadata_items, limit):
    from audit_gate import VERSIONS
    target = TRACE_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}.jsonl"
    with _TRACE_LOCK:
        if str(target) not in _ATTEMPT_FILES:
            if target.exists():
                for line in target.read_text().splitlines():
                    r=json.loads(line)
                    key=(str(target),r.get("run_id"),_canonical_study(r.get("study")),str(r.get("instance_id")),r.get("condition"))
                    _ATTEMPTS[key]=max(_ATTEMPTS.get(key,0),r.get("attempt",0))
            _ATTEMPT_FILES.add(str(target))
        result=[]
        for prompt,item in zip(prompts,metadata_items):
            m=dict(item);m["study"]=_canonical_study(m.get("study") or os.getenv("LLM_STUDY"))
            m["run_id"]=m.get("run_id") or os.getenv("LLM_RUN_ID")
            if m.get("instance_id") is None or m.get("condition") is None or not m["study"] or not m["run_id"]:
                raise ValueError("Each collection request requires run, study, stable instance ID and condition")
            key=(str(target),m["run_id"],m["study"],str(m["instance_id"]),m["condition"])
            m["attempt"]=_ATTEMPTS.get(key,0)+1;_ATTEMPTS[key]=m["attempt"]
            version_key={"system_engagement":"engagement","actionable_recommendation":"recourse"}.get(m["study"],m["study"])
            m["prompt_version"]=VERSIONS.get(version_key,"unknown")
            m["max_new_tokens"]=limit
            result.append(m)
        return result

def _image_metadata(path):
    if not path:return None
    from PIL import Image
    with Image.open(path) as im:
        return {"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),"width":im.width,"height":im.height,"format":im.format}

def _parse_response(study,raw):
    answer=re.findall(r'<answer>\s*(.*?)\s*</answer>',raw,re.I|re.S)
    if study in ["adult","bios","teaching","system_engagement"]:
        allowed={"adult":{"earns < 50k","earns > 50k"},"bios":{"teacher","psychologist","surgeon","professor","physician"},
          "teaching":{"0","1","0.0","1.0"},"system_engagement":set("1234567")}[study]
        if len(answer)!=1 or answer[0].lower() not in allowed:return None,"Expected exactly one valid answer tag"
        return answer[0].lower(),None
    if study=="actionable_recommendation":
        try:
            obj=json.loads(raw[raw.index('{'):raw.rindex('}')+1])
            if not all(type(obj[k]) is int and 1<=obj[k]<=7 for k in ['acceptance_score','actionability_score']):raise ValueError('Invalid rating')
            return {k:obj[k] for k in ['acceptance_score','actionability_score']},None
        except (ValueError,KeyError,TypeError):return None,"Invalid recourse JSON ratings"
    if study=="model_improvement":
        try:
            obj=json.loads(raw[raw.index('{'):raw.rindex('}')+1])
            return obj,None
        except (ValueError,TypeError):return None,"Invalid configuration JSON; experiment validates feature settings separately"
    return None,"Unknown study parser"

def _release_gpu():
    try:
        if _PAUSE_FILE.exists() and _PAUSE_FILE.read_text().strip() == str(os.getpid()):
            _PAUSE_FILE.unlink(missing_ok=True)
    except OSError: pass

def unload_model():
    global _MODEL, _PROCESSOR, _LOADED_ID, _VLLM, _VLLM_PROCESSOR, _VLLM_LOADED_ID
    with _MODEL_LOCK:
        _MODEL = _PROCESSOR = _LOADED_ID = None
        _VLLM = _VLLM_PROCESSOR = _VLLM_LOADED_ID = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        except (ImportError, RuntimeError): pass
        _release_gpu()
atexit.register(_release_gpu)

def _load_hf(spec):
    global _MODEL, _PROCESSOR, _LOADED_ID
    with _MODEL_LOCK:
        if _MODEL is not None and _LOADED_ID == spec.model_id: return _MODEL, _PROCESSOR
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("LLM_GPU_IDS", "0,1,2"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoProcessor
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable for local Hugging Face inference")
        unload_model(); _PAUSE_FILE.write_text(str(os.getpid())); time.sleep(1.5)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(spec.dtype, "auto")
        if spec.model_id.startswith("Qwen/Qwen3-VL-"):
            from transformers import Qwen3VLForConditionalGeneration as cls
        elif spec.model_id.startswith("meta-llama/Llama-3.2-"):
            from transformers import MllamaForConditionalGeneration as cls
        elif spec.model_id.startswith("google/gemma-3"):
            from transformers import Gemma3ForConditionalGeneration as cls
        elif spec.model_id.startswith("mistralai/"):
            from transformers import Mistral3ForConditionalGeneration as cls
        else:
            from transformers import AutoModelForImageTextToText as cls
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        _PROCESSOR = AutoProcessor.from_pretrained(
            spec.processor_id or spec.model_id,
            revision=spec.processor_revision or spec.revision, token=token)
        _MODEL = cls.from_pretrained(spec.model_id, token=token, torch_dtype=dtype,
          revision=spec.revision,
          device_map="auto", low_cpu_mem_usage=True,
          attn_implementation=os.getenv("LLM_ATTENTION", "sdpa")).eval()
        _LOADED_ID = spec.model_id
        return _MODEL, _PROCESSOR

def _call_hf(prompt, spec, system, temperature, seed, image_path, max_tokens):
    import torch
    model, processor = _load_hf(spec)
    content = []
    if image_path:
        p = Path(image_path).expanduser().resolve()
        if not p.is_file(): raise FileNotFoundError(f"Image not found: {p}")
        content.append({"type": "image", "image": str(p)})
    content.append({"type": "text", "text": prompt})
    messages = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else [])
    messages.append({"role": "user", "content": content})
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
      return_dict=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        generator = torch.Generator(device=model.device).manual_seed(seed)
        kwargs.update(temperature=max(float(temperature), 1e-5),
                      top_p=DEFAULT_TOP_P, generator=generator)
    with torch.inference_mode(): generated = model.generate(**inputs, **kwargs)
    return processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0], None

def _call_openai(prompt, spec, system, temperature, seed, image_path, max_tokens):
    from openai import OpenAI
    content = [{"type": "input_text", "text": prompt}]
    data, mime = _image(image_path)
    if data: content.insert(0, {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(data).decode()}"})
    kwargs = {"model": spec.model_id, "instructions": system or None,
      "input": [{"role": "user", "content": content}], "max_output_tokens": max_tokens}
    # Reasoning models may reject temperature; retrying without it is explicit and reproducible.
    kwargs["temperature"] = temperature
    try: response = OpenAI().responses.create(**kwargs)
    except Exception as exc:
        if "temperature" not in str(exc).lower(): raise
        kwargs.pop("temperature"); response = OpenAI().responses.create(**kwargs)
    summaries = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "reasoning":
            summaries += [getattr(x, "text", "") for x in getattr(item, "summary", []) or []]
    return response.output_text, "\n".join(filter(None, summaries)) or None

def _call_bedrock(prompt, spec, system, temperature, seed, image_path, max_tokens):
    import boto3
    content = []
    data, mime = _image(image_path)
    if data: content.append({"image": {"format": mime.split("/")[-1].replace("jpg", "jpeg"), "source": {"bytes": data}}})
    content.append({"text": prompt})
    request = {"modelId": spec.model_id, "messages": [{"role": "user", "content": content}],
      "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens}}
    if system: request["system"] = [{"text": system}]
    blocks = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1")).converse(**request)["output"]["message"]["content"]
    text = "\n".join(x["text"] for x in blocks if "text" in x)
    reasoning = "\n".join(x.get("reasoningContent", {}).get("reasoningText", {}).get("text", "") for x in blocks if "reasoningContent" in x).strip()
    return text, reasoning or None

def _load_vllm(spec):
    """Load one persistent offline engine, following the proven kmeans setup."""
    global _VLLM, _VLLM_PROCESSOR, _VLLM_LOADED_ID
    with _MODEL_LOCK:
        if _VLLM is not None and _VLLM_LOADED_ID == spec.model_id:
            return _VLLM, _VLLM_PROCESSOR
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("LLM_GPU_IDS", "0"))
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        # The G6e image has CUDA runtime libraries but no nvcc compiler. Avoid
        # FlashInfer's sampling JIT; attention and FP8 matmuls remain GPU-native.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if spec.model_id == "pytorch/gemma-3-27b-it-FP8":
            # Required by this TorchAO checkpoint until its compile-cache
            # composition issue is resolved upstream.
            os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
        from transformers import AutoProcessor
        from vllm import LLM
        unload_model(); _PAUSE_FILE.write_text(str(os.getpid())); time.sleep(1.5)
        is_gemma3_27b_fp8 = spec.model_id == "pytorch/gemma-3-27b-it-FP8"
        is_mistral_24b_fp8 = "Mistral-Small-3.2-24B" in spec.model_id
        processor_location = spec.processor_id or spec.model_id
        processor_revision = spec.processor_revision or spec.revision
        if is_mistral_24b_fp8:
            # This checkpoint is in Mistral's native consolidated format.
            # Native vLLM chat performs both Tekken tokenization and Pixtral
            # image preprocessing, avoiding lossy HF-tokenizer conversions.
            _VLLM_PROCESSOR = None
        else:
            _VLLM_PROCESSOR = AutoProcessor.from_pretrained(
                processor_location, revision=processor_revision,
                token=os.getenv("HF_TOKEN") or None)
        llm_kwargs = dict(
            model=spec.model_id,
            revision=spec.revision,
            seed=DEFAULT_SEED,
            dtype=spec.dtype,
            tensor_parallel_size=int(os.getenv("LLM_TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("LLM_GPU_MEMORY_UTILIZATION", "0.92")),
            # Profiling showed that 16 concurrent Gemma images at 8K leaves
            # safe activation/KV headroom on a single 46 GiB L40S.
            max_model_len=int(os.getenv(
                "LLM_MAX_MODEL_LEN", "8192" if (
                    is_gemma3_27b_fp8 or is_mistral_24b_fp8) else "16384")),
            max_num_seqs=int(os.getenv(
                "LLM_MAX_NUM_SEQS", "16" if (
                    is_gemma3_27b_fp8 or is_mistral_24b_fp8) else "64")),
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 1, "video": 0},
            trust_remote_code=False,
        )
        if is_mistral_24b_fp8:
            llm_kwargs.update(tokenizer_mode="mistral", config_format="mistral",
                               load_format="mistral")
        else:
            llm_kwargs.update(tokenizer=processor_location,
                               tokenizer_revision=processor_revision)
        _VLLM = LLM(**llm_kwargs)
        _VLLM_LOADED_ID = spec.model_id
        return _VLLM, _VLLM_PROCESSOR

def _call_vllm_batch(prompts, name, spec, system, temperature, base_seed,
                     image_path, max_tokens, trace_metadata):
    """Render once, then let vLLM continuously batch the complete request list."""
    from vllm import SamplingParams
    engine, processor = _load_vllm(spec)
    pil_image = None
    if image_path:
        from PIL import Image
        pil_image = Image.open(Path(image_path).expanduser().resolve()).convert("RGB")
    requests = []
    native_messages = []
    native_image_url = None
    if pil_image is not None and processor is None:
        image_bytes, image_mime = _image(image_path)
        native_image_url = (
            f"data:{image_mime};base64,{base64.b64encode(image_bytes).decode()}")
    for prompt in prompts:
        content = []
        if pil_image is not None:
            if processor is None:
                content.append({"type": "image_url",
                                "image_url": {"url": native_image_url}})
            else:
                content.append({"type": "image_pil", "image_pil": pil_image})
        content.append({"type": "text", "text": prompt})
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": content})
        if processor is None:
            native_messages.append(messages)
        else:
            rendered_messages = ([{
                "role": "system",
                "content": [{"type": "text", "text": system}],
            }] if system else [])
            rendered_messages.append({"role": "user", "content": [
                *([{"type": "image", "image": "<image>"}]
                  if pil_image is not None else []),
                {"type": "text", "text": prompt},
            ]})
            rendered = processor.apply_chat_template(
                rendered_messages, tokenize=False, add_generation_prompt=True)
            request = {"prompt": rendered}
            if pil_image is not None:
                request["multi_modal_data"] = {"image": pil_image}
            requests.append(request)
    seeds = [_request_seed(base_seed, name, system, prompt, _logical_seed_key(metadata))
             for prompt, metadata in zip(prompts, trace_metadata)]
    params = [SamplingParams(
        temperature=float(temperature),
        top_p=DEFAULT_TOP_P if temperature > 0 else 1.0,
        max_tokens=max_tokens,
        seed=seed,
    ) for seed in seeds]
    started = time.time()
    outputs = (engine.chat(native_messages, params, use_tqdm=True)
               if processor is None
               else engine.generate(requests, params, use_tqdm=True))
    elapsed = time.time() - started
    responses = []
    for prompt, seed, output, metadata in zip(prompts, seeds, outputs,
                                              trace_metadata):
        raw = output.outputs[0].text
        metadata=dict(metadata,output_tokens=len(output.outputs[0].token_ids),input_tokens=len(output.prompt_token_ids or []),finish_reason=output.outputs[0].finish_reason,stop_reason=output.outputs[0].stop_reason)
        reasoning, final = _split(raw)
        _save_trace(name, spec, prompt, system, temperature, seed, image_path,
                    raw, reasoning, final, elapsed / max(len(prompts), 1),
                    metadata)
        responses.append(raw)
    if pil_image is not None:
        pil_image.close()
    return responses

def _one(prompt, name, spec, system, temperature, base_seed, occurrence,
         image_path, max_tokens, trace_metadata):
    start = time.time()
    fn = {"huggingface": _call_hf, "openai": _call_openai, "bedrock": _call_bedrock}.get(spec.provider)
    if fn is None: raise ValueError(f"Unsupported provider {spec.provider!r}")
    seed = _request_seed(base_seed, name, system, prompt, _logical_seed_key(trace_metadata))
    raw, exposed_reasoning = fn(prompt, spec, system, temperature, seed, image_path, max_tokens)
    reasoning, final = _split(raw, exposed_reasoning)
    _save_trace(name, spec, prompt, system, temperature, seed, image_path, raw,
                reasoning, final, time.time() - start, trace_metadata,
                exposed_reasoning)
    return raw

def call_llm(prompts: Union[str, Sequence[str]], model_name: str = DEFAULT_MODEL,
             temperature: float = 1.0, image_path: Optional[str] = None,
             max_new_tokens: Optional[int] = None, batch_size: Optional[int] = None,
             system_prompt: str = DEFAULT_SYSTEM_PROMPT,
             seed: int = DEFAULT_SEED,
             trace_metadata: Optional[Union[Mapping[str, Any],
                                            Sequence[Mapping[str, Any]]]] = None
             ) -> list[str]:
    """Return one response per prompt, preserving input order across backends."""
    from audit_gate import check_approval
    check_approval()
    items = [prompts] if isinstance(prompts, str) else list(prompts)
    if not all(isinstance(x, str) for x in items): raise TypeError("Every prompt must be a string")
    if not items: return []
    if trace_metadata is None:
        metadata_items = [{} for _ in items]
    elif isinstance(trace_metadata, Mapping):
        metadata_items = [dict(trace_metadata) for _ in items]
    else:
        metadata_items = [dict(item) for item in trace_metadata]
        if len(metadata_items) != len(items):
            raise ValueError("trace_metadata must contain one item per prompt")
    spec, limit = _resolve(model_name), max_new_tokens or DEFAULT_MAX_NEW_TOKENS
    if os.getenv("LLM_FORCE_TEMPERATURE") is not None:
        temperature = float(os.environ["LLM_FORCE_TEMPERATURE"])
    from audit_gate import reserve_pilot_requests
    reserve_pilot_requests(items, model_name, metadata_items, float(temperature), limit, seed, image_path, system_prompt, DEFAULT_TOP_P)
    metadata_items = _allocate_attempts(model_name, items, metadata_items, limit)
    seen = {}
    indexed = []
    for prompt in items:
        occurrence = seen.get(prompt, 0); seen[prompt] = occurrence + 1
        indexed.append((prompt, occurrence))
    invoke = lambda item: _one(item[0], model_name, spec, system_prompt,
                               float(temperature), seed, item[1], image_path,
                               limit, item[2])
    if spec.provider == "vllm":
        return _call_vllm_batch(items, model_name, spec, system_prompt,
                                float(temperature), seed, image_path, limit,
                                metadata_items)
    indexed = [(prompt, occurrence, metadata_items[i])
               for i, (prompt, occurrence) in enumerate(indexed)]
    if spec.provider == "huggingface" or len(items) == 1: return [invoke(p) for p in indexed]
    with ThreadPoolExecutor(max_workers=max(1, min(batch_size or int(os.getenv("LLM_API_CONCURRENCY", "8")), len(items)))) as pool:
        return list(pool.map(invoke, indexed))

def call_model(model_name, system_prompt, prompt, temperature=1.0, image_path=None, max_new_tokens=None):
    return call_llm(prompt, model_name, temperature, image_path, max_new_tokens, system_prompt=system_prompt)[0]

def describe_model(model_name=DEFAULT_MODEL):
    """Resolve configuration without loading or calling a model."""
    return asdict(_resolve(model_name))
