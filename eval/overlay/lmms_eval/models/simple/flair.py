import hashlib
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

DIRECT = 'You are answering a multimodal question in DIRECT mode.\n\nAssigned mode: DIRECT\n\nOutput format (must follow exactly):\n<DIRECT>\n<think>\nBrief internal reasoning with minimal verification only (<=60 words).\n</think>\n<answer>\nFinal answer.\n</answer>\n</DIRECT>\n\nRules:\n- Use only DIRECT wrapper tags.\n- Exactly one <think> and one <answer>.\n- No text outside <DIRECT>...</DIRECT>.\n- Be decisive and concise.\n- Use only evidence visible in the provided input (video/image/text).\n- If options are provided (e.g., multiple choice), output the answer\n  in the expected option style.'
CONCISE = 'You are answering a multimodal question in CONCISE mode.\n\nAssigned mode: CONCISE\n\nOutput format (must follow exactly):\n<CONCISE>\n<think>\nCompact reasoning with key verification/falsification checks (<=200 words).\n</think>\n<answer>\nFinal answer.\n</answer>\n</CONCISE>\n\nRules:\n- Use only CONCISE wrapper tags.\n- Exactly one <think> and one <answer>.\n- No text outside <CONCISE>...</CONCISE>.\n- Ground claims in observed evidence; include only key checks.\n- Keep reasoning compact and useful.\n- Include grounding tags inside <think> when useful:\n  <obj>object_name</obj><box>[x1,y1,x2,y2]</box><t>time_in_seconds</t>\n- Do not output tags in <answer>.\n- If options are provided, output in the expected answer style.'
DEEP = 'You are answering a multimodal question in DEEP mode.\n\nAssigned mode: DEEP\n\nOutput format (must follow exactly):\n<DEEP>\n<think>\nDetailed verification/falsification reasoning with assumptions, alternatives,\ncounterexamples, and failure cases; grounded in observed evidence (<=1000 words).\n</think>\n<answer>\nFinal answer.\n</answer>\n</DEEP>\n\nRules:\n- Use only DEEP wrapper tags.\n- Exactly one <think> and one <answer>.\n- No text outside <DEEP>...</DEEP>.\n- Reason step-by-step and verify competing hypotheses before concluding.\n- Avoid unsupported claims; do not invent details.\n- Include explicit evidence references (objects/actions/time cues) when relevant.\n- Include grounding tags inside <think> when useful:\n  <obj>object_name</obj><box>[x1,y1,x2,y2]</box><t>time_in_seconds</t>\n- Do not output tags in <answer>.\n- If options are provided, output in the expected answer style.'
ADAPTIVE = 'You are in ADAPTIVE inference mode.\nYour job is to choose the cheapest reliable reasoning style for this\nquestion: DIRECT, CONCISE, or DEEP.\n\nMode definitions:\n- DIRECT: Fastest, minimal verification, shortest reasoning.\n- CONCISE: Moderate cost, compact but grounded checks.\n- DEEP: Highest cost, thorough verification with alternatives/failure-case analysis.\n\nDecision policy:\n1) Estimate task difficulty and ambiguity from the input.\n2) Prefer DIRECT when confidence is high and evidence is clear.\n3) Use CONCISE when moderate verification is needed.\n4) Use DEEP only when uncertainty, multi-step reasoning, conflicting evidence,\n   or strong grounding demands require it.\n\nThen answer using exactly one corresponding wrapper format:\n<DIRECT><think>...</think><answer>...</answer></DIRECT>\n<CONCISE><think>...</think><answer>...</answer></CONCISE>\n<DEEP><think>...</think><answer>...</answer></DEEP>\n\nRules:\n- Do NOT use <ADAPTIVE> wrapper.\n- Exactly one <think> and one <answer>.\n- No text outside the chosen wrapper.\n- Ground claims in provided evidence only.\n- If options are provided, output in the expected answer style.'

MODE_PROMPTS = {"DIRECT": DIRECT, "CONCISE": CONCISE, "DEEP": DEEP, "ADAPTIVE": ADAPTIVE, "NONE": None}
VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".flv", ".mpg", ".mpeg", ".wmv", ".3gp", ".ts")


def _bool(v):
    return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")


PLACEHOLDER = re.compile(r"<image[ _](\d+)>")
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()
SORTED_IMAGE_TASKS = ("emma", "mmmu")


@register_model("flair")
class Flair(lmms):
    is_simple = True

    def __init__(
        self,
        pretrained: str,
        mode: str = "ADAPTIVE",
        max_new_tokens: int = 2048,
        fps: float = 1.0,
        max_frames: int = 128,
        min_frames: int = 2,
        short_frame_tasks: str = "video_mmmu",
        short_max_frames: int = 32,
        frame_max_pixels: int = 401408,
        frame_cache: str = "",
        stats_dir: str = "",
        max_model_len: int = 40960,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 32768,
        max_images: int = 32,
        chunk_size: int = 128,
        decode_workers: int = 16,
        timestamps_line: bool = True,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        from swift.infer_engine import RequestConfig, VllmEngine

        self.mode = mode.upper()
        self.system = MODE_PROMPTS[self.mode]
        self.max_new_tokens = int(max_new_tokens)
        self.fps = float(fps)
        self.max_frames = int(max_frames)
        self.min_frames = int(min_frames)
        self.short_frame_tasks = [t for t in str(short_frame_tasks).split("|") if t]
        self.short_max_frames = int(short_max_frames)
        self.frame_max_pixels = int(frame_max_pixels)
        self.frame_cache = frame_cache or os.path.join(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "flair_frames")
        self.stats_dir = stats_dir
        self.chunk_size = int(chunk_size)
        self.decode_workers = int(decode_workers)
        self.timestamps_line = _bool(timestamps_line)
        self.batch_size_per_gpu = int(batch_size)
        limit = {"image": int(max_images), "video": 4}
        try:
            with open(os.path.join(pretrained, "config.json")) as f:
                hf_type = json.load(f).get("model_type")
        except (OSError, ValueError):
            hf_type = None
        swift_type = hf_type if hf_type in ("qwen2_5_vl", "qwen3_vl") else None
        self.engine = VllmEngine(
            pretrained,
            model_type=swift_type,
            template_type=swift_type,
            gpu_memory_utilization=float(gpu_memory_utilization),
            tensor_parallel_size=int(tensor_parallel_size),
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
            limit_mm_per_prompt=limit,
            enable_prefix_caching=True,
            engine_kwargs={"max_num_batched_tokens": int(max_num_batched_tokens)},
        )
        self.request_config = RequestConfig(max_tokens=self.max_new_tokens, temperature=0.0, top_p=1.0)
        self.tokenizer = self.engine.tokenizer
        add_request = self.engine._add_request

        def _add_request(inputs, *args, **kwargs):
            mm = inputs.get("mm_processor_kwargs") or {}
            fps = mm.get("fps")
            if isinstance(fps, (list, tuple)) and len(fps) == 1:
                mm["fps"] = float(fps[0])
            return add_request(inputs, *args, **kwargs)

        self.engine._add_request = _add_request

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def _frame_budget(self, task):
        if any(task.startswith(p) for p in self.short_frame_tasks):
            return self.short_max_frames
        return self.max_frames

    def _resize(self, img):
        w, h = img.size
        if w * h > self.frame_max_pixels:
            s = math.sqrt(self.frame_max_pixels / (w * h))
            img = img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)
        return img

    def _decode(self, path, budget):
        key = hashlib.md5(f"{os.path.abspath(path)}|{self.fps}|{budget}|{self.frame_max_pixels}".encode()).hexdigest()
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key, threading.Lock())
        with lock:
            return self._decode_locked(path, budget, key)

    def _decode_locked(self, path, budget, key):
        out = os.path.join(self.frame_cache, key[:2], key)
        meta = os.path.join(out, "meta.json")
        if os.path.exists(meta):
            with open(meta) as f:
                m = json.load(f)
            return [os.path.join(out, n) for n in m["files"]], m["times"]
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0), num_threads=2)
        total = len(vr)
        vfps = float(vr.get_avg_fps()) or 30.0
        duration = total / vfps
        n = int(min(budget, max(self.min_frames, math.floor(duration * self.fps))))
        n = max(1, min(n, total))
        times = [(i + 0.5) * duration / n for i in range(n)]
        idx = [min(total - 1, int(t * vfps)) for t in times]
        frames = vr.get_batch(idx).asnumpy()
        os.makedirs(out, exist_ok=True)
        files = []
        for i, fr in enumerate(frames):
            name = f"{i:03d}.jpg"
            self._resize(Image.fromarray(fr)).save(os.path.join(out, name), quality=95)
            files.append(name)
        tmp = f"{meta}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"files": files, "times": [round(t, 2) for t in times], "duration": duration}, f)
        os.replace(tmp, meta)
        return [os.path.join(out, n) for n in files], [round(t, 2) for t in times]

    def _visual_kind(self, v):
        if isinstance(v, Image.Image):
            return "image"
        if isinstance(v, str):
            return "video" if v.lower().endswith(VIDEO_EXT) else "image"
        if isinstance(v, dict) and "video" in v:
            return "video_dict"
        return "unknown"

    def _interleave(self, text, images, task):
        occ = PLACEHOLDER.findall(text)
        if not occ or not images:
            return None
        names = sorted(set(f"image_{k}" for k in occ))
        if any(task.startswith(p) for p in SORTED_IMAGE_TASKS) and len(names) == len(images):
            table = {n: images[i] for i, n in enumerate(names)}
            order = [table[f"image_{k}"] for k in occ]
        elif len(occ) == len(images):
            order = list(images)
        else:
            return None
        return PLACEHOLDER.sub("<image>", text), order

    def _build(self, ctx, visuals, task):
        images, videos, ts_lines = [], [], []
        budget = self._frame_budget(task)
        for v in visuals or []:
            kind = self._visual_kind(v)
            if kind == "image":
                img = v if isinstance(v, Image.Image) else Image.open(v)
                images.append(img.convert("RGB"))
            elif kind in ("video", "video_dict"):
                path = v["video"] if kind == "video_dict" else v
                files, times = self._decode(path, budget)
                videos.append(files)
                ts_lines.append("Frame timestamps (seconds): " + ", ".join(f"{t:.2f}" for t in times))
            else:
                eval_logger.warning(f"flair: unsupported visual {type(v)}")
        text = (ctx or "").replace("<image>", "").replace("<video>", "")
        inline = self._interleave(text, images, task)
        if inline is not None:
            text, images = inline
            head = ""
        else:
            head = "<image>" * len(images)
        for line in ts_lines:
            head += "<video>" + (line + "\n" if self.timestamps_line else "")
        system = [{"role": "system", "content": self.system}] if self.system else []
        return {"messages": system + [{"role": "user", "content": head + text.lstrip()}], "images": images, "videos": videos}

    def _prepare(self, req):
        ctx, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
        visuals = doc_to_visual(self.task_dict[task][split][doc_id])
        if visuals is not None and not isinstance(visuals, (list, tuple)):
            visuals = [visuals]
        return self._build(ctx, list(visuals or []), task)

    def _stats(self, task, rows):
        if not self.stats_dir:
            return
        os.makedirs(self.stats_dir, exist_ok=True)
        with open(os.path.join(self.stats_dir, f"{task}.jsonl"), "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        from swift.infer_engine import InferRequest

        results = [None] * len(requests)
        pbar = tqdm(total=len(requests), desc="flair")
        pool = ThreadPoolExecutor(self.decode_workers)
        order = list(range(len(requests)))
        for s in range(0, len(order), self.chunk_size):
            ids = order[s : s + self.chunk_size]
            built = list(pool.map(lambda i: self._safe_prepare(requests[i]), ids))
            infer_ids, infer_reqs = [], []
            for i, b in zip(ids, built):
                if b is None:
                    results[i] = ""
                else:
                    infer_ids.append(i)
                    infer_reqs.append(InferRequest(**b))
            outs = self.engine.infer(infer_reqs, self.request_config, use_tqdm=False) if infer_reqs else []
            rows = {}
            for i, o in zip(infer_ids, outs):
                text, ntok, fin = "", 0, "error"
                if o is not None and not isinstance(o, Exception):
                    text = o.choices[0].message.content or ""
                    ntok = o.usage.completion_tokens
                    fin = o.choices[0].finish_reason
                else:
                    eval_logger.error(f"flair: inference error {o}")
                results[i] = text
                ctx, _, _, doc_id, task, split = requests[i].args
                rows.setdefault(task, []).append({"doc_id": doc_id, "tokens": ntok, "finish": fin, "mode": _mode(text)})
            for t, r in rows.items():
                self._stats(t, r)
            pbar.update(len(ids))
        pool.shutdown()
        pbar.close()
        return results

    def _safe_prepare(self, req):
        try:
            return self._prepare(req)
        except Exception as e:
            eval_logger.error(f"flair: failed to prepare {req.args[3]} {req.args[4]}: {e}")
            return None

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until_multi_round(self, requests):
        raise NotImplementedError


def _mode(text):
    for m in ("DIRECT", "CONCISE", "DEEP"):
        if f"<{m}>" in text:
            return m
    return None
