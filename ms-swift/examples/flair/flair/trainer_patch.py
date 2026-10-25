import os
from typing import Any, List

import torch
from accelerate.utils import gather_object

from .parsing import THINK, heuristic_span_scores, spans
from .prompts import MODE_PROMPTS, SLOT_MODES
from .replay import _flat, get_replay
from .scorer import get_scorer


def slot_modes():
    v = os.environ.get('FLAIR_SLOT_MODES')
    return v.split(',') if v else SLOT_MODES


def mode_for_slot(slot, k):
    modes = slot_modes()
    if k == len(modes):
        return modes[slot]
    return modes[min(len(modes) - 1, int(slot * len(modes) / k))]


def set_system(messages, text):
    if messages and messages[0].get('role') == 'system':
        messages[0]['content'] = text
    else:
        messages.insert(0, {'role': 'system', 'content': text})


def _flat_ids(ids):
    if not ids:
        return None
    if isinstance(ids[0], list):
        return [t for x in ids for t in x]
    return list(ids)


def credit_weights(tokenizer, sample, scores):
    ids = _flat_ids(sample.response_token_ids)
    if ids is None:
        content = sample.messages[-1].get('content') if sample.messages else None
        if not isinstance(content, str):
            return None
        ids = tokenizer(content, add_special_tokens=False)['input_ids']
    pieces = [tokenizer.decode([t]) for t in ids]
    starts = []
    pos = 0
    for p in pieces:
        starts.append(pos)
        pos += len(p)
    text = ''.join(pieces)
    m = THINK.search(text)
    if not m:
        return None
    ts, te = m.start(1), m.end(1)
    think = m.group(1)
    sp = spans(think)
    if not sp:
        return None
    if scores is None or len(scores) != len(sp):
        scores = heuristic_span_scores(think)
    raw = []
    inside = []
    j = 0
    for c in starts:
        if ts <= c < te:
            rel = c - ts
            while j + 1 < len(sp) and sp[j + 1][0] <= rel:
                j += 1
            raw.append(1.0 + scores[j] if sp[j][0] <= rel < sp[j][1] else 1.0)
            inside.append(True)
        else:
            raw.append(1.0)
            inside.append(False)
    if not any(inside):
        return None
    floor = float(os.environ.get('FLAIR_CREDIT_FLOOR', '0.05'))
    clip = float(os.environ.get('FLAIR_CREDIT_CLIP', '4.0'))
    raw = [max(r, floor) for r in raw]
    mean = sum(raw) / len(raw)
    w = [r / mean for r in raw]
    if clip > 0:
        w = [min(x, clip) for x in w]
    return w


def apply_replay_is(trainer, replay, samples, batch_encoded_inputs, mode):
    weights = []
    if not trainer.template.padding_free:
        for batch, enc in zip(trainer.split_by_mini_batches(samples), batch_encoded_inputs):
            gb = enc['grpo_batch']
            if gb.advantages is None or gb.old_per_token_logps is None:
                continue
            for row, s in enumerate(batch):
                if not s.extra.get('flair_replay'):
                    continue
                pos = gb.completion_mask[row].nonzero().squeeze(-1)
                cur = gb.old_per_token_logps[row, pos].tolist()
                w = replay.is_weight(cur, _flat(s.rollout_logprobs))
                gb.advantages[row] = gb.advantages[row] * w
                weights.append(w)
    stats = dict(replay.stats)
    allw = weights
    if allw:
        stats['replay_is_mean'] = sum(allw) / len(allw)
        stats['replay_is_clipped'] = sum(1.0 for w in allw if w in (replay.w_min, replay.w_max)) / len(allw)
    for key, val in stats.items():
        trainer._metrics[mode][f'flair/{key}'].append(float(val))


def allow_frame_list_videos():
    from swift.infer_engine import protocol
    for cls in (protocol.InferRequest, protocol.RolloutInferRequest):
        if 'videos' in getattr(cls, '__annotations__', {}):
            cls.__annotations__['videos'] = List[Any]
        if 'videos' in cls.__dataclass_fields__:
            cls.__dataclass_fields__['videos'].type = List[Any]


def scalar_video_fps():
    from swift.infer_engine.vllm_engine import VllmEngine
    if getattr(VllmEngine, '_flair_fps_patched', False):
        return
    orig = VllmEngine._add_request

    def _add_request(self, inputs, *args, **kwargs):
        mm = inputs.get('mm_processor_kwargs') or {}
        fps = mm.get('fps')
        if isinstance(fps, (list, tuple)) and fps and len(set(float(x) for x in fps)) == 1:
            inputs = dict(inputs, mm_processor_kwargs=dict(mm, fps=float(fps[0])))
        return orig(self, inputs, *args, **kwargs)

    VllmEngine._add_request = _add_request
    VllmEngine._flair_fps_patched = True


def apply_patch():
    allow_frame_list_videos()
    scalar_video_fps()
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    if getattr(GRPOTrainer, '_flair_patched', False):
        return
    orig_pre = GRPOTrainer._preprocess_inputs
    orig_post = GRPOTrainer._postprocess_batch
    orig_rollout = GRPOTrainer._rollout_samples

    def _rollout_samples(self, inputs):
        replay = get_replay()
        if not (self.model.training and replay.enabled):
            return orig_rollout(self, inputs)
        inputs = replay.mix(inputs)
        samples = orig_rollout(self, inputs)
        return replay.reuse_completions(samples)

    def _preprocess_inputs(self, samples):
        samples = orig_pre(self, samples)
        k = self.num_generations if self.model.training else self.num_generations_eval
        me = self.accelerator.process_index
        keys = [str(s.extra.get('sid') or s.prompt_id) for s in samples]
        everything = gather_object([(me, i, key) for i, key in enumerate(keys)])
        self._flair_round = getattr(self, '_flair_round', 0) + 1
        occ = {}
        mine = {}
        for r, i, key in everything:
            o = occ.get(key, 0)
            occ[key] = o + 1
            if r == me:
                mine[i] = (o % k, f'{key}#{self._flair_round}#{o // k}')
        for i, s in enumerate(samples):
            slot, gid = mine[i]
            mode = mode_for_slot(slot, k)
            s.extra['flair_slot'] = slot
            s.extra['flair_assigned'] = mode
            s.extra['flair_gid'] = gid
            set_system(s.messages, MODE_PROMPTS[mode])
        return samples

    def _postprocess_batch(self, samples, batch_encoded_inputs):
        orig_post(self, samples, batch_encoded_inputs)
        scorer = get_scorer()
        mode = 'train' if self.model.training else 'eval'
        for key, val in scorer.metrics.items():
            self._metrics[mode][f'flair/{key}'].append(float(val))
        replay = get_replay()
        if mode == 'train' and replay.enabled:
            apply_replay_is(self, replay, samples, batch_encoded_inputs, mode)
            replay.store(samples)
        if os.environ.get('FLAIR_TOKEN_CREDIT', '1') != '1' or self.template.padding_free:
            return
        tokenizer = getattr(self.processing_class, 'tokenizer', self.processing_class)
        chunks = self.split_by_mini_batches(samples)
        ratios = []
        for batch, enc in zip(chunks, batch_encoded_inputs):
            gb = enc['grpo_batch']
            if gb.advantages is None or gb.advantages.dim() != 2:
                continue
            for row, s in enumerate(batch):
                w = credit_weights(tokenizer, s, scorer.credit.get(s.request_id))
                if w is None:
                    continue
                pos = gb.completion_mask[row].nonzero().squeeze(-1)
                n = min(len(pos), len(w))
                if n == 0:
                    continue
                wt = torch.tensor(w[:n], device=gb.advantages.device, dtype=gb.advantages.dtype)
                gb.advantages[row, pos[:n]] = gb.advantages[row, pos[:n]] * wt
                ratios.append(float(wt.max()))
        if ratios:
            self._metrics[mode]['flair/credit_max'].append(sum(ratios) / len(ratios))

    GRPOTrainer._preprocess_inputs = _preprocess_inputs
    GRPOTrainer._rollout_samples = _rollout_samples
    GRPOTrainer._postprocess_batch = _postprocess_batch
    GRPOTrainer._flair_patched = True
