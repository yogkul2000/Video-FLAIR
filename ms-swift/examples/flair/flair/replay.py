import copy
import math
import os
import random

from accelerate.utils import gather_object

from .scorer import get_scorer


def _env(name, default):
    return float(os.environ.get(name, default))


def _flat(x):
    if not x:
        return []
    if isinstance(x[0], list):
        return [t for y in x for t in y]
    return list(x)


class Replay:

    def __init__(self):
        self.frac = _env('FLAIR_REPLAY_FRAC', '0.125')
        self.reuse = _env('FLAIR_REPLAY_REUSE', '0.5')
        self.w_min = _env('FLAIR_REPLAY_IS_MIN', '0.5')
        self.w_max = _env('FLAIR_REPLAY_IS_MAX', '2.0')
        self.cooldown = int(_env('FLAIR_REPLAY_COOLDOWN', '20'))
        self.max_visits = int(_env('FLAIR_REPLAY_MAX_VISITS', '2'))
        self.store_top = int(_env('FLAIR_REPLAY_STORE', '512'))
        self.rows = {}
        self.completions = {}
        self.visits = {}
        self.last = {}
        self.active = set()
        self.round = 0
        self.stats = {}

    @property
    def enabled(self):
        return self.frac > 0

    def _buffer(self):
        return get_scorer().replay.entries

    def _top(self):
        buf = self._buffer()
        ranked = sorted(buf.items(), key=lambda kv: (-kv[1]['priority'], kv[0]))
        return {k for k, _ in ranked[:self.store_top]}

    def mix(self, inputs):
        self.round += 1
        self.active = set()
        self.stats = {'replay_prompts': 0.0}
        local = {}
        for r in inputs:
            sid = r.get('sid')
            if sid is None:
                return inputs
            local.setdefault(str(sid), r)
        everything = gather_object([(k, copy.deepcopy(v)) for k, v in local.items()])
        keys = []
        for sid, row in everything:
            if sid not in self.rows:
                self.rows[sid] = row
            if sid not in keys:
                keys.append(sid)
        buf = self._buffer()
        if len(self.rows) > 4 * max(len(buf), 1) + len(keys):
            keep = set(buf) | set(keys)
            self.rows = {k: v for k, v in self.rows.items() if k in keep}
        if not buf:
            return inputs
        rng = random.Random(self.round)
        target = self.frac * len(keys)
        n = int(target) + (1 if rng.random() < target - int(target) else 0)
        present = set(keys)
        cands = sorted(
            ((e['priority'], s) for s, e in buf.items()
             if s in self.rows and s not in present and self.visits.get(s, 0) < self.max_visits
             and self.round - self.last.get(s, -10**9) >= self.cooldown),
            key=lambda x: (-x[0], x[1]))
        n = min(n, len(cands), len(keys))
        if n <= 0:
            return inputs
        victims = rng.sample(sorted(keys), n)
        mapping = {v: s for v, (_, s) in zip(victims, cands[:n])}
        for s in mapping.values():
            self.visits[s] = self.visits.get(s, 0) + 1
            self.last[s] = self.round
        self.active = set(mapping.values())
        self.stats['replay_prompts'] = float(n)
        return [copy.deepcopy(self.rows[mapping[str(r.get('sid'))]]) if str(r.get('sid')) in mapping else r
                for r in inputs]

    def _chosen(self, sid, slot):
        if self.reuse >= 1:
            return True
        return random.Random(f'{sid}:{slot}:{self.round}').random() < self.reuse

    def reuse_completions(self, samples):
        want = []
        for s in samples:
            sid = str(s.extra.get('sid'))
            slot = s.extra.get('flair_slot')
            if sid in self.active and slot is not None and self._chosen(sid, slot):
                want.append((sid, int(slot)))
        all_want = gather_object(want)
        have = []
        for sid, slot in all_want:
            c = self.completions.get(sid, {}).get(slot)
            if c is not None:
                have.append(((sid, slot), c))
        pool = dict(gather_object(have))
        wanted = set(want)
        used = 0
        for s in samples:
            slot = s.extra.get('flair_slot')
            key = (str(s.extra.get('sid')), int(slot) if slot is not None else None)
            if key not in pool or key not in wanted:
                continue
            c = pool[key]
            if not s.messages or s.messages[-1].get('role') != 'assistant':
                continue
            s.messages[-1]['content'] = c['text']
            s.response_token_ids = copy.deepcopy(c['ids'])
            s.rollout_logprobs = copy.deepcopy(c['lps'])
            s.response_loss_mask = []
            s.finish_reason = c['finish']
            s.extra['flair_replay'] = 1
            used += 1
        self.stats['replay_reused'] = float(sum(gather_object([used])))
        return samples

    def store(self, samples):
        top = self._top()
        for s in samples:
            sid = str(s.extra.get('sid'))
            slot = s.extra.get('flair_slot')
            if s.extra.get('flair_replay') or sid not in top or slot is None:
                continue
            if not s.rollout_logprobs or not s.response_token_ids:
                continue
            if not s.messages or s.messages[-1].get('role') != 'assistant':
                continue
            self.completions.setdefault(sid, {})[int(slot)] = {
                'text': s.messages[-1].get('content'),
                'ids': copy.deepcopy(s.response_token_ids),
                'lps': copy.deepcopy(s.rollout_logprobs),
                'finish': s.finish_reason,
            }
        for sid in [k for k in self.completions if k not in top]:
            del self.completions[sid]

    def is_weight(self, current_logps, behavior_logps):
        n = min(len(current_logps), len(behavior_logps))
        if n == 0:
            return 1.0
        d = sum(float(current_logps[i]) - float(behavior_logps[i]) for i in range(n)) / n
        w = math.exp(max(-20.0, min(20.0, d)))
        return min(self.w_max, max(self.w_min, w))


_REPLAY = None


def get_replay():
    global _REPLAY
    if _REPLAY is None:
        _REPLAY = Replay()
    return _REPLAY
