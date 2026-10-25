import heapq
import json
import os
import time

import torch.distributed as dist

from . import answers
from .parsing import has_ground_tags, parse, plain_words, span_score_mean
from .prompts import MODES
from .verifier import VerifierClient, heuristic_verdict, span_texts

COMPONENTS = ('ans', 'format', 'comply', 'cost', 'balance', 'select', 'verifier')
DEFAULT_WEIGHTS = {'ans': 1.0, 'format': 0.25, 'comply': 0.6, 'cost': 0.2, 'balance': 0.2, 'select': 0.4, 'verifier': 0.35}
HARD_LIMIT = {'DIRECT': 120, 'CONCISE': 600, 'DEEP': None}
BASE_COST = {'DIRECT': 0.0, 'CONCISE': 0.08, 'DEEP': 0.20}
MU = {'DIRECT': 1.0, 'CONCISE': 1.1, 'DEEP': 1.25, 'ADAPTIVE': 1.1}
CFG = {
    'K': 8,
    'overflow_scale': 0.25,
    'r_ok': 0.5,
    'lambda_L': 0.5,
    'alpha_w': 0.25,
    'beta_alp': 0.2,
    'L0': 512.0,
    'delta': 0.35,
    'select_bonus': 0.8,
    'balance_target': 2.0,
    'balance_bias': 0.2,
    'w_t': 0.35,
    'w_s': 0.35,
    'w_h': 0.30,
    'w_span': 0.25,
    'tag_penalty': -0.6,
    'pref_bonus': 0.4,
    'replay_gamma': 0.15,
    'replay_capacity': 4096,
    'verifier_enable_step': 100,
    'solved_threshold': 0.999,
    'pair_margin': 0.15,
    'max_fail_rate': 0.5,
    'max_bad_batches': 5,
}


def _cfg_from_env():
    for k, v in list(CFG.items()):
        e = os.environ.get('FLAIR_' + k.upper())
        if e not in (None, ''):
            CFG[k] = type(v)(float(e)) if isinstance(v, (int, float)) else e
    w = os.environ.get('FLAIR_REWARD_WEIGHTS')
    if w:
        for k, x in zip(COMPONENTS, w.split(',')):
            DEFAULT_WEIGHTS[k] = float(x)


_cfg_from_env()


def gather(obj):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        out = [None] * dist.get_world_size()
        dist.all_gather_object(out, obj)
        return out
    return [obj]


def rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = {}
        self.heap = []

    def get(self, key):
        e = self.entries.get(key)
        return e['priority'] if e else 0.0

    def update(self, key, priority, mode, hardness):
        e = self.entries.get(key)
        if e is not None and priority <= e['priority']:
            return
        self.entries[key] = {'priority': priority, 'mode': mode, 'hardness': hardness, 'time': time.time()}
        heapq.heappush(self.heap, (priority, key))
        while len(self.entries) > self.capacity and self.heap:
            p, k = heapq.heappop(self.heap)
            if k in self.entries and self.entries[k]['priority'] == p:
                del self.entries[k]

    def state_dict(self):
        return dict(self.entries)


def _question(messages):
    for m in messages or []:
        if m.get('role') == 'user':
            c = m.get('content')
            if isinstance(c, list):
                c = ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
            return (c or '').replace('<video>', '').replace('<image>', '').strip()
    return ''


def _ntokens(ids, text):
    if ids:
        if isinstance(ids[0], list):
            return sum(len(x) for x in ids)
        return len(ids)
    return max(1, int(len((text or '').split()) * 1.3))


class FlairScorer:

    def __init__(self):
        self.verifier = VerifierClient()
        self.replay = ReplayBuffer(int(CFG['replay_capacity']))
        self.cache_key = None
        self.cache = None
        self.credit = {}
        self.metrics = {}
        self.bad_batches = 0
        self.pair_dir = os.environ.get('FLAIR_PAIR_DIR', '')

    def components(self, completions, **kw):
        key = tuple(kw.get('request_id') or [])
        if key and key == self.cache_key:
            return self.cache
        out = self._compute(completions, **kw)
        self.cache_key, self.cache = key, out
        return out

    def _compute(self, completions, **kw):
        n = len(completions)
        state = kw.get('trainer_state')
        step = int(getattr(state, 'global_step', 0) or 0)
        slots = kw.get('flair_slot') or [6] * n
        assigned = kw.get('flair_assigned') or ['ADAPTIVE'] * n
        gids = kw.get('flair_gid') or [str(i) for i in range(n)]
        sids = kw.get('sid') or gids
        sols = kw.get('solution') or [''] * n
        atypes = kw.get('answer_type') or [None] * n
        msgs = kw.get('messages') or [[]] * n
        rids = kw.get('request_id') or [str(i) for i in range(n)]
        tok_ids = kw.get('response_token_ids') or [None] * n
        images = kw.get('images') or [None] * n
        videos = kw.get('videos') or [None] * n

        recs = []
        for i, c in enumerate(completions):
            text = c if isinstance(c, str) else str(c)
            p = parse(text)
            adaptive = assigned[i] == 'ADAPTIVE'
            produced = p['mode']
            eff = produced if (p['valid'] and produced in MODES) else None
            w = plain_words(p['think'])
            limit = HARD_LIMIT.get(eff) if eff else None
            overflow = max(0.0, (w - limit) / limit) if limit else 0.0
            r_ans = answers.score(p['answer'], sols[i], atypes[i]) if p['answer'] else 0.0
            r_format = 1.0 if p['valid'] else 0.0
            mode_ok = eff is not None and (adaptive or eff == assigned[i])
            r_comply = (CFG['r_ok'] - CFG['lambda_L'] * overflow) if mode_ok else -2.0
            cost_eff = BASE_COST[eff or 'DEEP'] + CFG['overflow_scale'] * overflow
            recs.append({
                'i': i,
                'gid': gids[i],
                'sid': sids[i],
                'slot': int(slots[i]),
                'assigned': assigned[i],
                'produced': eff,
                'mode_ok': mode_ok,
                'r_ans': r_ans,
                'r_format': r_format,
                'r_comply': r_comply,
                'cost_eff': cost_eff,
                'ntok': _ntokens(tok_ids[i], text),
                'think': p['think'],
                'answer': p['answer'],
                'text': text,
                'tags': has_ground_tags(p['think']),
            })

        use_verifier = self.verifier.enabled and step >= int(CFG['verifier_enable_step'])
        verdicts = [None] * n
        if use_verifier:
            items = []
            for r in recs:
                st = span_texts(r['think'])
                items.append((self.verifier.build_messages(
                    _question(msgs[r['i']]), r['text'], sols[r['i']], st, images[r['i']], videos[r['i']]), len(st)))
            verdicts = self.verifier.score(items)
            fails = sum(v is None for v in verdicts)
            rate = fails / max(1, n)
            self.metrics['verifier_fail_rate'] = rate
            all_rates = gather(rate)
            if max(all_rates) > CFG['max_fail_rate']:
                self.bad_batches += 1
                if self.bad_batches >= int(CFG['max_bad_batches']):
                    raise RuntimeError(f'verifier unreachable or unparsable for {self.bad_batches} consecutive batches')
            else:
                self.bad_batches = 0
        for r, v in zip(recs, verdicts):
            h = heuristic_verdict(r['think'])
            v = v or h
            if v.get('span_scores') is None:
                v = dict(v, span_scores=h['span_scores'])
            r['verdict'] = v
            r['v_t'] = v['temporal_grounding']
            r['v_s'] = v['spatial_grounding']
            r['v_h'] = v['human_alignment']
            r['s_span'] = span_score_mean(r['think'], v['span_scores'])
            r['p_old'] = self.replay.get(r['sid'])
            adaptive = r['assigned'] == 'ADAPTIVE'
            d_tag = CFG['tag_penalty'] if (r['produced'] == 'DIRECT' and r['tags']) else 0.0
            pref = v.get('preferred_mode')
            d_pref = (CFG['pref_bonus'] if r['produced'] == pref else -CFG['pref_bonus']) if (adaptive and pref) else 0.0
            d_replay = CFG['replay_gamma'] * min(r['p_old'] / 2.0, 1.0)
            r['s_ver'] = (CFG['w_t'] * r['v_t'] + CFG['w_s'] * r['v_s'] + CFG['w_h'] * r['v_h'] +
                          CFG['w_span'] * r['s_span'] + d_tag + d_pref + d_replay)

        keys = ('gid', 'sid', 'slot', 'assigned', 'produced', 'mode_ok', 'r_ans', 'cost_eff', 'v_t', 'v_s', 's_ver')
        world = gather([{k: r[k] for k in keys} for r in recs])
        groups = {}
        for part in world:
            for r in part:
                groups.setdefault(r['gid'], []).append(r)
        gstats = {gid: self._group_stats(rows) for gid, rows in groups.items()}
        if use_verifier:
            for part in world:
                for r in part:
                    g = gstats[r['gid']]
                    self.replay.update(r['sid'], abs(r['s_ver']) + (1 - r['r_ans']), r['produced'], 1 - g['p_solved'])

        out = {k: [] for k in COMPONENTS}
        composite = []
        for r in recs:
            g = gstats[r['gid']]
            adaptive = r['assigned'] == 'ADAPTIVE'
            if adaptive:
                b = 0.0
            elif r['produced']:
                t = CFG['balance_target']
                b = max(-1.0, min(1.0, CFG['balance_bias'] + (t - g['counts'].get(r['produced'], 0)) / t))
            else:
                b = -CFG['balance_bias']
            if not adaptive:
                sel = 0.0
            elif r['produced'] is None:
                sel = -CFG['select_bonus']
            elif g['m_star'] is None:
                sel = 0.0
            else:
                sel = CFG['select_bonus'] if r['produced'] == g['m_star'] else -CFG['select_bonus']
            mu = MU['ADAPTIVE'] if adaptive else MU.get(r['assigned'], 1.0)
            p_base = r['cost_eff'] * (CFG['alpha_w'] + (1 - CFG['alpha_w']) * r['r_ans'])
            p_alp = CFG['beta_alp'] * (r['ntok'] / CFG['L0']) * g['p_solved'] * mu
            vals = {
                'ans': r['r_ans'],
                'format': r['r_format'],
                'comply': r['r_comply'],
                'cost': -(p_base + p_alp),
                'balance': b,
                'select': sel,
                'verifier': max(-2.0, min(2.0, r['s_ver'])) if use_verifier else 0.0,
            }
            for k in COMPONENTS:
                out[k].append(vals[k])
            r['vals'] = vals
            r['composite'] = sum(DEFAULT_WEIGHTS[k] * vals[k] for k in COMPONENTS)
            composite.append(r['composite'])
            self.credit[rids[r['i']]] = r['verdict']['span_scores']

        self._log(recs, gstats, composite)
        if self.pair_dir:
            self._dump_pairs(recs, gstats, msgs, sols, images, videos, step)
        if len(self.credit) > 50000:
            self.credit = dict(list(self.credit.items())[-20000:])
        return out

    def _group_stats(self, rows):
        K = len(rows)
        solved = sum(r['r_ans'] >= CFG['solved_threshold'] for r in rows) / max(1, K)
        p_solved = max(1.0 / max(1, K), solved)
        counts = {}
        for r in rows:
            if r['produced'] and r['assigned'] != 'ADAPTIVE':
                counts[r['produced']] = counts.get(r['produced'], 0) + 1
        util = {}
        for m in MODES:
            ctl = [r for r in rows if r['assigned'] == m and r['mode_ok']]
            if not ctl:
                continue
            a = sum(r['r_ans'] for r in ctl) / len(ctl)
            gg = sum((r['v_t'] + r['v_s']) / 2 for r in ctl) / len(ctl)
            c = sum(r['cost_eff'] for r in ctl) / len(ctl)
            util[m] = (a + CFG['delta'] * gg) * (1 - c)
        m_star = None
        if util:
            best = max(util.values())
            for m in MODES:
                if m in util and abs(util[m] - best) <= 1e-9:
                    m_star = m
                    break
        return {'p_solved': p_solved, 'counts': counts, 'util': util, 'm_star': m_star}

    def _log(self, recs, gstats, composite):
        n = max(1, len(recs))
        ad = [r for r in recs if r['assigned'] == 'ADAPTIVE']
        m = self.metrics
        m['composite'] = sum(composite) / n
        m['format_ok'] = sum(r['r_format'] for r in recs) / n
        m['mode_ok'] = sum(r['mode_ok'] for r in recs) / n
        for mode in MODES:
            m[f'adaptive_{mode.lower()}'] = sum(r['produced'] == mode for r in ad) / max(1, len(ad))
            stars = [g['m_star'] for g in gstats.values()]
            m[f'mstar_{mode.lower()}'] = sum(s == mode for s in stars) / max(1, len(stars))
            ctl = [r for r in recs if r['assigned'] == mode]
            if ctl:
                m[f'ans_{mode.lower()}'] = sum(r['r_ans'] for r in ctl) / len(ctl)
                m[f'ntok_{mode.lower()}'] = sum(r['ntok'] for r in ctl) / len(ctl)
        if ad:
            m['adaptive_select_match'] = sum(r['vals']['select'] > 0 for r in ad) / len(ad)
            m['ans_adaptive'] = sum(r['r_ans'] for r in ad) / len(ad)
        m['replay_size'] = len(self.replay.entries)

    def _dump_pairs(self, recs, gstats, msgs, sols, images, videos, step):
        local = []
        for r in recs:
            i = r['i']
            local.append({
                'gid': r['gid'],
                'composite': r['composite'],
                'completion': r['text'],
                'verdict': r['verdict'],
                'mode': r['produced'],
                'question': _question(msgs[i]),
                'solution': sols[i],
                'images': images[i],
                'videos': videos[i],
            })
        world = gather(local)
        if rank() != 0:
            return
        by = {}
        for part in world:
            for r in part:
                by.setdefault(r['gid'], []).append(r)
        lines = []
        for gid, rows in by.items():
            if len(rows) < 2:
                continue
            hi = max(rows, key=lambda r: r['composite'])
            lo = min(rows, key=lambda r: r['composite'])
            if hi['composite'] - lo['composite'] <= CFG['pair_margin']:
                continue
            lines.append(json.dumps({
                'step': step,
                'gid': gid,
                'question': hi['question'],
                'solution': hi['solution'],
                'images': hi['images'],
                'videos': hi['videos'],
                'm_star': gstats[gid]['m_star'],
                'hi': {k: hi[k] for k in ('completion', 'verdict', 'composite', 'mode')},
                'lo': {k: lo[k] for k in ('completion', 'verdict', 'composite', 'mode')},
            }, ensure_ascii=False))
        os.makedirs(self.pair_dir, exist_ok=True)
        with open(os.path.join(self.pair_dir, 'state.json.tmp'), 'w') as f:
            json.dump({'step': step, 'time': time.time()}, f)
        os.replace(os.path.join(self.pair_dir, 'state.json.tmp'), os.path.join(self.pair_dir, 'state.json'))
        if lines:
            with open(os.path.join(self.pair_dir, 'pairs.jsonl'), 'a') as f:
                f.write('\n'.join(lines) + '\n')


SCORER = None


def get_scorer():
    global SCORER
    if SCORER is None:
        SCORER = FlairScorer()
    return SCORER
