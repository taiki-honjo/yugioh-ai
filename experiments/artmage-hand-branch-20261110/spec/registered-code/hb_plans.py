"""Paired hands for the hand-dependent branches (no engine import).

Group A (襲撃 start): the only difference inside a pair is one card — a non-engine monster (no spell/trap cost
kept) vs a spell/trap cost. Demonstrations verified in this condition:
  A-nocost : the old 襲撃 route (Medius① → 獄神獣, 獄神獣② fuses Medius + 獄神獣 into Maiden; article plan g2rN57Xb,
             rewritten for the batch teacher with the Medius② cost rule)
  A-cost   : the C1 route (Medius① → ネルヴェド, Maiden by discarding the cost; replay wsMMwXxE)
Group B (学都 start): one monster vs a second spell/trap cost.
  B-1cost  : the C2 route (Graphrea sets 改変; replay XUS4azXa)
  B-2cost  : the F4 route (Graphrea sets 襲撃; replay dAvrxdzV; amended batch plan)
Both routes are legal in some of these hands; a demo is "the route verified in this condition", not the only correct
play. No rule that maps a count to a choice is given to the model.
"""
import collections

import artmage_teacher as T
import batch_plans as BP
from artmage_teacher import a, opt, group, rule, mark, opp

ASSAULT, MEDIUS, BEAST, MAIDEN, FIELD = BP.ASSAULT, BP.MEDIUS, BP.BEAST, BP.MAIDEN, BP.FIELD
AWAKENING, GRAPH, VARNISH, LITERA, FINEMELT, CROSS = BP.AWAKENING, BP.GRAPH, BP.VARNISH, BP.LITERA, BP.FINEMELT, T.CROSS
DIAKTOROS, NERVED = BP.DIAKTOROS, BP.NERVED
INFINITE, FUNSAI, SANSEN = BP.INFINITE, BP.FUNSAI, BP.SANSEN
ASH, G, WARASHI, USAGI, DRIVER, GAMMA = BP.ASH, BP.G, BP.WARASHI, BP.USAGI, BP.DRIVER, BP.GAMMA
COST_CARDS = {INFINITE, FUNSAI, SANSEN}
COST_ORDER = [INFINITE, SANSEN, FUNSAI]  # teacher only: which cost to discard first when two different ones are held

OLD_ROUTE = [
    a('Activate', ASSAULT), a('Activate', ASSAULT), opt(a('None', MEDIUS)),
    a('Summon', MEDIUS), a('Activate', MEDIUS), a('None', BEAST), a('Activate', 0, effect=T.SPECIAL_SUMMON_OPTION),
    a('Activate', ASSAULT),
    a('Activate', BEAST, message='select_idlecmd'), opt(a('None', MAIDEN)),
    group((a('Activate', MAIDEN), [a('None', FIELD)]), (a('Activate', BEAST), [a('None', AWAKENING)])),
    a('Activate', FIELD), a('Activate', FIELD), opt(a('None', GRAPH)),
    a('Activate', GRAPH), a('Activate', GRAPH), a('None', VARNISH),
    a('Activate', VARNISH), a('None', ASSAULT),
    a('Activate', FIELD), opt(a('None', LITERA)),
    a('Activate', LITERA), a('Activate', LITERA), a('None', ASSAULT),
    a('Activate', FIELD), opt(a('None', FINEMELT)), opt(a('None', FINEMELT)),
    a('SpSummon', CROSS), a('None', GRAPH), a('None', LITERA),
    a('Activate', FINEMELT), mark(a('Activate', FINEMELT), 'before_draw'),
    a('Activate', CROSS), opt(a('None', LITERA)),
    a('Activate', MEDIUS), rule('medius_cost', optional=True),
    a('None', 0, phase='End'),
    opp(a('Activate', MAIDEN)), opp(opt(a('None', DIAKTOROS))), opp(opt(a('None', MAIDEN))), opp(a('Activate', CROSS)), opp(opt(a('None', BEAST))),
]
PLANS = {'A-nocost': OLD_ROUTE, 'A-cost': BP.C1_PLAN, 'B-1cost': BP.C2_PLAN, 'B-2cost': BP.F4_PLAN}
ROUTE_LABEL = {'A-nocost': '旧来の襲撃（獄神獣経由）', 'A-cost': 'C1（ネルヴェド・乙女）', 'B-1cost': 'C2（グラフレアで改変）', 'B-2cost': 'F4（グラフレアで襲撃）'}


def _pair(pid, group_, seed, base, variants, split):
    return dict(pair=pid, group=group_, seed=seed, split=split,
                hands={v: list(base) + [card] for v, card in variants.items()}, differing={v: card for v, card in variants.items()})


PAIRS = [
    _pair('A1', 'A', 1112001, [ASSAULT, ASH, ASH, G], {'A-nocost': DRIVER, 'A-cost': FUNSAI}, 'train'),
    _pair('A2', 'A', 1112002, [ASSAULT, WARASHI, GAMMA, USAGI], {'A-nocost': ASH, 'A-cost': SANSEN}, 'train'),
    _pair('A3', 'A', 1112003, [ASSAULT, ASH, WARASHI, DRIVER], {'A-nocost': GAMMA, 'A-cost': INFINITE}, 'train'),
    _pair('A4', 'A', 1112004, [ASSAULT, ASH, G, USAGI], {'A-nocost': WARASHI, 'A-cost': INFINITE}, 'train'),
    _pair('B1', 'B', 1112005, [FIELD, INFINITE, G, DRIVER], {'B-1cost': USAGI, 'B-2cost': SANSEN}, 'train'),
    _pair('B2', 'B', 1112006, [FIELD, FUNSAI, ASH, GAMMA], {'B-1cost': WARASHI, 'B-2cost': FUNSAI}, 'train'),
    _pair('B3', 'B', 1112007, [FIELD, SANSEN, WARASHI, USAGI], {'B-1cost': ASH, 'B-2cost': INFINITE}, 'train'),
    _pair('B4', 'B', 1112008, [FIELD, INFINITE, ASH, ASH], {'B-1cost': GAMMA, 'B-2cost': INFINITE}, 'train'),
    _pair('AV1', 'A', 1112101, [ASSAULT, ASH, GAMMA, WARASHI], {'A-nocost': G, 'A-cost': FUNSAI}, 'validation'),
    _pair('AV2', 'A', 1112102, [ASSAULT, USAGI, DRIVER, ASH], {'A-nocost': WARASHI, 'A-cost': SANSEN}, 'validation'),
    _pair('BV1', 'B', 1112103, [FIELD, FUNSAI, G, USAGI], {'B-1cost': DRIVER, 'B-2cost': INFINITE}, 'validation'),
    _pair('BV2', 'B', 1112104, [FIELD, INFINITE, WARASHI, DRIVER], {'B-1cost': G, 'B-2cost': FUNSAI}, 'validation'),
]
HANDS = {f"{p['pair']}-{v}": dict(pair=p['pair'], group=p['group'], variant=v, seed=p['seed'], split=p['split'], hand=h, plan=v)
         for p in PAIRS for v, h in p['hands'].items()}
SUBSTITUTE = {  # one-move diagnostics on the earlier validation hands (batch root records)
    'F1-V': 'Medius①: ネルヴェド instead of 獄神獣', 'F3-V': 'Graphrea set: 改変 instead of 襲撃', 'F4-V': 'Graphrea set: 襲撃 instead of 改変',
}


class BranchTeacher(BP.BatchTeacher):
    """BatchTeacher + a cost-discard default: a selection whose options are only spell/trap costs in hand and that
    no plan step covers is answered with COST_ORDER."""

    def decide(self, s):
        try:
            return super().decide(s)
        except T.Unresolved:
            options = [x for x in s['actions'] if not x['finish']]
            if (s['message'] in ('select_card', 'select_unselect_card') and options
                    and all(x['code'] in COST_CARDS and str(x.get('spec', '')).startswith('h') for x in options)):
                for code in COST_ORDER:
                    idx = next((x['index'] for x in options if x['code'] == code), None)
                    if idx is not None:
                        return idx, dict(kind='cost-order')
            raise


def branch_steps(seq, ids):
    """Steps of the two decisions under study, from the raw observation (audit/diagnostics only, never a model input).
    A: a selection offering both ネルヴェド and 獄神獣 (the Medius① choice). B: a selection offering 改変 and 襲撃
    with no trap among the options (Graphrea sets a spell; 獄神獣③ and Maiden① also offer traps)."""
    import numpy as np
    out = []
    for t in range(len(seq['labels'])):
        rows = seq['actions_'][t]
        valid = np.where(rows[:, 3] != 0)[0]
        offered = {ids.get(int(rows[i, 1]) * 256 + int(rows[i, 2]), 0) for i in valid}
        chosen = ids.get(int(rows[int(seq['labels'][t]), 1]) * 256 + int(rows[int(seq['labels'][t]), 2]), 0)
        if {NERVED, BEAST} <= offered and chosen in (NERVED, BEAST) and seq['message'][t] != 11:
            out.append(dict(step=t, branch='A: Medius①', label=chosen))
        if {VARNISH, ASSAULT} <= offered and not offered & {AWAKENING, BP.IMPASTO} and chosen in (VARNISH, ASSAULT) and seq['message'][t] != 11:
            out.append(dict(step=t, branch='B: Graphrea set', label=chosen))
    return out


def hand_key(hand):
    return tuple(sorted(hand))
