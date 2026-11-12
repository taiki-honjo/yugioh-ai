"""Basic Artmage opening group F0-F6: train/validation hands, seeds and engine-level plans (no engine import).

Plans are engine-level readings of the saved duel-simulator replays (manual board tool, not a rules engine) or,
for F2, a sequence built from card text (the linked replay starts from 襲撃, so the direct-Medius line is NOT
claimed to be shown in a replay). A plan is evidence only after the engine accepts every step.

  F0 襲撃 basic (old train 12; validation only here; old article plan)     replay g2rN57Xb
  F1 襲撃 + one trap cost (= C1)                                           replay wsMMwXxE
  F2 Medius (direct) + one spell/trap cost (C1 from the Medius summon on)  constructed from card text
  F3 学都 + one spell/trap cost (= C2)                                     replay XUS4azXa
  F4 学都 + two spell/trap costs                                           replay dAvrxdzV
  F5 獄神獣 + Fainmelt                                                     replay vxs2HMAV
  F6 テルミナス                                                            replay xXerEKuW
"メディウス＋モンスターコスト" stays out (heading only, no shown line).
"""
import collections

import artmage_teacher as T
from artmage_teacher import a, opt, group, rule, mark, opp

ASSAULT, MEDIUS, BEAST, MAIDEN, FIELD = T.ASSAULT, T.MEDIUS, T.BEAST, T.MAIDEN, T.FIELD
AWAKENING, GRAPH, VARNISH, LITERA, FINEMELT = T.AWAKENING, T.GRAPH, T.VARNISH, T.LITERA, T.FINEMELT
DIAKTOROS, NERVED, SHI, MABOROSHI, FANCY = T.DIAKTOROS, T.NERVED, T.SHI, T.MABOROSHI, T.FANCY
INFINITE, ASH, G, WARASHI, USAGI, DRIVER, GAMMA = T.INFINITE, T.ASH, T.G, T.WARASHI, T.USAGI, T.PSY_DRIVER, T.PSY_GAMMA
TERMINUS, VIDRIUM, IMPASTO, LITTLE_KNIGHT, FUNSAI, SANSEN = 25661743, 70488851, 44654994, 29301450, 97045737, 25311006
ADD_TO_HAND_OPTION = 1190  # !system 1190 手札に加える
FILLER_PRIORITY = [DRIVER, GAMMA, WARASHI, USAGI, G, ASH]  # Medius② cost when the Fainmelt draw is not a non-engine monster


def phase(step, value):
    return dict(step, _phase=value)


C1_PLAN = T.CONDITIONS['C1']['plan']
C2_PLAN = T.CONDITIONS['C2']['plan']
assert C1_PLAN[0] == a('Activate', ASSAULT) and C1_PLAN[3] == a('Summon', MEDIUS)
F2_PLAN = C1_PLAN[3:]

F4_PLAN = [
    a('Activate', FIELD), a('Activate', FIELD), opt(a('None', INFINITE)), opt(a('None', GRAPH)),
    a('Activate', GRAPH), a('Activate', GRAPH), a('None', ASSAULT),
    a('Activate', ASSAULT), a('Activate', ASSAULT), opt(a('None', MEDIUS)),
    a('Summon', MEDIUS), a('Activate', MEDIUS), a('None', NERVED), a('Activate', 0, effect=T.SPECIAL_SUMMON_OPTION),
    a('SpSummon', MAIDEN),
    group((a('Activate', MAIDEN), [a('None', IMPASTO)]), (a('Activate', NERVED), [opt(a('None', BEAST))])),
    a('Activate', BEAST, message='select_idlecmd'), opt(a('None', DIAKTOROS)),
    a('Activate', BEAST), a('None', VARNISH),
    a('SpSummon', SHI), a('None', DIAKTOROS), a('None', GRAPH),
    a('Activate', SHI), opt(a('None', MABOROSHI)),
    a('Activate', FIELD), opt(a('None', MABOROSHI)), opt(a('None', LITERA)),
    a('Activate', VARNISH), a('None', AWAKENING),
    a('Activate', FIELD), opt(a('None', FINEMELT)), opt(a('None', FINEMELT)),
    a('Activate', FINEMELT), mark(a('Activate', FINEMELT), 'before_draw'),
    a('Activate', MABOROSHI), opt(a('None', DIAKTOROS)),
    a('Activate', MEDIUS), rule('medius_cost', optional=True),
    a('Activate', LITERA), a('Activate', LITERA), a('None', AWAKENING),
    a('Set', AWAKENING),
    a('None', 0, phase='End'),
]

F5_PLAN = [
    a('Summon', BEAST),
    a('Activate', BEAST, message='select_idlecmd'), opt(a('None', MAIDEN)), opt(a('None', FINEMELT)),
    group((a('Activate', BEAST), [a('None', FIELD)]), (a('Activate', MAIDEN), [a('None', ASSAULT)])),
    a('Activate', ASSAULT), a('Activate', ASSAULT), opt(a('None', MEDIUS)),
    a('Activate', FIELD),
    a('Summon', MEDIUS), a('Activate', MEDIUS), a('None', NERVED), a('Activate', 0, effect=T.SPECIAL_SUMMON_OPTION),
    a('SpSummon', SHI), a('None', MEDIUS), a('None', NERVED),
    group((a('Activate', SHI), [opt(a('None', MABOROSHI))]), (a('Activate', NERVED), [a('None', FINEMELT)])),
    a('Activate', FIELD), opt(a('None', MABOROSHI)), opt(a('None', GRAPH)),
    a('Activate', GRAPH), a('Activate', GRAPH), a('None', VARNISH),
    a('Activate', VARNISH), a('None', AWAKENING),
    a('Activate', FIELD), opt(a('None', LITERA)),
    a('Activate', LITERA), a('Activate', LITERA), a('None', AWAKENING),
    a('Activate', MEDIUS), a('None', GRAPH),
    a('Set', AWAKENING),
    a('None', 0, phase='End'),
    # opponent turn (article: fuse with Maiden at the start of the opponent's turn; replay: 覚醒 → Medius → 獄神獣, 獄神使② → Little Knight)
    opp(phase(a('Activate', MAIDEN), 1)), opp(opt(a('None', DIAKTOROS))), opp(opt(a('None', MAIDEN))),
    opp(phase(a('Activate', AWAKENING), 4)), opp(a('None', MEDIUS)), opp(a('Activate', MEDIUS)),
    opp(opt(a('Activate', SHI))), opp(opt(a('None', LITTLE_KNIGHT))), opp(opt(a('None', SHI))), opp(opt(a('None', MEDIUS))),
    opp(a('None', BEAST)), opp(a('Activate', 0, effect=T.SPECIAL_SUMMON_OPTION)),
]

F6_PLAN = [
    a('Activate', TERMINUS), a('None', NERVED), a('Activate', TERMINUS), a('None', VIDRIUM),
    a('Activate', VIDRIUM), a('Activate', VIDRIUM), dict(act='Activate', effect=10011, message='select_option'), opt(a('None', NERVED)),
    a('Activate', VIDRIUM), a('None', BEAST),
    a('Summon', BEAST),
    a('Activate', BEAST, message='select_idlecmd'), opt(a('None', MAIDEN)), opt(a('None', NERVED)),
    group((a('Activate', BEAST), [a('None', ASSAULT)]), (a('Activate', MAIDEN), [a('None', FIELD)]), (a('Activate', NERVED), [a('None', FINEMELT)])),
    a('Activate', ASSAULT), a('Activate', ASSAULT), opt(a('None', MEDIUS)),
    a('Activate', FIELD),
    a('Summon', MEDIUS), a('Activate', MEDIUS), opt(a('None', BEAST)), a('Activate', 0, effect=ADD_TO_HAND_OPTION),
    a('SpSummon', SHI), a('None', MEDIUS), a('None', FINEMELT),
    a('Activate', SHI), opt(a('None', MABOROSHI)),
    a('Activate', FIELD), opt(a('None', MABOROSHI)), opt(a('None', GRAPH)),
    a('Activate', GRAPH), a('Activate', GRAPH), a('None', VARNISH),
    a('Activate', VARNISH), a('None', AWAKENING),
    a('Activate', FIELD), opt(a('None', LITERA)),
    a('Activate', LITERA), a('Activate', LITERA), a('None', FINEMELT),
    a('Activate', FINEMELT), mark(a('Activate', FINEMELT), 'before_draw'),
    a('Activate', MEDIUS), a('None', BEAST),
    a('None', 0, phase='End'),
]

CONDITIONS = {
    'F0': dict(label='襲撃の基本展開', reference='https://duel-simulator.miyauchidp.dev/replay/g2rN57Xb', plan='article_witness', train_reuse=['old train 12']),
    'F1': dict(label='襲撃＋罠コスト1枚', reference='https://duel-simulator.miyauchidp.dev/replay/wsMMwXxE', plan=C1_PLAN, train_reuse=['C1']),
    'F2': dict(label='メディウス（直接）＋魔法罠コスト1枚', reference='constructed from card text; replay wsMMwXxE starts from 襲撃', plan=F2_PLAN, train_reuse=[]),
    'F3': dict(label='学都＋魔法罠コスト1枚', reference='https://duel-simulator.miyauchidp.dev/replay/XUS4azXa', plan=C2_PLAN, train_reuse=['C2']),
    'F4': dict(label='学都＋魔法罠コスト2枚', reference='https://duel-simulator.miyauchidp.dev/replay/dAvrxdzV', plan=F4_PLAN, train_reuse=[]),
    'F5': dict(label='獄神獣＋ファインメルト', reference='https://duel-simulator.miyauchidp.dev/replay/vxs2HMAV', plan=F5_PLAN, train_reuse=[]),
    'F6': dict(label='テルミナス', reference='https://duel-simulator.miyauchidp.dev/replay/xXerEKuW', plan=F6_PLAN, train_reuse=[]),
}

TRAIN = {  # new train demonstrations (record + replay each)
    'F1-T1': dict(condition='F1', hand=[ASSAULT, INFINITE, ASH, USAGI, DRIVER], seed=1111001),
    'F2-T1': dict(condition='F2', hand=[MEDIUS, INFINITE, ASH, G, WARASHI], seed=1111002),
    'F2-T2': dict(condition='F2', hand=[MEDIUS, FUNSAI, USAGI, GAMMA, DRIVER], seed=1111003),
    'F3-T1': dict(condition='F3', hand=[FIELD, FUNSAI, ASH, G, WARASHI], seed=1111004),
    'F4-T1': dict(condition='F4', hand=[FIELD, INFINITE, INFINITE, ASH, G], seed=1111005),
    'F4-T2': dict(condition='F4', hand=[FIELD, INFINITE, FUNSAI, USAGI, WARASHI], seed=1111006),
    'F5-T1': dict(condition='F5', hand=[BEAST, FINEMELT, ASH, G, WARASHI], seed=1111007),
    'F5-T2': dict(condition='F5', hand=[BEAST, FINEMELT, USAGI, GAMMA, DRIVER], seed=1111008),
    'F6-T1': dict(condition='F6', hand=[TERMINUS, ASH, G, WARASHI, USAGI], seed=1111009),
    'F6-T2': dict(condition='F6', hand=[TERMINUS, ASH, GAMMA, DRIVER, WARASHI], seed=1111010),
}
VALIDATION = {  # never readable by the training loader
    'F0-V': dict(condition='F0', hand=[ASSAULT, USAGI, G, GAMMA, ASH], seed=1111101),
    'F1-V': dict(condition='F1', hand=[ASSAULT, FUNSAI, ASH, WARASHI, DRIVER], seed=1111102),
    'F2-V': dict(condition='F2', hand=[MEDIUS, INFINITE, USAGI, WARASHI, DRIVER], seed=1111103),
    'F3-V': dict(condition='F3', hand=[FIELD, INFINITE, USAGI, GAMMA, WARASHI], seed=1111104),
    'F4-V': dict(condition='F4', hand=[FIELD, FUNSAI, FUNSAI, ASH, DRIVER], seed=1111105),
    'F5-V': dict(condition='F5', hand=[BEAST, FINEMELT, ASH, WARASHI, DRIVER], seed=1111106),
    'F6-V': dict(condition='F6', hand=[TERMINUS, G, GAMMA, USAGI, DRIVER], seed=1111107),
}
PAST_EVALUATION_HANDS = {  # earlier evaluation hands that must not move into train
    'C1v': [ASSAULT, FUNSAI, USAGI, GAMMA, DRIVER], 'C2v': [FIELD, SANSEN, USAGI, GAMMA, DRIVER],
}


class BatchTeacher(T.Teacher):
    """Teacher for the batch: '_phase' steps match only in that phase; a choice between identical copies (same code,
    same zone) is resolved to the first; the Medius② cost rule uses FILLER_PRIORITY."""

    def _find(self, step, s):
        if '_phase' in step and s['phase'] != step['_phase']:
            return None
        return super()._find(step, s)

    def _rule(self, name, s):
        if name != 'medius_cost' or s['message'] not in ['select_card', 'select_unselect_card']:
            return None
        candidates = {x['code']: x['index'] for x in s['actions'] if x['code'] and str(x.get('spec', '')).startswith('h')}
        if not candidates:
            return None
        before = collections.Counter(self.marks.get('before_draw', {}))
        for code in (T.own_hand(s) - before):
            if code in T.NON_ENGINE_MONSTERS and code in candidates:
                return candidates[code]
        return next((candidates[c] for c in FILLER_PRIORITY if c in candidates), None)

    def decide(self, s):
        acts, msg = s['actions'], s['message']
        found = self._match(s) if self.steps or self.pending else None
        if found:
            idx, why = found
            if acts[idx]['act'] in ['Summon', 'SpSummon', 'Activate'] and acts[idx]['code']:
                self.summon_context = acts[idx]['code']
            return idx, why
        head = self.head()
        if msg in ['select_chain', 'select_effectyn', 'select_yesno']:
            if head is not None and s['turn_player'] == 0:
                self.unmatched_chain_heads.append(dict(turn=s['turn_count'], message=msg, head=head,
                                                       offered=[(x['act'], x['code'], x.get('effect')) for x in acts]))
            return T.passive(s), dict(kind='default-pass')
        if msg in ['select_place', 'select_disfield']:
            return self._place(s), dict(kind='default-place')
        if msg == 'select_position':
            return next((x['index'] for x in acts if x['position'] == 1), 0), dict(kind='default-position')
        if len(acts) == 1:
            return 0, dict(kind='only-option')
        if msg in ['select_card', 'select_unselect_card', 'select_tribute']:
            options = [x for x in acts if not x['finish']]
            if len(options) > 1 and options[0]['code'] and len({(x['code'], str(x.get('spec', ''))[:1]) for x in options}) == 1:
                return options[0]['index'], dict(kind='identical-copies')
            if any(x['finish'] for x in acts):
                return next(x['index'] for x in acts if x['finish']), dict(kind='default-finish')
        if msg == 'select_option':
            # amendment 1: summon method — 「神芸学都アルトメギア」の効果で召喚する (as in replay dAvrxdzV) over 通常召喚
            field = next((x['index'] for x in acts if x['code'] == FIELD and x.get('effect') == 10010), None)
            if field is not None:
                return field, dict(kind='summon-method-field')
        if head is None and msg == 'select_idlecmd':
            return T.passive(s), dict(kind='default-pass')
        if s['turn_player'] != 0 or s['phase'] != 4:
            return T.passive(s), dict(kind='default-pass')
        raise T.Unresolved(f'planned step not legal at {msg}: {head}')


def hand_key(hand):
    return tuple(sorted(hand))
