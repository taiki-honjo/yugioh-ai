"""Scripted teacher and fixed plans for the two new Artmage opening demonstrations (no engine import here).

Plans were transcribed before any engine run from the public reference replays (duel-simulator save-states
wsMMwXxE and XUS4azXa; card identities from the replay's deck image, whose 40/15 order equals the pinned
recipe). The simulator is a manual board tool, not a rules engine, so the plan below is the engine-level
reading of those moves; it is evidence only after the engine accepts every step.

Plan entries:
  a(act, code, **keys)       an action whose fields equal keys ('message' compares the prompt type)
  opt(step)                  the prompt may be auto-resolved by the env (single candidate); skipped only if a
                             later entry matches the current prompt
  group((step, picks), ...)  simultaneous optional triggers, any activation order; each activated member's picks are
                             queued last-in-first-out (chain resolution order) ahead of the plan; at the next
                             main-phase idle prompt unused optional picks are dropped and a missing required pick
                             stops the run
  rule(name)                 a choice computed from the public snapshot by a fixed rule (see Teacher._rule)
  mark(step, name)           like step; records the own hand when matched
  opp(step)                  allowed only in the opponent's turn; if never offered it is reported, not forced
"""
import collections

ASSAULT, MEDIUS, BEAST, MAIDEN, FIELD = 1122030, 97556336, 23829452, 74631897, 74733322
AWAKENING, GRAPH, VARNISH, LITERA, FINEMELT, CROSS = 23599634, 60946049, 74011784, 97434754, 34541940, 50277355
DIAKTOROS, NERVED, SHI, MABOROSHI, FANCY = 27184601, 17473466, 31822037, 36709484, 4993187
INFINITE, ASH, G, WARASHI, USAGI, PSY_DRIVER, PSY_GAMMA = 10045474, 14558127, 23434538, 73642296, 59438930, 49036338, 38814750
SPECIAL_SUMMON_OPTION = 1152  # !system 1152 特殊召喚 (1190 = 手札に加える)
NON_ENGINE_MONSTERS = {PSY_DRIVER, PSY_GAMMA, USAGI, ASH, G, WARASHI}
FILLER_RETURN_PRIORITY = [WARASHI, ASH, G]
HAND, MZONE = 2, 4
# Link markers of 無垢なる祈りの獄神使 (cards.cdb def=130 = TOP 0x80 | BOTTOM 0x02): left EMZ (place 6) points to own
# main zone sequence 1, right EMZ (place 7) to sequence 3.
EMZ_BELOW = [(6, 1), (7, 3)]


def a(act, code, **extra):
    return dict(act=act, code=code, **extra)


def opt(step):
    return dict(step, _opt=True)


def group(*members):
    return {'_group': [dict(step=step, picks=list(picks)) for step, picks in members]}


def rule(name, optional=False):
    return {'_rule': name, '_opt': optional}


def mark(step, name):
    return dict(step, _mark=name)


def opp(step):
    return dict(step, _opp=True)


FILLERS = [ASH, G, WARASHI]
CONDITIONS = {
    'C1': dict(
        label='wsMMwXxE: 襲撃 → メディウス、魔法罠コスト1枚（無限泡影）、ファインメルトのドローをメディウス②で戻す',
        reference='https://duel-simulator.miyauchidp.dev/replay/wsMMwXxE',
        reference_title='メディウス【襲撃】+コスト',
        article_headings=['メディウス+モンスターコスト', 'メディウス【襲撃】+魔法罠コスト'],
        hand=[ASSAULT, INFINITE] + FILLERS, seed=1110001,
        plan=[
            a('Activate', ASSAULT), a('Activate', ASSAULT), opt(a('None', MEDIUS)),
            a('Summon', MEDIUS), a('Activate', MEDIUS), a('None', NERVED), a('Activate', 0, effect=SPECIAL_SUMMON_OPTION),
            a('SpSummon', MAIDEN),
            group((a('Activate', NERVED), [opt(a('None', BEAST))]), (a('Activate', MAIDEN), [opt(a('None', FIELD))])),
            a('Activate', BEAST, message='select_idlecmd'), opt(a('None', DIAKTOROS)),
            a('Activate', BEAST), a('None', AWAKENING),
            a('Activate', FIELD), a('Activate', FIELD), a('None', GRAPH),
            a('Activate', GRAPH), a('Activate', GRAPH), a('None', VARNISH),
            a('Activate', VARNISH), a('None', ASSAULT),
            a('Activate', FIELD), a('None', LITERA),
            a('SpSummon', SHI), a('None', GRAPH), a('None', DIAKTOROS),
            a('Activate', SHI), opt(a('None', MABOROSHI)),
            a('Activate', FIELD), a('None', FINEMELT), opt(a('None', FINEMELT)),
            a('Activate', MABOROSHI), opt(a('None', DIAKTOROS)),
            a('Activate', LITERA), a('Activate', LITERA), a('None', AWAKENING),
            a('Activate', FINEMELT), mark(a('Activate', FINEMELT), 'before_draw'),
            a('Activate', MEDIUS), rule('medius_cost', optional=True),
            a('Set', AWAKENING),
            a('None', 0, phase='End'),
            opp(a('Activate', SHI)), opp(opt(a('None', FANCY))), opp(opt(a('None', SHI))), opp(a('None', MEDIUS)),
        ],
        reference_end=dict(own_turn_monsters=[MAIDEN, DIAKTOROS, LITERA, FINEMELT, MEDIUS, SHI], field=FIELD, set_spell_trap=[AWAKENING],
                           face_up_spell_trap=[ASSAULT],
                           opponent_turn='獄神使② → W:Pファンシーボール（獄神使＋メディウス）'),
    ),
    'C2': dict(
        label='XUS4azXa: 学都＋魔法罠コスト1枚（無限泡影）',
        reference='https://duel-simulator.miyauchidp.dev/replay/XUS4azXa',
        reference_title='学都+魔法罠コスト',
        article_headings=['学都+魔法罠コスト'],
        hand=[FIELD, INFINITE] + FILLERS, seed=1110002,
        plan=[
            a('Activate', FIELD), a('Activate', FIELD), a('None', GRAPH),
            a('Activate', GRAPH), a('Activate', GRAPH), a('None', VARNISH),
            a('Activate', VARNISH), a('None', LITERA),
            a('Activate', LITERA), a('Activate', LITERA), opt(a('None', VARNISH)),
            a('Activate', FIELD), a('None', BEAST), opt(a('None', BEAST)),
            a('Summon', BEAST),
            a('Activate', BEAST, message='select_idlecmd'), opt(a('None', MAIDEN)), opt(a('None', GRAPH)), opt(a('None', BEAST)),
            group((a('Activate', MAIDEN), [a('None', AWAKENING)]), (a('Activate', BEAST), [a('None', ASSAULT)])),
            a('Activate', ASSAULT), a('Activate', ASSAULT), opt(a('None', MEDIUS)),
            a('Summon', MEDIUS), a('Activate', MEDIUS), a('None', NERVED), a('Activate', 0, effect=SPECIAL_SUMMON_OPTION),
            a('SpSummon', SHI), a('None', MEDIUS), a('None', NERVED),
            group((a('Activate', SHI), [opt(a('None', MABOROSHI))]), (a('Activate', NERVED), [a('None', FINEMELT)])),
            a('Activate', MEDIUS), a('None', FINEMELT),
            a('Activate', FIELD), a('None', FINEMELT), opt(a('None', FINEMELT)),
            a('Activate', FINEMELT), mark(a('Activate', FINEMELT), 'before_draw'),
            a('None', 0, phase='End'),
        ],
        reference_end=dict(own_turn_monsters=[MAIDEN, LITERA, MEDIUS, FINEMELT, SHI], field=FIELD, set_spell_trap=[AWAKENING],
                           face_up_spell_trap=[ASSAULT], opponent_turn='リプレイに相手ターンの操作なし'),
    ),
}


def matches(step, action, message):
    for key, value in step.items():
        if key.startswith('_'):
            continue
        if key == 'message':
            if message != value:
                return False
        elif action.get(key) != value:
            return False
    return True


def passive(s):
    """Same order as article_witness.passive: cancel/finish, then End/Main2/Battle phase, else the only option."""
    for x in s['actions']:
        if x['act'] == 'Cancel' or x['finish']:
            return x['index']
    for phase in ['End', 'Main2', 'Battle']:
        for x in s['actions']:
            if x['phase'] == phase:
                return x['index']
    if len(s['actions']) == 1:
        return 0
    raise RuntimeError('Unresolved passive choice')


class Unresolved(RuntimeError):
    pass


def own_hand(s):
    return collections.Counter(c['code'] for c in s['cards'] if c['controller'] == 0 and c['location'] == HAND)


class Teacher:
    """Scripted player 0. Every decision is logged with the rule that produced it."""

    def __init__(self, plan):
        self.steps = [dict(x) if '_group' not in x else {'_group': [dict(m) for m in x['_group']]} for x in plan]
        self.pending = []          # picks of activated group members, chain-resolution (LIFO) order
        self.skipped_optional = []
        self.summon_context = None
        self.marks = {}
        self.log = []
        self.unmatched_chain_heads = []

    # ------------------------------------------------------------------ matching
    def _find(self, step, s):
        return next((x['index'] for x in s['actions'] if matches(step, x, s['message'])), None)

    def _match(self, s):
        own_turn = s['turn_player'] == 0
        if s['message'] == 'select_idlecmd' and self.pending:
            required = [p for p in self.pending if not p.get('_opt')]
            if required:
                raise Unresolved(f'expected selection never offered before returning to the main phase: {required}')
            self.skipped_optional.extend(self.pending)
            self.pending = []
        for j, pick in enumerate(self.pending):
            idx = self._find(pick, s)
            if idx is not None:
                self.skipped_optional.extend(self.pending[:j])
                del self.pending[:j + 1]
                return idx, dict(kind='group-pick', step=pick)
            if not pick.get('_opt'):
                break
        i = 0
        while i < len(self.steps):
            entry = self.steps[i]
            if entry.get('_opp') and own_turn:
                return None
            if '_group' in entry:
                for member in entry['_group']:
                    idx = self._find(member['step'], s)
                    if idx is not None:
                        entry['_group'].remove(member)
                        self.pending[0:0] = member['picks']
                        self.skipped_optional.extend(self.steps[:i])
                        del self.steps[:i]
                        if not entry['_group']:
                            del self.steps[0]
                        return idx, dict(kind='group', step=member['step'])
                return None
            if '_rule' in entry:
                idx = self._rule(entry['_rule'], s)
                if idx is not None:
                    self.skipped_optional.extend(self.steps[:i])
                    del self.steps[:i + 1]
                    return idx, dict(kind='rule', rule=entry['_rule'])
            else:
                idx = self._find(entry, s)
                if idx is not None:
                    self.skipped_optional.extend(self.steps[:i])
                    del self.steps[:i + 1]
                    if entry.get('_mark'):
                        self.marks[entry['_mark']] = dict(own_hand(s))
                    return idx, dict(kind='plan', step={k: v for k, v in entry.items() if not k.startswith('_') or k == '_opt'})
            if not entry.get('_opt'):
                return None
            i += 1
        return None

    def _rule(self, name, s):
        if name != 'medius_cost':
            raise ValueError(name)
        if s['message'] not in ['select_card', 'select_unselect_card']:
            return None
        candidates = {x['code']: x['index'] for x in s['actions'] if x['code'] and str(x.get('spec', '')).startswith('h')}
        if not candidates:
            return None
        before = collections.Counter(self.marks.get('before_draw', {}))
        drawn = [code for code in (own_hand(s) - before)]
        for code in drawn:
            if code in NON_ENGINE_MONSTERS and code in candidates:
                return candidates[code]
        for code in FILLER_RETURN_PRIORITY:
            if code in candidates:
                return candidates[code]
        return None

    # ------------------------------------------------------------------ defaults
    def _place(self, s):
        acts = s['actions']
        if self.summon_context == SHI:
            occupied = {c['sequence'] for c in s['cards'] if c['controller'] == 0 and c['location'] == MZONE}
            for place, below in EMZ_BELOW:
                if below not in occupied:
                    idx = next((x['index'] for x in acts if x['place'] == place), None)
                    if idx is not None:
                        return idx
        prefer = {MAIDEN: 1, CROSS: 6, FINEMELT: 3}.get(self.summon_context)
        return next((x['index'] for x in acts if x['place'] == prefer), acts[0]['index'])

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
            return passive(s), dict(kind='default-pass')
        if msg in ['select_place', 'select_disfield']:
            return self._place(s), dict(kind='default-place')
        if msg == 'select_position':
            return next((x['index'] for x in acts if x['position'] == 1), 0), dict(kind='default-position')
        if len(acts) == 1:
            return 0, dict(kind='only-option')
        if msg in ['select_card', 'select_unselect_card'] and any(x['finish'] for x in acts):
            return next(x['index'] for x in acts if x['finish']), dict(kind='default-finish')
        if head is None and msg == 'select_idlecmd':
            return passive(s), dict(kind='default-pass')
        if s['turn_player'] != 0 or s['phase'] != 4:
            return passive(s), dict(kind='default-pass')
        raise Unresolved(f'planned step not legal at {msg}: {head}')

    def head(self):
        return None if not self.steps else ({'group': [m['step'] for m in self.steps[0]['_group']]} if '_group' in self.steps[0]
                                            else {k: v for k, v in self.steps[0].items()})

    def remaining_required(self):
        """Plan entries and group picks still pending, excluding optional prompts and opponent-turn entries."""
        return [e for e in self.pending + self.steps if not e.get('_opt') and not e.get('_opp')]

    def __call__(self, s):
        idx, why = self.decide(s)
        self.log.append(dict(turn=s['turn_count'], message=s['message'], index=idx, why=why))
        return idx
