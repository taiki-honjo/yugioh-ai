"""Scripted decisions for the B2B conditions (pure, no engine). Our side = player 0.

Common route candidate (R0-R3, A1): in the opponent's draw phase, Remix (2) picks Ash Blossom and Reco from the
graveyard, Reco goes to hand, Ash is special summoned, then Remix's optional S summon makes Kewl Tune B2B from Ash
+ Zaren. Then, when the opponent activates Rukina's search (a monster effect), B2B (3) (description 0) per condition:
  R0 no B2B effect | R1 target Cue -> hand | R2 target Cue -> special summon, no S summon |
  R3 target Cue -> special summon, then S summon Trackmaker from Cue + Effect Veiler in hand
  A0 no Remix / no B2B; our turn 3: Zaren attacks (a second attack only if offered)
  A1 B2B (3) targets Zaren -> special summon (attack position); our turn 3: Zaren attacks, a second attack if offered
Every choice is taken only if it is among the legal actions; anything else is recorded and answered passively.
These are route candidates to be confirmed by the engine, not verified lines."""
RUK, ASH, RECO, CUE, ZAREN, REMIX, B2B, TRACK, VEILER = 13597785, 14558127, 89392810, 16387555, 4891376, 88170262, 65961304, 42781164, 97268402
CARD_EFFECT_OFFSET = 10010
TO_HAND, SPECIAL_SUMMON = 1190, 1152          # system strings offered by aux.SelectFromOptions in B2B (3)
BATTLE_PHASES = {0x8, 0x10, 0x20, 0x40, 0x80}
CONDITIONS = ['R0', 'R1', 'R2', 'R3', 'A0', 'A1']
B2B_TARGET = {'R1': CUE, 'R2': CUE, 'R3': CUE, 'A1': ZAREN}
B2B_OPTION = {'R1': TO_HAND, 'R2': SPECIAL_SUMMON, 'R3': SPECIAL_SUMMON, 'A1': SPECIAL_SUMMON}
SELECTIONS = ('select_card', 'select_unselect_card', 'select_sum', 'select_tribute')


def effect_index(a):
    e = a.get('effect', -1)
    return e - CARD_EFFECT_OFFSET if e is not None and e >= CARD_EFFECT_OFFSET else None


def new_state(condition):
    assert condition in CONDITIONS
    return dict(condition=condition, ctx=None, picked=[], remix=None, b2b=None, attacks=[], battle_entered=None, log=[])


def decide(raw, st, step, passive):
    """Return (index, label). `passive` is the witness passive() (declines / ends phases)."""
    cond, msg, acts, chain = st['condition'], raw['message'], raw['actions'], raw['public_chain']
    find = lambda pred: next((a for a in acts if pred(a)), None)

    def take(a, label, **extra):
        st['log'].append(dict(step=step, message=msg, label=label, code=a.get('code'), act=a.get('act'), effect=a.get('effect'), **extra))
        return a['index'], label

    def fallback(label):
        try:
            idx = passive(raw)
        except Exception:
            idx = acts[0]['index']
        st['log'].append(dict(step=step, message=msg, label=label, offered=[(a.get('act'), a.get('code'), a.get('effect')) for a in acts]))
        return idx, label

    b2b_route = cond in ('R0', 'R1', 'R2', 'R3', 'A1')
    # ---- prompts inside our own effects
    if msg == 'select_place':
        return take(acts[0], 'place-first')
    if msg == 'select_position':
        a = find(lambda a: a.get('position') == 1)  # face-up attack when offered
        return take(a or acts[0], 'position-attack' if a else 'position-first')
    if msg in SELECTIONS and st['ctx']:
        prefs = {'remix': [ASH, RECO] if msg != 'select_card' else [RECO],     # pick two; then the one that goes to hand
                 'remix-synchro': [B2B, ASH, ZAREN],
                 'b2b': [B2B_TARGET.get(cond)],
                 'b2b-synchro': [TRACK, CUE, VEILER],
                 'attack': [RUK]}[st['ctx']]
        for code in prefs:
            # select_card = a single choice (e.g. which card goes to hand, which monster, which target): earlier picks do not matter;
            # multi-pick prompts skip what this effect already picked
            a = find(lambda a, code=code: a.get('code') == code and (msg == 'select_card' or code not in st['picked']))
            if a:
                st['picked'].append(code)
                return take(a, f"{st['ctx']}-pick")
        a = find(lambda a: a.get('finish'))
        return take(a, f"{st['ctx']}-finish") if a else fallback(f"{st['ctx']}-UNEXPECTED-selection")
    if msg == 'select_yesno':
        if st['ctx'] == 'remix' and find(lambda a: a['act'] == 'Activate' and a.get('code') == REMIX and effect_index(a) == 1):
            if b2b_route:
                st['ctx'], st['picked'] = 'remix-synchro', []
                return take(find(lambda a: a['act'] == 'Activate'), 'remix-synchro-yes')
            return fallback('remix-synchro-no')
        if st['ctx'] == 'b2b' and find(lambda a: a['act'] == 'Activate' and a.get('code') == B2B and effect_index(a) == 1):
            if cond == 'R3':
                st['ctx'], st['picked'] = 'b2b-synchro', []
                return take(find(lambda a: a['act'] == 'Activate'), 'b2b-synchro-yes')
            return fallback('b2b-synchro-no')
        return fallback('yesno-pass')
    if msg == 'select_option' and st['ctx'] == 'b2b':
        want = B2B_OPTION.get(cond)
        a = find(lambda a: a.get('effect') == want)
        st['b2b']['option_offered'] = sorted(x.get('effect') for x in acts)
        return take(a, 'b2b-option') if a else fallback('b2b-option-NOT-OFFERED')
    # ---- chain windows
    if msg == 'select_chain':
        if b2b_route and st['remix'] is None and raw['turn_player'] == 1 and raw['phase'] == 1:
            a = find(lambda a: a['act'] == 'Activate' and a.get('code') == REMIX and effect_index(a) == 0)
            if a:
                st['remix'], st['ctx'], st['picked'] = step, 'remix', []
                return take(a, 'remix')
        if cond in B2B_TARGET and st['b2b'] is None and len(chain) == 1 and chain[0]['player'] == 1 and chain[0]['code'] == RUK:
            a = find(lambda a: a['act'] == 'Activate' and a.get('code') == B2B and effect_index(a) == 0)
            if a:
                st['b2b'], st['ctx'], st['picked'] = dict(step=step), 'b2b', []
                return take(a, 'b2b')
            st['log'].append(dict(step=step, message=msg, label='b2b-NOT-OFFERED'))
        return fallback('pass')
    # ---- our turn 3 battle (A conditions)
    if cond in ('A0', 'A1') and raw['turn_player'] == 0 and raw['turn_count'] >= 3:
        if msg == 'select_idlecmd' and st['battle_entered'] is None:
            a = find(lambda a: a.get('phase') == 'Battle')
            if a:
                st['battle_entered'] = step
                return take(a, 'enter-battle')
            return fallback('battle-NOT-OFFERED')
        if msg == 'select_battlecmd':
            attack = find(lambda a: a['act'] in ('Attack', 'DirectAttack') and a.get('code') == ZAREN)
            if attack and len(st['attacks']) < 2:
                st['attacks'].append(dict(step=step, act=attack['act'], lp=(raw.get('lp0'), raw.get('lp1'))))
                st['ctx'], st['picked'] = 'attack', []
                return take(attack, 'attack')
            end = find(lambda a: a.get('phase') in ('Main2', 'End'))
            st['log'].append(dict(step=step, message=msg, label='zaren-attack-offered' if attack else 'zaren-attack-not-offered',
                                  attacks_done=len(st['attacks'])))
            return take(end, 'end-battle') if end else fallback('end-battle-NOT-OFFERED')
    return fallback('pass')


def finished(raw, st):
    """Stop rule. R: when our turn 3 begins. A: once we have left the battle phase of turn 3 (or turn 4 began)."""
    if st['condition'].startswith('R'):
        return raw['turn_count'] >= 3
    return raw['turn_count'] > 3 or (st['battle_entered'] is not None and raw['phase'] in (0x100, 0x200))
