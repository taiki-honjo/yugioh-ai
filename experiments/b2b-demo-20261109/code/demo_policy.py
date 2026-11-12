"""Scripted candidate demonstrations of B2B use (pure, no engine). Our side = player 0.

Start: the own-turn B2B landing (B2B, Jukejoint, set Kewl Tune Synchro, Effect Veiler x3), opponent O1 in turn 2.
  DM-A  recovery/deploy: when Rukina's search is activated, B2B (3) targets Cue in the graveyard, special summons it, then
        S summons Trackmaker from Cue + Effect Veiler in hand; Trackmaker (1) searches (Kewl Tune Synchro preferred).
        Turn 3: B2B attacks a weaker face-up opponent monster, then direct attacks while offered.
  DM-B  attack: B2B (3) special summons Zaren (attack position); no S summon. Turn 3: same attack rule; Zaren may attack too.
Attack rule (turn 3): attack a face-up opponent monster only with an attacker of higher ATK; otherwise direct attacks by
B2B or Zaren while offered; zero-ATK monsters never attack; then leave the battle phase. These are candidate lines to be
checked by the engine and reviewed; they are not rules for the model ("always use B2B" is not intended)."""
RUK, CUE, ZAREN, B2B, TRACK, VEILER, SYNC, PLAYLIST = 13597785, 16387555, 4891376, 65961304, 42781164, 97268402, 78058681, 0
CARD_EFFECT_OFFSET = 10010
SPECIAL_SUMMON = 1152
ATK = {B2B: 3200, ZAREN: 2800, RUK: 2500, CUE: 900, TRACK: 0}
ATTACKERS = (B2B, ZAREN)
CONDITIONS = ['DM-A', 'DM-B']
SELECTIONS = ('select_card', 'select_unselect_card', 'select_sum', 'select_tribute')


def effect_index(a):
    e = a.get('effect', -1)
    return e - CARD_EFFECT_OFFSET if e is not None and e >= CARD_EFFECT_OFFSET else None


def new_state(condition):
    assert condition in CONDITIONS
    return dict(condition=condition, ctx=None, picked=[], b2b=None, track_search=None, battle_entered=None, attacks=[], log=[])


def decide(raw, st, step, passive):
    cond, msg, acts, chain = st['condition'], raw['message'], raw['actions'], raw['public_chain']
    find = lambda pred: next((a for a in acts if pred(a)), None)

    def take(a, label):
        st['log'].append(dict(step=step, message=msg, label=label, act=a.get('act'), code=a.get('code'), effect=a.get('effect')))
        return a['index'], label

    def fallback(label):
        try:
            idx = passive(raw)
        except Exception:
            idx = acts[0]['index']
        st['log'].append(dict(step=step, message=msg, label=label, offered=[(a.get('act'), a.get('code'), a.get('effect')) for a in acts]))
        return idx, label

    if msg == 'select_place':
        return take(acts[0], 'place-first')
    if msg == 'select_position':
        a = find(lambda a: a.get('position') == 1)
        return take(a or acts[0], 'position-attack' if a else 'position-first')
    if msg in SELECTIONS and st['ctx']:
        prefs = {'b2b': [CUE] if cond == 'DM-A' else [ZAREN], 'b2b-synchro': [TRACK, CUE, VEILER],
                 'track-search': [SYNC], 'attack': [RUK]}[st['ctx']]
        for code in prefs:
            a = find(lambda a, code=code: a.get('code') == code and (msg == 'select_card' or code not in st['picked']))
            if a:
                st['picked'].append(code)
                return take(a, f"{st['ctx']}-pick")
        if st['ctx'] == 'track-search' and msg == 'select_card':
            return take(acts[0], 'track-search-first-offered')
        a = find(lambda a: a.get('finish'))
        return take(a, f"{st['ctx']}-finish") if a else fallback(f"{st['ctx']}-UNEXPECTED-selection")
    if msg == 'select_option' and st['ctx'] == 'b2b':
        a = find(lambda a: a.get('effect') == SPECIAL_SUMMON)
        return take(a, 'b2b-option-special-summon') if a else fallback('b2b-option-NOT-OFFERED')
    if msg == 'select_yesno':
        if st['ctx'] == 'b2b' and find(lambda a: a['act'] == 'Activate' and a.get('code') == B2B and effect_index(a) == 1):
            if cond == 'DM-A':
                st['ctx'], st['picked'] = 'b2b-synchro', []
                return take(find(lambda a: a['act'] == 'Activate'), 'b2b-synchro-yes')
            return fallback('b2b-synchro-no')
        return fallback('yesno-pass')
    if msg in ('select_chain', 'select_effectyn'):
        if st['b2b'] is None and raw['turn_player'] == 1 and len(chain) == 1 and chain[0]['player'] == 1 and chain[0]['code'] == RUK:
            a = find(lambda a: a['act'] == 'Activate' and a.get('code') == B2B and effect_index(a) == 0)
            if a:
                st['b2b'], st['ctx'], st['picked'] = step, 'b2b', []
                return take(a, 'b2b')
            st['log'].append(dict(step=step, message=msg, label='b2b-NOT-OFFERED'))
        if cond == 'DM-A' and st['b2b'] is not None and st['track_search'] is None and raw['turn_player'] == 1:
            a = find(lambda a: a['act'] == 'Activate' and a.get('code') == TRACK and effect_index(a) == 0)
            if a:
                st['track_search'], st['ctx'], st['picked'] = step, 'track-search', []
                return take(a, 'track-search')
        return fallback('pass')
    if raw['turn_player'] == 0 and raw['turn_count'] >= 3:
        if msg == 'select_idlecmd' and st['battle_entered'] is None:
            a = find(lambda a: a.get('phase') == 'Battle')
            if a:
                st['battle_entered'] = step
                return take(a, 'enter-battle')
            return fallback('battle-NOT-OFFERED')
        if msg == 'select_battlecmd':
            targets = [c for c in raw['cards'] if c['controller'] == 1 and c['location'] == 4 and c['position'] & 5]
            for code in ATTACKERS:  # attack a weaker face-up monster first
                a = find(lambda a, code=code: a['act'] == 'Attack' and a.get('code') == code)
                if a and targets and all(ATK.get(t['code'], 99999) < ATK[code] for t in targets):
                    st['attacks'].append(dict(step=step, attacker=code, kind='monster', lp=(raw.get('lp0'), raw.get('lp1'))))
                    st['ctx'], st['picked'] = 'attack', []
                    return take(a, 'attack-monster')
            for code in ATTACKERS:
                a = find(lambda a, code=code: a['act'] == 'DirectAttack' and a.get('code') == code)
                if a:
                    st['attacks'].append(dict(step=step, attacker=code, kind='direct', lp=(raw.get('lp0'), raw.get('lp1'))))
                    return take(a, 'attack-direct')
            end = find(lambda a: a.get('phase') in ('Main2', 'End'))
            return take(end, 'end-battle') if end else fallback('end-battle-NOT-OFFERED')
    return fallback('pass')


def finished(raw):
    return raw['turn_count'] >= 4
