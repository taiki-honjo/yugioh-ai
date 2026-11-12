"""Pure analysis for opening-capability-eval-v1 probes (no engine, no model).

Works on run() traces: a list of {step, snapshot, selected}, where snapshot is a compact
benchmark_snapshot() (cards with code/controller/location/sequence/position/level, public_chain,
legal actions) and selected is the executed legal action. The evaluator may read the opponent's
hidden hand here; nothing from this module is ever fed to the model.
"""
import collections
import json
from pathlib import Path

CARD_EFFECT_OFFSET = 10010  # ygopro.h: action effect = description index + 10010; 0 = default description
HAND, MZONE, SZONE, GRAVE, REMOVED, EXTRA = 2, 4, 8, 16, 32, 64
RUK, ASH, REMIX, BARONNE, CRYSTAL, RHAP, SPIRIT = 13597785, 14558127, 88170262, 84815190, 50954680, 24092792, 12375297
ENGINE_RUN_LIMIT = 12


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def effect_index(action):
    """Description index of an Activate action (0 for Stringid(code,0)), or None for the default description."""
    effect = action.get('effect', -1)
    return effect - CARD_EFFECT_OFFSET if effect is not None and effect >= CARD_EFFECT_OFFSET else None


def meaning(action):
    """What an action does, independent of its index in the option list."""
    keys = ['act', 'code', 'name', 'effect', 'spec', 'place', 'position', 'number', 'phase', 'finish', 'response']
    out = {k: action.get(k) for k in keys}
    out['effect_index'] = effect_index(action)
    return out


def zone(snapshot, controller, location):
    return collections.Counter(c['code'] for c in snapshot['cards'] if c['controller'] == controller and c['location'] == location)


def diff(before, after):
    return dict(added=dict(after - before), removed=dict(before - after))


# ---------------------------------------------------------------- O1 stages (preparation / response / outcome)
def o1_stages(trace, final):
    """Split the opponent turn of a run into preparation (our activations before the O1 search is activated),
    the O1 search activation, our responses while it is on the chain, and the actual outcome of the search.

    The outcome is judged only from the opponent's hand contents between the search activation and the moment
    its chain has resolved: a newly added card means the search resolved. Removing the activating card is
    recorded separately and never counted as stopping the search."""
    out = dict(preparation=[], search=None, responses=[], outcome='unverified', evidence={}, notes=[])
    turn = [i for i, row in enumerate(trace) if row['snapshot']['turn_player'] == 1]
    if not turn:
        out['notes'].append('no opponent-turn decisions in the trace')
        return out
    search_at = next((i for i in turn if trace[i]['snapshot']['to_play'] == 1 and trace[i]['selected']['act'] == 'Activate'
                      and trace[i]['selected']['code'] == RUK), None)
    # preparation: each own activation before the search and what its chain changed for us
    for i in [i for i in turn if search_at is None or i < search_at]:
        row = trace[i]
        if row['snapshot']['to_play'] != 0 or row['selected']['act'] != 'Activate':
            continue
        end = next((j for j in range(i + 1, len(trace)) if trace[j]['snapshot']['chain_size'] == 0), None)
        after = trace[end]['snapshot'] if end is not None else final
        before = row['snapshot']
        picks = [meaning(trace[j]['selected']) for j in range(i + 1, end if end is not None else len(trace))
                 if trace[j]['snapshot']['to_play'] == 0 and trace[j]['selected']['act'] == 'None' and trace[j]['selected']['code']]
        out['preparation'].append(dict(
            step=row['step'], phase=before['phase'], chain_at_activation=before['public_chain'], action=meaning(row['selected']),
            picks=picks, resolved_at_step=trace[end]['step'] if end is not None else None,
            own_hand=diff(zone(before, 0, HAND), zone(after, 0, HAND)),
            own_monsters=diff(zone(before, 0, MZONE), zone(after, 0, MZONE)),
            own_graveyard=diff(zone(before, 0, GRAVE), zone(after, 0, GRAVE))))
    if search_at is None:
        out['notes'].append('O1 search was never activated')
        return out
    activation = trace[search_at]['snapshot']
    out['search'] = dict(step=trace[search_at]['step'], phase=activation['phase'], action=meaning(trace[search_at]['selected']))
    # responses: our activations while the O1 search is on the chain
    resolved_row = None
    for j in range(search_at + 1, len(trace)):
        s = trace[j]['snapshot']
        on_chain = any(link['player'] == 1 and link['code'] == RUK for link in s['public_chain'])
        if s['chain_size'] == 0:
            resolved_row = j
            break
        if on_chain and s['to_play'] == 0:
            legal = [meaning(a) for a in s['actions'] if a['act'] == 'Activate']
            chosen = trace[j]['selected']
            if chosen['act'] == 'Activate':
                out['responses'].append(dict(step=trace[j]['step'], chain_link=len(s['public_chain']) + 1,
                                             action=meaning(chosen), legal_activations=legal))
    after = trace[resolved_row]['snapshot'] if resolved_row is not None else final
    if after['chain_size'] != 0:
        out['notes'].append('the chain holding the search had not resolved at the end of the record')
        return out
    hand = diff(zone(activation, 1, HAND), zone(after, 1, HAND))
    visible = lambda s: zone(s, 1, MZONE) + zone(s, 1, SZONE) + zone(s, 1, GRAVE) + zone(s, 1, REMOVED)
    lost_visible = visible(activation) - visible(after)  # cards that may have moved hand-wards from public zones
    from_hidden = {code: n - lost_visible[code] for code, n in hand['added'].items() if n - lost_visible[code] > 0}
    source_left_field = zone(activation, 1, MZONE)[RUK] > zone(after, 1, MZONE)[RUK]
    out['evidence'] = dict(opponent_hand=hand, added_from_hidden_zone=from_hidden,
                           resolved_snapshot_step=trace[resolved_row]['step'] if resolved_row is not None else 'final',
                           search_source_left_field=source_left_field)
    if not hand['added'] and not hand['removed']:
        out['outcome'] = 'search_not_applied'  # no card reached the opponent's hand while that chain resolved
    elif from_hidden:
        out['outcome'] = 'search_resolved'
        if hand['removed']:
            out['notes'].append('the opponent hand also lost cards during the chain')
    else:
        out['notes'].append('opponent hand changed in a way this check cannot attribute; left unverified')
    if source_left_field:
        out['notes'].append('the search source left the field; recorded separately, not counted as stopping the search')
    return out


# ---------------------------------------------------------------- response candidates per window
def windows(trace, extra_snapshots=()):
    """Our decision points during the opponent's turn, with every legal Activate (card + effect index)."""
    rows = []
    for row in list(trace) + [dict(step=None, snapshot=s, selected=None) for s in extra_snapshots]:
        s = row['snapshot']
        if s['turn_player'] != 1 or s['to_play'] != 0:
            continue
        rows.append(dict(step=row['step'], phase=s['phase'], message=s['message'], chain=s['public_chain'],
                         activations=[meaning(a) for a in s['actions'] if a['act'] == 'Activate'],
                         chosen=meaning(row['selected']) if row['selected'] else None))
    return rows


def baronne_checks(trace, extra_snapshots=()):
    """Baronne (1) is an ignition effect (description index 0), (2) a quick effect (index 1), per the pinned script.
    Control: (1) appears among our own-turn legal actions. Check: it never appears in the opponent's turn."""
    own_turn_first = [row['step'] for row in trace if row['snapshot']['turn_player'] == 0
                      and any(a['act'] == 'Activate' and a['code'] == BARONNE and effect_index(a) == 0 for a in row['snapshot']['actions'])]
    opp = windows(trace, extra_snapshots)
    first_in_opp = [w['step'] for w in opp if any(a['code'] == BARONNE and a['effect_index'] == 0 for a in w['activations'])]
    second_in_opp = [w['step'] for w in opp if any(a['code'] == BARONNE and a['effect_index'] == 1 for a in w['activations'])]
    return dict(own_turn_control_steps=own_turn_first, opponent_turn_windows=len(opp),
                baronne_1_in_opponent_turn=first_in_opp, baronne_2_in_opponent_turn=second_in_opp,
                ok=bool(own_turn_first) and not first_in_opp and bool(opp))


# ---------------------------------------------------------------- replay comparison
def compare_step(recorded, snapshot, executed):
    """Observable-snapshot and action-meaning equality for one step; returns a list of differing keys."""
    a, b = normalized(recorded['snapshot']), normalized(snapshot)
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if normalized(meaning(recorded['selected'])) != normalized(meaning(executed)):
        keys.append('selected_meaning')
    return keys


# ---------------------------------------------------------------- engine-run ledger
class Ledger:
    """Counts every new real-engine run (one run = one duel from the opening hand). Refuses beyond the limit."""
    def __init__(self, path, limit=ENGINE_RUN_LIMIT):
        self.path, self.limit = Path(path), limit

    def rows(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []

    def started(self):
        return [r for r in self.rows() if r['event'] == 'start']

    def start(self, purpose, case_id, time):
        n = len(self.started())
        if n >= self.limit:
            raise RuntimeError(f'engine-run limit reached ({n}/{self.limit}); not starting {purpose} {case_id}')
        row = dict(event='start', run=n + 1, purpose=purpose, case_id=case_id, time_jst=time)
        with self.path.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        return row['run']

    def finish(self, run, status, time, **extra):
        with self.path.open('a') as handle:
            handle.write(json.dumps(dict(event='finish', run=run, status=status, time_jst=time, **extra), ensure_ascii=False) + '\n')
