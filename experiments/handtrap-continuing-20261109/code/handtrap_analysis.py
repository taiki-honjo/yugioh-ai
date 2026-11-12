"""Pure analysis for the Maxx C / Droll continuing-effect conditions (no engine).

Setting: player 0 (the deploying side) follows a fixed teacher route during turn 1; player 1 (the side holding
Maxx C and Droll & Lock Bird) only activates them according to the condition. Opponent-hidden zones are read by
the evaluator only. Attribution rules are stated in each function; anything outside them is left unattributed."""
import collections
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'capability_analysis', Path('/home/hatk8/projects/yugioh-ai/experiments/capability-probe-20261109/code/analysis.py'))
A = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A)

G, DROLL = 23434538, 94145021
HAND, MZONE, SZONE, GRAVE, REMOVED = 2, 4, 8, 16, 32


def zone(s, controller, location):
    return collections.Counter(c['code'] for c in s['cards'] if c['controller'] == controller and c['location'] == location)


def visible(s, controller):
    return zone(s, controller, MZONE) + zone(s, controller, SZONE) + zone(s, controller, GRAVE) + zone(s, controller, REMOVED)


def from_hidden(before, after, controller):
    """Cards added to a hand that did not come from that player's public zones (i.e. from deck/extra)."""
    added = zone(after, controller, HAND) - zone(before, controller, HAND)
    lost = visible(before, controller) - visible(after, controller)
    return collections.Counter({code: n - lost[code] for code, n in added.items() if n - lost[code] > 0})


def activation_step(trace, player, code):
    return next((r['step'] for r in trace if r['snapshot']['to_play'] == player and r['selected']['act'] == 'Activate'
                 and r['selected']['code'] == code), None)


def resolved_after(trace, step):
    """First recorded step after `step` whose snapshot has an empty chain (the activation's chain has resolved)."""
    if step is None:
        return None
    return next((r['step'] for r in trace if r['step'] > step and r['snapshot']['chain_size'] == 0), None)


def events(trace, end):
    """Per transition between consecutive snapshots: player-0 monster entries, player-0 adds from hidden zones,
    player-1 adds from hidden zones. A transition is attributed to the step whose action preceded it."""
    snaps = [(r['step'], r['snapshot']) for r in trace] + [('end', end)]
    out = []
    for (step, before), (_, after) in zip(snaps, snaps[1:]):
        entries = sum((zone(after, 0, MZONE) - zone(before, 0, MZONE)).values())
        p0_adds, p1_adds = from_hidden(before, after, 0), from_hidden(before, after, 1)
        if entries or p0_adds or p1_adds:
            out.append(dict(step=step, p0_monster_entries=entries, p0_adds_from_hidden=dict(p0_adds), p1_adds_from_hidden=dict(p1_adds)))
    return out


def deploying_actions(trace):
    return [A.normalized(A.meaning(r['selected'])) for r in trace if r['snapshot']['to_play'] == 0]


def summarize(trace, end, start, blocked=None):
    """Condition summary. `start` is the snapshot before any decision (both opening hands)."""
    g_at, droll_at = activation_step(trace, 1, G), activation_step(trace, 1, DROLL)
    droll_live = resolved_after(trace, droll_at)
    ev = events(trace, end)
    after = lambda step, key: None if step is None else sum(sum(e[key].values()) if isinstance(e[key], dict) else e[key]
                                                            for e in ev if e['step'] != 'end' and e['step'] >= step)
    total = lambda key: sum(sum(e[key].values()) if isinstance(e[key], dict) else e[key] for e in ev)
    p1_draws = collections.Counter()
    for e in ev:
        p1_draws.update(e['p1_adds_from_hidden'])
    return dict(
        g_activation_step=g_at, droll_activation_step=droll_at, droll_in_effect_from_step=droll_live,
        p0_monster_entries=total('p0_monster_entries'), p0_monster_entries_after_g=after(g_at, 'p0_monster_entries'),
        p0_monster_entries_after_droll=after(droll_live, 'p0_monster_entries'),
        p0_adds_from_hidden=total('p0_adds_from_hidden'), p0_adds_after_droll=after(droll_live, 'p0_adds_from_hidden'),
        p0_add_events=[dict(step=e['step'], cards=e['p0_adds_from_hidden']) for e in ev if e['p0_adds_from_hidden']],
        p1_cards_gained_from_hidden=dict(p1_draws), p1_gained_total=sum(p1_draws.values()),
        p1_gained_after_droll=after(droll_live, 'p1_adds_from_hidden'),
        p1_gain_events=[dict(step=e['step'], cards=e['p1_adds_from_hidden']) for e in ev if e['p1_adds_from_hidden']],
        p1_hand_start=sorted(zone(start, 1, HAND).elements()), p1_hand_end=sorted(zone(end, 1, HAND).elements()),
        p0_end=dict(monsters=sorted(zone(end, 0, MZONE).elements()), spell_trap=sorted(zone(end, 0, SZONE).elements()),
                    hand_count=sum(zone(end, 0, HAND).values()), graveyard=sorted(zone(end, 0, GRAVE).elements())),
        deploying_decisions=len(deploying_actions(trace)), script_blocked=blocked)


def compare(base_trace, base_end, trace, end):
    """Where the deploying side's executed actions first differ from the baseline, and the end-board difference."""
    a, b = deploying_actions(base_trace), deploying_actions(trace)
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
    board = lambda s: zone(s, 0, MZONE) + zone(s, 0, SZONE)
    return dict(identical_deploying_actions=a == b, first_divergence_index=first,
                baseline_decisions=len(a), decisions=len(b),
                end_board_missing_vs_baseline=dict(board(base_end) - board(end)),
                end_board_extra_vs_baseline=dict(board(end) - board(base_end)))


def legal_in_baseline(base_trace, action):
    """Was this deploying action (by meaning, ignoring the option index) legal somewhere in the baseline?"""
    key = lambda m: (m['act'], m['code'], m['effect'])
    target = key(A.meaning(action))
    return [r['step'] for r in base_trace if r['snapshot']['to_play'] == 0 and any(key(A.meaning(x)) == target for x in r['snapshot']['actions'])]
