"""Pure analysis for the Zaren multi-link chain conditions (no engine).

Zaren (2) (pinned c4891376.lua): description 1; condition: chain link >= 2; on activation the player chooses
description 2 = negate and destroy THAT effect (the link Zaren responds to) or description 3 = negate and destroy
the effect THAT link chained to (one link below). Only options whose link can be negated are offered."""
import collections
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'capability_analysis', Path('/home/hatk8/projects/yugioh-ai/experiments/capability-probe-20261109/code/analysis.py'))
A = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A)

ZAREN, VEILER, RUK = 4891376, 97268402, 13597785
HAND, MZONE, GRAVE = 2, 4, 16
OPTION_THIS_LINK, OPTION_LINK_BELOW, ACTIVATION = 2, 3, 1


def our_windows(trace):
    """Our decision points in the opponent turn, with chain length and Zaren's offered descriptions."""
    out = []
    for row in trace:
        s = row['snapshot']
        if s['turn_player'] != 1 or s['to_play'] != 0:
            continue
        zaren = sorted(A.effect_index(a) for a in s['actions'] if a['act'] == 'Activate' and a['code'] == ZAREN)
        out.append(dict(step=row['step'], message=s['message'], chain_length=len(s['public_chain']),
                        chain=[dict(player=c['player'], code=c['code']) for c in s['public_chain']],
                        zaren_descriptions=zaren, chosen=A.meaning(row['selected'])))
    return out


def zaren_legality(trace):
    """Per chain length: windows seen and windows where Zaren (2) (description 1) was a legal activation."""
    seen, legal = collections.Counter(), collections.defaultdict(list)
    for w in our_windows(trace):
        if w['message'] != 'select_chain':
            continue
        seen[w['chain_length']] += 1
        if ACTIVATION in w['zaren_descriptions']:
            legal[w['chain_length']].append(w['step'])
    return {n: dict(windows=seen[n], zaren_legal_at_steps=legal[n]) for n in sorted(seen)}


def zaren_option(trace):
    """The select_option prompt after Zaren's activation: offered descriptions and the one chosen."""
    for w in our_windows(trace):
        if w['message'] == 'select_option' and w['zaren_descriptions']:
            return dict(step=w['step'], offered=w['zaren_descriptions'], chosen=w['chosen']['effect_index'])
    return None


def longest_chain(trace):
    best = []
    for row in trace:
        if row['snapshot']['turn_player'] != 1:
            continue
        chain = row['snapshot']['public_chain']
        if len(chain) > len(best):
            best = chain
    return [dict(player=c['player'], code=c['code'], description=c['description']) for c in best]


def where(snapshot, controller, code):
    return sorted(c['location'] for c in snapshot['cards'] if c['controller'] == controller and c['code'] == code)


def outcome(trace, end):
    """Search result (judged from the opponent hand only, see capability analysis) plus where the involved cards are."""
    o1 = A.o1_stages(trace, end)
    activation = next((r['snapshot'] for r in trace if r['snapshot']['to_play'] == 1 and r['selected']['act'] == 'Activate'
                       and r['selected']['code'] == RUK), None)
    return dict(search=o1['outcome'], search_evidence=o1['evidence'], notes=o1['notes'],
                responses=[dict(step=r['step'], chain_link=r['chain_link'], code=r['action']['code'], effect_index=r['action']['effect_index'])
                           for r in o1['responses']],
                longest_chain=longest_chain(trace),
                rukina_locations_before=where(activation, 1, RUK) if activation else None, rukina_locations_after=where(end, 1, RUK),
                our_veiler_locations_after=where(end, 0, VEILER), our_zaren_locations_after=where(end, 0, ZAREN))
