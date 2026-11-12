"""Decode saved duel-simulator replays (manual board tool, not a rules engine) into card codes; offline.

Card identity in a replay is the card's index in the uploaded deck image (main-N / extra-N). The image of all
three Artmage replays (snapshot/references/deck-image.jpg) lists the recipe in the same order as the pinned
decks/Artmage.ydk; the mapping below relies on that visual check (main 0-39, extra 0-14).
"""
import json
from pathlib import Path
import sqlite3

P = Path('/home/hatk8/projects/yugioh-ai')
W = P / 'experiments/artmage-parallel-20261110'
REF = W / 'snapshot/references'
CDB = P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb'


def recipe_order():
    main, extra, section = [], [], None
    for line in (W / 'decks/Artmage.ydk').read_text().splitlines():
        line = line.strip()
        if line in ('#main', '#extra', '!side'):
            section = line
        elif line.isdigit():
            (main if section == '#main' else extra if section == '#extra' else []).append(int(line))
    return main, extra


def decode(path):
    main, extra = recipe_order()
    db = sqlite3.connect(CDB)
    name = lambda code: db.execute('select name from texts where id=?', (code,)).fetchone()[0]

    def card(cid):
        kind, idx = cid.split('-')[:2]
        code = (main if kind == 'main' else extra)[int(idx)]
        return dict(id=cid, code=code, name=name(code))

    raw = json.loads(Path(path).read_text())
    data = json.loads(raw['stateJson'])['data']
    me = data['initialState']['players']['self']
    ops = []
    for i, op in enumerate(data['operations']):
        row = dict(i=i, type=op['type'])
        if op.get('cardId'):
            row['card'] = card(op['cardId'])
        if op.get('from') or op.get('to'):
            row['from'] = '{}{}'.format(op.get('from', {}).get('zoneType', ''), op.get('from', {}).get('zoneIndex', ''))
            row['to'] = '{}{}'.format(op.get('to', {}).get('zoneType', ''), op.get('to', {}).get('zoneIndex', ''))
        if op['type'] == 'endTurn':
            row['metadata'] = op.get('metadata')
        ops.append(row)
    shown_hand = [card(c['id']) for c in me['hand']] + [o['card'] for o in ops if o.get('from', '').startswith('deck') and o.get('to', '').startswith('hand')
                                                         and not any(p['type'] == 'activate' for p in ops[:o['i']])]
    return dict(id=raw['id'], title=raw['title'], created=raw['createdAt'], deck_image_hash=raw['deckImageHash'],
                operation_count=len(ops), initial_hand_shown=[card(c['id']) for c in me['hand']],
                hand_built_before_first_activation=shown_hand, operations=ops,
                note='Manual simulator: draws/searches/summons are moves; zones and order are the uploader\'s, not engine-checked.')


def main():
    out = {rid: decode(REF / f'{rid}.json') for rid in ['g2rN57Xb', 'wsMMwXxE', 'XUS4azXa']}
    (REF / 'decoded.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    for rid, d in out.items():
        print(rid, d['title'], d['operation_count'], [c['name'] for c in d['hand_built_before_first_activation']])


if __name__ == '__main__':
    main()
