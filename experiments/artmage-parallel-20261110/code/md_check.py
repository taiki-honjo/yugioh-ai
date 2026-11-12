"""Offline MD limit check of the pinned 40/15 recipe against a saved secondary source (game8 423063).

Reads snapshot/md-check/game8-423063.html (saved copy, hashed) and the full 禁止/制限/準制限 lists in it,
normalises names (NFKC, no spaces, dash variants), and compares every card name in the recipe.
This is not the official in-game list; the result says so.
"""
import collections
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

P = Path('/home/hatk8/projects/yugioh-ai')
W = P / 'experiments/artmage-parallel-20261110'
SRC = W / 'snapshot/md-check/game8-423063.html'
CDB = P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb'
LIMIT = {'forbidden': 0, 'limited': 1, 'semi_limited': 2}


def norm(name):
    name = unicodedata.normalize('NFKC', name)
    name = re.sub(r'[\s　]', '', name)
    return re.sub(r'[－‐‑–—―ー-]', '-', name)


def recipe():
    main, extra, section = [], [], None
    for line in (W / 'decks/Artmage.ydk').read_text().splitlines():
        line = line.strip()
        if line in ('#main', '#extra', '!side'):
            section = line
        elif line.isdigit():
            (main if section == '#main' else extra if section == '#extra' else []).append(int(line))
    return collections.Counter(main + extra)


def lists(text):
    marks = [('forbidden', '禁止カード一覧</h3>'), ('limited', '制限カード一覧</h3>'), ('semi_limited', '準制限カード一覧</h3>')]
    starts = []
    for key, mark in marks:
        positions = [m.start() for m in re.finditer(re.escape(mark), text)]
        # the full lists come last in the page body; '制限カード一覧</h3>' also matches inside '準制限カード一覧</h3>'
        pos = [p for p in positions if not (key == 'limited' and text[p - 1:p] == '準')]
        starts.append((key, pos[-1]))
    end = text.find('良いプレイ環境を作るためのルール', starts[-1][1])
    out = {}
    for i, (key, start) in enumerate(starts):
        stop = starts[i + 1][1] if i + 1 < len(starts) else end
        seg = text[start:stop]
        names = [html.unescape(x) for x in re.findall(r'alt="([^"]+?)画像"', seg)]
        names += [html.unescape(x).strip() for x in re.findall(r'<a class="a-link"[^>]*>([^<]+)</a>', seg)]
        out[key] = sorted(set(n for n in names if n))
    return out


def main():
    text = SRC.read_text(encoding='utf-8')
    found = lists(text)
    lookup = {}
    for key, names in found.items():
        for n in names:
            lookup.setdefault(norm(n), []).append(key)
    db = sqlite3.connect(CDB)
    rows, over = [], []
    for code, count in sorted(recipe().items()):
        name = db.execute('select name from texts where id=?', (code,)).fetchone()[0]
        status = lookup.get(norm(name), [])
        allowed = min([LIMIT[s] for s in status], default=3)
        rows.append(dict(code=code, name=name, copies=count, listed_as=status, allowed=allowed, ok=count <= allowed))
        if count > allowed:
            over.append(rows[-1])
    plain = re.sub(r'<[^>]+>', ' ', text)
    updated = re.search(r'最終更新日：\s*(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2})', plain)
    start = text.find('11/5より新リミットレギュレーションが適用')
    change = text[start:text.find('環境上位の複数デッキから規制', start)]
    latest = sorted(set(html.unescape(x) for x in re.findall(r'alt="([^"]+?)画像"', change)))
    names = {norm(r['name']) for r in rows}
    full = {norm(n) for v in found.values() for n in v}
    out = dict(source='https://game8.jp/yugioh-masterduel/423063', saved_copy=str(SRC), saved_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
               source_last_updated=updated.group(1) if updated else None, list_sizes={k: len(v) for k, v in found.items()},
               recipe_cards=len(rows), listed=[r for r in rows if r['listed_as']], over_limit=over, rows=rows,
               latest_change_2026_11_05=latest, latest_change_in_recipe=[n for n in latest if norm(n) in names],
               latest_change_missing_from_full_lists=[n for n in latest if norm(n) not in full],
               passed_against_this_source=not over,
               scope='Secondary web source (game8), not the in-game official list. Card-pool existence in MD is not checked by this list.')
    (W / 'md-check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ['source_last_updated', 'list_sizes', 'recipe_cards', 'passed_against_this_source']}, ensure_ascii=False))
    for r in out['listed']:
        print('LISTED', r['code'], r['name'], r['copies'], r['listed_as'])


if __name__ == '__main__':
    main()
