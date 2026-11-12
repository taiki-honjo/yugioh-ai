"""Offline split check (no engine): complete-hand multisets of new demo hands vs the old Artmage splits.

Permutations of one complete hand are one group. New hands must not share a group with the old validation
or holdout; sharing with old train is reported (allowed for train-side demos). Also re-checks that the old
splits themselves keep each group in a single split.
"""
import collections
import json
from pathlib import Path
import sys

P = Path('/home/hatk8/projects/yugioh-ai')
W = P / 'experiments/artmage-parallel-20261110'
OLD_MANIFEST = P / 'experiments/opponent-article-training-20261108/run-20261108-145203/manifest.json'
sys.path.insert(0, str(W / 'code'))
from artmage_teacher import CONDITIONS  # noqa: E402


def key(hand):
    return tuple(sorted(hand))


def check(old_cases, new_hands):
    groups = collections.defaultdict(set)
    for split, cases in old_cases.items():
        for case in cases:
            groups[key(case['hand'])].add(split)
    old_mixed = {str(k): sorted(v) for k, v in groups.items() if len(v) > 1}
    new = {}
    for name, hand in new_hands.items():
        splits = sorted(groups.get(key(hand), set()))
        new[name] = dict(hand=list(hand), group=list(key(hand)), old_splits=splits,
                         overlaps_validation_or_holdout=bool({'validation', 'holdout'} & set(splits)))
    new_keys = collections.Counter(key(h) for h in new_hands.values())
    return dict(old_groups=len(groups), old_groups_in_more_than_one_split=old_mixed, new=new,
                new_hands_sharing_a_group=[list(k) for k, n in new_keys.items() if n > 1],
                passed=not old_mixed and not any(x['overlaps_validation_or_holdout'] for x in new.values()))


def main():
    manifest = json.loads(OLD_MANIFEST.read_text())['decks']['Artmage']['cases']
    out = check(manifest, {name: c['hand'] for name, c in CONDITIONS.items()})
    out['old_manifest'] = str(OLD_MANIFEST)
    out['old_case_counts'] = {k: len(v) for k, v in manifest.items()}
    (W / 'split-check.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ['old_groups', 'old_groups_in_more_than_one_split', 'new_hands_sharing_a_group', 'passed']}, ensure_ascii=False))
    for name, x in out['new'].items():
        print(name, x['group'], 'old splits:', x['old_splits'])
    if not out['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
