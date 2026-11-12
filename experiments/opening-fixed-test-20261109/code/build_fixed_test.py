"""Build and freeze the fixed opening test from spec.json. Teacher-only verification:
NO model is loaded or evaluated here. Uses the production v2 snapshot code read-only."""
import collections, copy, itertools, json, random, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C

spec_path = OUT / 'spec.json'
spec = C.read_json(spec_path)
assert spec['status'] == 'pre-registered' and not (OUT / 'manifest.json').exists()
cfg = C.load_config(P / 'config/continuous-elfnotes.json')
C.runtime_check(cfg)
common = C.backend().policy_common(cfg)
import all_opening_cases as bank_module
w = common.v  # all_opening_witness
SEED = spec['seed_base']; PER = spec['hands_per_family']; MAX = spec['max_attempts_per_family']

# ---- exclusion: every 5-card hand multiset in any JSON case definition of the project
excluded, sources = set(), 0
def collect(value):
    if isinstance(value, dict):
        hand = value.get('hand')
        if isinstance(hand, list) and len(hand) == 5 and all(isinstance(x, int) for x in hand):
            excluded.add(tuple(sorted(hand)))
        for child in value.values():
            collect(child)
    elif isinstance(value, list):
        for child in value:
            collect(child)
for base in [P / 'experiments', P / 'evaluation']:
    for path in base.rglob('*.json'):
        if path.is_relative_to(OUT) or path.stat().st_size > 30 * 1024 * 1024:
            continue
        try:
            before = len(excluded); collect(json.loads(path.read_text())); sources += len(excluded) > before
        except (ValueError, UnicodeDecodeError):
            continue

# ---- templates and candidate orders
protos = bank_module.bank(include_baronne=True)
templates, groups = {}, {}
for proto in protos:
    templates.setdefault(proto['eval_family'], proto)
limits = collections.Counter(w.v.CARDS['main'])
POOL = [w.v.VEILER, w.v.ASH, w.v.DROLL, w.v.G, w.v.OGRE, 52038441, 70088809]
def group_key(t):
    return (tuple(sorted(t['required'])), t['family'] == 'elf-rukina-regina', bool(t.get('recover')))
for family, t in templates.items():
    key = group_key(t)
    if key in groups:
        continue
    pool = list(POOL)
    if key[2]:
        pool.remove(w.v.ASH)
    if key[1]:
        pool.remove(w.v.VEILER)
    hands = []
    for tail in itertools.combinations_with_replacement(pool, 5 - len(t['required'])):
        hand = list(t['required']) + list(tail)
        if tuple(sorted(hand)) in excluded or any(n > limits[c] for c, n in collections.Counter(hand).items()):
            continue
        hands.append(hand)
    random.Random(SEED + len(groups)).shuffle(hands)
    groups[key] = dict(index=len(groups), hands=hands)

# ---- teacher-only verification and acceptance
cases, attempts = [], []
for fi, (family, t) in enumerate(templates.items()):
    group = groups[group_key(t)]; accepted = 0
    for ai, hand in enumerate(group['hands'][:MAX]):
        if accepted == PER:
            break
        seed = SEED + 1000 * fi + ai
        hand = list(hand); random.Random(seed).shuffle(hand)
        case = copy.deepcopy(t)
        case.update(hand=hand, seed=seed, split=spec['split'], group_id=ai, case_id=f'{family}-ft{ai:02d}', arm=t.get('arm', 'pass'))
        case = json.loads(json.dumps(case))  # exactly the form that is frozen and later evaluated
        result, _ = w.run(bank_module.normalize(case))  # policy=None: the registered teacher plays
        ok = bool(result['passed']) and result['lua_errors'] == 0
        attempts.append(dict(family=family, attempt=ai, hand=hand, seed=seed, teacher_passed=ok, failure=result.get('failure'),
                             chosen_variant=result.get('chosen_variant'), decisions=len(result['trace'])))
        print('TEACHER', family, ai, ok, result.get('failure'), flush=True)
        if ok:
            cases.append(case); accepted += 1
    if accepted < PER:
        print('SHORTFALL', family, accepted, flush=True)

all_split_hands = {tuple(sorted(c['hand'])) for c in cases}
assert not all_split_hands & excluded
assert len({c['case_id'] for c in cases}) == len(cases)
cases_path = OUT / 'cases.json'
cases_path.write_text(json.dumps(dict(cases=cases), ensure_ascii=False, indent=1) + '\n')
counts = collections.Counter(c['eval_family'] for c in cases)
snapshot = C.read_json(Path(cfg['root']) / 'snapshot/hashes.json')
manifest = dict(
    name=spec['name'], status='frozen', frozen_jst=C.now(), spec_file='spec.json', spec_sha256=C.sha(spec_path),
    cases_file='cases.json', cases_sha256=C.sha(cases_path), total=len(cases), families=dict(sorted(counts.items())),
    shortfall={f: PER - counts.get(f, 0) for f in templates if counts.get(f, 0) < PER},
    excluded_hand_multisets=len(excluded), json_files_contributing_hands=sources,
    overlap_with_existing_hands=0, teacher_attempts=len(attempts), teacher_rejected=sum(not a['teacher_passed'] for a in attempts),
    attempts=attempts, lua_errors=w.native.get_lua_error_count(), model_evaluated=False, frozen_before_any_model_evaluation=True,
    code=dict(production_root=cfg['root'], snapshot_hashes_sha256=C.sha(Path(cfg['root']) / 'snapshot/hashes.json'),
              all_opening_witness=snapshot['policy-code/all_opening_witness.py'], all_opening_cases=snapshot['policy-code/all_opening_cases.py'],
              center_placement=snapshot['policy-code/center_placement.py'], builder_sha256=C.sha(__file__)))
assert manifest['lua_errors'] == 0
C.atomic_json(OUT / 'manifest.json', manifest)
print('FROZEN', json.dumps({k: manifest[k] for k in ['total', 'families', 'shortfall', 'excluded_hand_multisets', 'teacher_attempts',
                                                     'teacher_rejected', 'cases_sha256']}), flush=True)
