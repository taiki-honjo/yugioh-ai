"""v1 vs v2 placement adapter on identical fixed cases (evaluation only; nothing is trained or adopted).

Compares pass/fail, failure reason, the full selected-action trace, placement events and
position events. v1 is read from the production snapshot (read-only); v2 from the new
version directory. Writes only into this experiment directory.
"""
import copy, hashlib, importlib.util, json, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); OUT = Path(__file__).resolve().parent.parent
V1 = P / 'experiments/continuous-elfnotes-openings-20261109/snapshot/policy-code/center_placement.py'
V2 = P / 'evaluation/placement-remix-fix-20261109/code'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0, str(V2))
import all_openings_common as common
import all_opening_cases as bank
import center_placement as cp2
from battle_position import PositionPolicy
spec = importlib.util.spec_from_file_location('center_placement_v1', V1)
cp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp1)
assert sha(V1) == '805e68213710ad411ae4c5a3fc47bd998a8eb7ed2113a3335386eeed7db00b3c'
assert Path(cp2.__file__).resolve() == (V2 / 'center_placement.py').resolve()
codes = common.v.v.engine.ID_CODES
cfg = json.loads((P / 'config/continuous-elfnotes.json').read_text())
policy_cfg = json.loads((P / 'config/latest-verified-opening-policy.json').read_text())
manifest = json.loads(Path(cfg['opening_manifest']).read_text())
validation, seen = [], set()
for case in manifest['decks']['ElfnotesKewlTune']['cases']['validation']:  # same de-duplication as openings_job
    key = (case['eval_family'], tuple(sorted(case['hand'])), case['seed'])
    if key not in seen:
        seen.add(key); validation.append(case)
fresh = json.loads((P / 'evaluation/center-placement-20261108-v4/cases.json').read_text())
models = {'adopted': (Path(policy_cfg['selected_checkpoint']), [('validation-66', validation), ('placement-fixed-44', fresh)]),
          'a-ppo-256-seed10000777': (P / 'experiments/opening-rehearsal-ab-20261109/a-ppo/cycles/256-seed10000777/candidate',
                                     [('validation-66', validation)])}

def stack(module, model):
    return PositionPolicy(module.PlacementPolicy(common.Policy(model / 'base.flax_model', model / 'chain-head.flax_model'), codes), codes)

def run(case, policy):
    r, _ = common.v.run(bank.normalize(case), policy=policy)
    return dict(passed=bool(r['passed']), failure=r['failure'], trace=[x['selected'] for x in r['trace']],
                placement=copy.deepcopy(policy.base.memory.events), positions=copy.deepcopy(policy.positions.events),
                remix_diagnostics=copy.deepcopy(getattr(policy.base.memory, 'diagnostics', [])))

report = dict(v1_sha256=sha(V1), v2_sha256=sha(V2 / 'center_placement.py'), suites=[], differences=[])
for name, (model, suites) in models.items():
    old, new = stack(cp1, model), stack(cp2, model)
    for suite, cases in suites:
        rows = []
        for case in cases:
            a, b = run(case, old), run(case, new)
            same = {k: a[k] == b[k] for k in ['passed', 'failure', 'trace', 'placement', 'positions']}
            rows.append(dict(case_id=case['case_id'], v1_passed=a['passed'], v2_passed=b['passed'], identical=all(same.values()),
                             decisions=len(a['trace']), placement_events=len(a['placement']), v2_remix_diagnostics=b['remix_diagnostics']))
            if not all(same.values()):
                report['differences'].append(dict(model=name, suite=suite, case_id=case['case_id'], fields=[k for k, v in same.items() if not v]))
            print('CASE', name, suite, case['case_id'], a['passed'], b['passed'], rows[-1]['identical'], flush=True)
        report['suites'].append(dict(model=name, checkpoint=str(model), base_sha256=sha(model / 'base.flax_model'),
                                     head_sha256=sha(model / 'chain-head.flax_model'), suite=suite, cases=len(rows),
                                     v1_passed=sum(r['v1_passed'] for r in rows), v2_passed=sum(r['v2_passed'] for r in rows),
                                     identical=sum(r['identical'] for r in rows),
                                     decisions=sum(r['decisions'] for r in rows), placement_events=sum(r['placement_events'] for r in rows),
                                     rows=rows))
report['lua_errors'] = common.v.native.get_lua_error_count()
report['all_identical'] = not report['differences']
(OUT / 'equivalence.json').write_text(json.dumps(report, indent=1) + '\n')
print('EQUIVALENCE', json.dumps({k: report[k] for k in ['all_identical', 'lua_errors']}),
      [(s['model'], s['suite'], s['v1_passed'], s['v2_passed'], s['identical'], s['cases']) for s in report['suites']], flush=True)
