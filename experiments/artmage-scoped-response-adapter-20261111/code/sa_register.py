"""registration.json (before prepare / training / engine runs) and amendment-1.json (fixes the cache + prechecks, before training).

  sa_register.py register
  sa_register.py fix-cache
"""
import filecmp
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-scoped-response-adapter-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import sa_runs as SR  # noqa: E402
import sa_model as SM  # noqa: E402
import rl_data as RD  # noqa: E402
import artmage_entry as E  # noqa: E402

P = Path('/home/hatk8/projects/yugioh-ai')
HB = P / 'experiments/artmage-hand-branch-20261110'
FLC = P / 'experiments/artmage-functional-landing-compare-20261110'
NORET = P / 'experiments/artmage-response-learning-20261111'
RET = P / 'experiments/artmage-response-context-retention-20261111'
JST = timezone(timedelta(hours=9))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_files(root):
    out = {}
    for site in SR.A_SITES:
        d = root / 'runs' / f'A-{site}'
        out[f'A-{site}'] = sha(d / 'record.json')
    for n in SR.B_NAMES:
        out[f'B-{n}'] = sha(root / 'runs' / f'B-{n}' / 'record.json')
    for n in SR.C_NAMES:
        out[f'C-{n}'] = sha(root / 'eval/retain/candidate' / f'{n}.json')
    return out


def register():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    pins = {k: v for k, v in E.verify_pins().items()}
    prev_pins = json.loads((NORET / 'registration.json').read_text())['pins']
    assert pins == prev_pins
    frozen = {n: dict(copy=str(ROOT / 'models/frozen' / n), sha256=sha(ROOT / 'models/frozen' / n), source=str(HB / 'train/candidate' / n),
                      byte_identical_to_source=filecmp.cmp(ROOT / 'models/frozen' / n, HB / 'train/candidate' / n, shallow=False))
              for n in ['base.flax_model', 'chain-head.flax_model']}
    assert all(v['byte_identical_to_source'] for v in frozen.values()) and (frozen['base.flax_model']['sha256'], frozen['chain-head.flax_model']['sha256']) == SM.FROZEN_SHA
    start_files = {f'A/B-{n}': sha(FLC / 'runs' / n / 'record.json') for n in SR.B_NAMES}
    start_files.update({f'C-{n}': sha(HB / 'eval/retain/candidate' / f'{n}.json') for n in SR.C_NAMES})
    reg = dict(
        status='registered_before_prepare_training_and_engine_runs', registered_jst=datetime.now(JST).isoformat(timespec='seconds'),
        spec=dict(copy='spec/artmage-scoped-response-adapter-plan-2026-11-11.md', sha256=sha(ROOT / 'spec/artmage-scoped-response-adapter-plan-2026-11-11.md')),
        scope_audit=dict(copy='spec/artmage-response-scope-audit-2026-11-11.json', sha256=sha(ROOT / 'spec/artmage-response-scope-audit-2026-11-11.json'),
                         reproduced='audit_artmage_response_scope.py re-run to a new output: byte-identical (13 retention trajectories: 0 gated own decisions; S1/S2 targets gated)'),
        frozen=frozen, frozen_note='never in the optimizer; batch_stats untouched; never re-serialised (candidate base/head are byte copies)',
        policy=dict(type=SM.TYPE, version=SM.VERSION, files=['base.flax_model (frozen)', 'chain-head.flax_model (frozen)', 'response-adapter.flax_model (new)', 'manifest.json'],
                    factory='sa_model.create_adapter_policy (refuses a missing file, other type/version, any hash mismatch, a base/head other than the frozen pair)',
                    inference='frozen Policy.infer exactly as Policy.act (same object and jit); if gated: logits + adapter(features(info), actions_); else the frozen logits are returned',
                    gate='actions_[0, 3] == 2 (select_chain) AND public chain length >= 1 AND features(info) role == 2; role = the latest chain link player '
                         '(ygopro.h: public_chain_.back()[1]) compared with to_play; no card ids',
                    adapter='opponent_article_bc_common.ChainHead, new parameters, init PRNGKey(11111501), zero output layer (delta exactly 0 at start); stateless',
                    limit='the adapter corrects the frozen state-dependent logits from the public chain and action features only; it does not read the board',
                    guarantee='outside the gate, the same input/history/backend gives the original logits; a different response changes the rest of the duel normally'),
        data=dict(manifest=str(RD.MANIFEST), sha256=sha(RD.MANIFEST), targets='response-S1 (step 149, ディアクトロス②), response-S2 (step 169, ファインメルト③)',
                  cache='cache/targets.npz from the frozen Policy along the saved histories on the CPU (the evaluation backend); fixed in amendment-1 before training'),
        training=dict(updates=512, batch='S1 + S2 every update', updated='the new adapter only', optimizer='fresh Adam, lr 1e-3, eps 1e-5',
                      clip='global norm 1.0 on the adapter', seed=11111501, loss='legal-option CE of frozen logits + delta, mean over S1/S2',
                      not_used=['PPO', 'value', 'entropy', 'KL', 'old 40 sequences (diagnostics only)'], device='CPU',
                      logged='per update: loss, per-sequence CE, pre-clip gradient norm, clip flag, update norm, max |delta|; before/after: P(target), P(pass), CE, delta',
                      no=['coefficient / lr / update changes after seeing results', 'more candidates', 'intermediate selection']),
        prechecks=['zero adapter vs the original entry on all 42 sequences: RNN states, final logits, post-position-adapter actions identical',
                   'gate from public features == gate from saved snapshots on saved runs', 'synthetic non-zero adapter: ignored off-target, used on-target',
                   'factory refusals (hash mismatch, version, missing file)', 'after training: off-target logits/actions identical; reload equality; finite weights'],
        evaluation=dict(same_as='docs/artmage-response-learning-plan-2026-11-11.md (A 4, B 8, C 13)',
                        runner='sa_runs.py (code-changes-sa_runs.diff): the 3-file factory (sa_model.AdapterEntry) is passed explicitly to every runner; '
                               'each record carries the 3 hashes, type/version and the adapter log',
                        compared=dict(start_model=start_files, no_retention_candidate=run_files(NORET), retention_candidate=run_files(RET)),
                        checks=['1. A: S1 stopped by the candidate itself, S2 stop kept',
                                '2. B: the pre-intervention development / reshaping stays the start model\'s, S1 stopped, the two 2-cost stops kept '
                                '(after a different in-scope response, path differences are normal and separated)',
                                '3. C: the 12 start-model successes kept; any adapter activation investigated against the gate and the old record; '
                                'F3-V not in the denominator; F2-V / F4-V recovery and new successes shown separately'],
                        checks_note='local observation points, not production acceptance conditions'),
        engine=dict(cap=28, plan='A 4 + B 8 + C 13 = 25', spare='3 only for re-running after a harness fix that keeps the conditions', failures_count=True,
                    previous_unused_not_carried=True),
        not_done=['PPO', 'normal battles', 'adoption', 'acceptance changes', 'keep-vs-reshape re-test', 'B2B', 'production / registry changes'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, jobs=1),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
        code_changes=dict(sa_runs=dict(file='code-changes-sa_runs.diff', sha256=sha(ROOT / 'code-changes-sa_runs.diff'))),
    )
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=str) + '\n')
    print('REGISTERED', sha(path))


def fix_cache():
    path = ROOT / 'amendment-1.json'
    if path.exists():
        raise SystemExit('amendment-1 exists')
    if (ROOT / 'train/updates.jsonl').exists():
        raise SystemExit('training started')
    reg = json.loads((ROOT / 'registration.json').read_text())
    prep = json.loads((ROOT / 'cache/prepare.json').read_text())
    if not prep['all_prechecks_passed']:
        raise SystemExit('prechecks failed: report before continuing')
    cache = ROOT / 'cache/targets.npz'
    assert prep['cache']['sha256'] == sha(cache)
    am = dict(amendment=1, registered_jst=datetime.now(JST).isoformat(timespec='seconds'), registration_sha256=sha(ROOT / 'registration.json'),
              before_training=True, change='none to code or conditions; fixes the target cache and the precheck results',
              cache=dict(file=str(cache), sha256=sha(cache), source=prep['cache']['source'], backend=prep['backend'], frozen=prep['frozen_sha256']),
              prechecks=dict(file=str(ROOT / 'cache/prepare.json'), sha256=sha(ROOT / 'cache/prepare.json'), passed=True), code=reg['code'])
    path.write_text(json.dumps(am, ensure_ascii=False, indent=1) + '\n')
    print('AMENDMENT', sha(path))


if __name__ == '__main__':
    {'register': register, 'fix-cache': fix_cache}[sys.argv[1]]()
