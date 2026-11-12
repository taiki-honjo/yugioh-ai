"""66-case opening retention of the single B2B-trial candidate (one job, 66 episodes; the evaluation gate's job-openings).

Baseline: the existing adopted-model result in the production root (opening-baseline.json, 65/66), reused because the
weights, the cases (opening manifest), the evaluation code / policy code and the adapters are identical (only the
embedded root path in verify_opponent_article_basics.py differs). Retention is judged on the set of passing case_ids,
not the count. The experiment root's STOP is lifted only while the job runs. Refuses to run twice.
"""
import json
from pathlib import Path
import sys

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-train-20261109'
ROOT = OUT / 'root'
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C  # noqa: E402


def main():
    result_path = OUT / 'retention/opening-retention.json'
    assert not result_path.exists(), 'retention already evaluated'
    cand = C.read_json(OUT / 'candidate.json')
    candidate = Path(cand['candidate'])
    assert C.sha(candidate / 'base.flax_model') == cand['base_sha256'] and C.sha(candidate / 'chain-head.flax_model') == cand['head_sha256']
    cfg = C.load_config(OUT / 'config.json')
    C.runtime_check(cfg)
    prod_cfg = json.loads((P / 'config/continuous-elfnotes.json').read_text())
    baseline_path = Path(prod_cfg['root']) / 'opening-baseline.json'
    baseline = C.read_json(baseline_path)
    state = C.read_json(ROOT / 'state.json')
    directory = OUT / 'retention'
    command = [cfg['python'], '-B', '-u', str(P / 'scripts/continuous_learning.py'), 'job-openings', '--config', cfg['_file'],
               '--candidate', str(candidate), '--result', str(result_path)]
    assert (ROOT / 'STOP').read_text() == 'now'
    (ROOT / 'STOP').unlink()
    try:
        C.run_job(cfg, state, command, directory, 'opening-retention')
    finally:
        (ROOT / 'STOP').write_text('now')
        C.save_state(cfg, C.read_json(ROOT / 'state.json'), child_pid=None)
    got = C.read_json(result_path)
    base_ids, new_ids = set(baseline['passed_case_ids']), set(got['passed_case_ids'])
    summary = dict(evaluated_jst=C.now(), episodes=got['total'], candidate=str(candidate), candidate_base_sha256=cand['base_sha256'],
                   candidate_head_sha256=cand['head_sha256'], baseline=dict(path=str(baseline_path), sha256=C.sha(baseline_path), passed=baseline['passed'],
                   total=baseline['total'], checkpoint=baseline['checkpoint']),
                   candidate_passed=got['passed'], total=got['total'], lua_errors=got['lua_errors'], status=got['status'],
                   lost_case_ids=sorted(base_ids - new_ids), gained_case_ids=sorted(new_ids - base_ids),
                   all_baseline_cases_retained=base_ids <= new_ids, holdout_used=got['holdout_used'],
                   normal_summon_usage_scored=got['normal_summon_usage_scored'])
    C.atomic_json(OUT / 'retention/retention-summary.json', summary)
    print('RETENTION', json.dumps({k: summary[k] for k in ['candidate_passed', 'total', 'lua_errors', 'lost_case_ids', 'gained_case_ids', 'all_baseline_cases_retained']}))


if __name__ == '__main__':
    main()
