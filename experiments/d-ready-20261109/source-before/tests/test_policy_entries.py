"""Both Elfnotes inference entries load the adapter version and weights of the CURRENT production
config, play a verified opening, and refuse a process where another adapter version is already
imported. Each check runs in a fresh process (CPU); nothing is trained or written."""
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

P = Path('/home/hatk8/projects/yugioh-ai')
CFG = json.loads((P / 'config/continuous-elfnotes.json').read_text())
POLICY = json.loads(Path(CFG['verified_policy_config']).read_text())
STATE = json.loads((Path(CFG['root']) / 'state.json').read_text())
V1_CODE = P / 'evaluation/battle-position-20261108/code'


def probe(entry, preload_v1=False, case_id='rukina-pass-g08-p0'):
    script = textwrap.dedent(f'''
        import hashlib, json, sys
        from pathlib import Path
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        sys.path.insert(0, {str(P / 'scripts')!r})
        if {preload_v1!r}:
            sys.path.insert(0, {str(V1_CODE)!r}); import center_placement; sys.path.pop(0)
        import {entry} as entry
        try:
            policy = entry.create_policy()
        except RuntimeError as exc:
            print('RESULT', json.dumps(dict(refused=str(exc)))); raise SystemExit(0)
        mods = {{n: sys.modules[n].__file__ for n in ['center_placement', 'battle_position', 'all_openings_common']}}
        common = sys.modules['all_openings_common']
        import all_opening_cases as cases
        manifest = json.loads(Path({CFG['opening_manifest']!r}).read_text())
        case = next(c for c in manifest['decks']['ElfnotesKewlTune']['cases']['validation'] if c['case_id'] == {case_id!r})
        result, _ = common.v.run(cases.normalize(case), policy=policy)
        base = policy.base.base
        print('RESULT', json.dumps(dict(modules=mods, center_placement_sha256=sha(mods['center_placement']),
              chain=[type(policy).__name__, type(policy.base).__name__, type(base).__name__],
              passed=bool(result['passed']), lua_errors=common.v.native.get_lua_error_count())))
    ''')
    done = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=600,
                          env={'JAX_PLATFORMS': 'cpu', 'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1',
                               'HOME': str(Path.home())})
    line = next((l for l in done.stdout.splitlines() if l.startswith('RESULT ')), None)
    if line is None:
        raise AssertionError(done.stdout[-2000:] + done.stderr[-4000:])
    return json.loads(line[7:])


class EntryTests(unittest.TestCase):
    def check(self, found, code_root):
        self.assertEqual(found['center_placement_sha256'], POLICY['placement_adapter']['center_placement_sha256'])
        for path in found['modules'].values():
            self.assertTrue(Path(path).resolve().is_relative_to(Path(code_root).resolve()), path)
        self.assertEqual(found['chain'], ['PositionPolicy', 'PlacementPolicy', 'Policy'])
        self.assertTrue(found['passed'])
        self.assertEqual(found['lua_errors'], 0)

    def test_continuous_entry_uses_the_production_root_snapshot(self):
        self.assertEqual(STATE['status'], 'stopped')
        self.assertTrue((Path(CFG['root']) / 'STOP').exists())
        self.check(probe('continuous_opening_policy'), Path(CFG['root']) / 'snapshot/policy-code')
        snapshot = json.loads((Path(CFG['root']) / 'snapshot/hashes.json').read_text())
        self.assertEqual(snapshot['policy-code/center_placement.py'], POLICY['placement_adapter']['center_placement_sha256'])

    def test_verified_entry_uses_the_production_policy_config(self):
        self.check(probe('verified_opening_policy'), POLICY['placement_adapter']['runtime_code_directory'])

    def test_both_entries_use_the_adopted_weights(self):
        import hashlib
        best = Path(STATE['best_checkpoint'])
        for name, key in [('base.flax_model', 'base_sha256'), ('chain-head.flax_model', 'head_sha256')]:
            digest = hashlib.sha256((best / name).read_bytes()).hexdigest()
            self.assertEqual(digest, POLICY[key])
            self.assertEqual(digest, hashlib.sha256((Path(POLICY['selected_checkpoint']) / name).read_bytes()).hexdigest())

    def test_a_process_with_another_adapter_version_is_refused(self):
        for entry in ['continuous_opening_policy', 'verified_opening_policy']:
            found = probe(entry, preload_v1=True)
            self.assertIn('use a fresh process', found.get('refused', ''), entry)


if __name__ == '__main__':
    unittest.main(verbosity=2)
