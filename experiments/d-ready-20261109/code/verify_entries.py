"""Load both inference entries in fresh processes and compare what they actually loaded:
the numeric base/head parameters (hash of the loaded arrays), the files they came from, and the
placement/position adapter modules. Read-only; nothing is trained or written except the report."""
import json, subprocess, sys, textwrap
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); A = Path(__file__).resolve().parent.parent

PROBE = textwrap.dedent('''
    import hashlib, json, sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, {scripts!r})
    import {entry} as entry
    policy = entry.create_policy()
    neural = policy.base.base
    def digest(tree):
        h = hashlib.sha256()
        def walk(v, prefix=''):
            if isinstance(v, dict):
                for k in sorted(v): walk(v[k], prefix + '/' + k)
            else:
                h.update(prefix.encode()); h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
        walk(tree); return h.hexdigest()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    mods = {{n: sys.modules[n].__file__ for n in ['center_placement', 'battle_position', 'all_openings_common']}}
    print('RESULT', json.dumps(dict(entry={entry!r}, chain=[type(policy).__name__, type(policy.base).__name__, type(neural).__name__],
        loaded_base_params=digest(neural.base), loaded_head_params=digest(neural.head),
        modules=mods, center_placement_sha256=sha(mods['center_placement']), battle_position_sha256=sha(mods['battle_position']))))
''')
cfg = json.loads((P / 'config/continuous-elfnotes.json').read_text())
policy = json.loads(Path(cfg['verified_policy_config']).read_text())
state = json.loads((Path(cfg['root']) / 'state.json').read_text())
rows = {}
for entry in ['continuous_opening_policy', 'verified_opening_policy']:
    done = subprocess.run([sys.executable, '-B', '-c', PROBE.format(scripts=str(P / 'scripts'), entry=entry)], capture_output=True, text=True,
                          timeout=600, env={'JAX_PLATFORMS': 'cpu', 'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1', 'HOME': str(Path.home())})
    line = next(l for l in done.stdout.splitlines() if l.startswith('RESULT '))
    rows[entry] = json.loads(line[7:])
c, v = rows['continuous_opening_policy'], rows['verified_opening_policy']
import hashlib
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = dict(
    production_config_sha256=sha(P / 'config/continuous-elfnotes.json'), production_root=cfg['root'],
    verified_policy_config=cfg['verified_policy_config'],
    continuous_entry_checkpoint=state['best_checkpoint'],
    continuous_entry_files=dict(base=sha(Path(state['best_checkpoint']) / 'base.flax_model'), head=sha(Path(state['best_checkpoint']) / 'chain-head.flax_model')),
    verified_entry_checkpoint=policy['selected_checkpoint'],
    verified_entry_files=dict(base=sha(Path(policy['selected_checkpoint']) / 'base.flax_model'), head=sha(Path(policy['selected_checkpoint']) / 'chain-head.flax_model')),
    entries=rows,
    same_loaded_base_params=c['loaded_base_params'] == v['loaded_base_params'],
    same_loaded_head_params=c['loaded_head_params'] == v['loaded_head_params'],
    same_placement_adapter=c['center_placement_sha256'] == v['center_placement_sha256'] == policy['placement_adapter']['center_placement_sha256'],
    same_position_adapter=c['battle_position_sha256'] == v['battle_position_sha256'])
report['files_match_new_adopted_pair'] = (report['continuous_entry_files'] == report['verified_entry_files'] ==
                                         dict(base=policy['base_sha256'], head=policy['head_sha256']))
(A / 'entry-check.json').write_text(json.dumps(report, indent=1) + '\n')
print(json.dumps({k: report[k] for k in ['same_loaded_base_params', 'same_loaded_head_params', 'same_placement_adapter',
                                         'same_position_adapter', 'files_match_new_adopted_pair']}))
print(json.dumps(dict(continuous=c['modules']['center_placement'], verified=v['modules']['center_placement'],
                      files=report['continuous_entry_files'])))
