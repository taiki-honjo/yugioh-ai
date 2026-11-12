"""Restore the Windows controller script C:\\Users\\hatk8\\AppData\\Local\\YugiohAI\\continuous-elfnotes.ps1.

The original install (2026-11-08, run from the Codex desktop app) was redirected by MSIX file-system
virtualization into the Codex package's LocalCache, so the real path the Startup shortcut and the
(disabled) scheduled task point to never existed. This copies the ORIGINAL bytes back to the real path.
It does not modify the script, does not execute it, does not touch the Startup shortcut or the
scheduled task, does not remove STOP and does not start training. Idempotent; refuses to overwrite a
different file."""
import json, os, re, shutil, subprocess, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); A = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C

WIN_LOCAL = Path('/mnt/c/Users/hatk8/AppData/Local')
VIRTUAL = WIN_LOCAL / 'Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/YugiohAI'
WORKDIR = Path('/mnt/c/Users/hatk8/Documents/Codex/2026-10-27/new-chat/work/continuous-elfnotes')
TARGET = WIN_LOCAL / 'YugiohAI'
STARTUP = Path('/mnt/c/Users/hatk8/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/YugiohAI-Continuous-Elfnotes.lnk')
SCRIPT_SHA = 'ba92a90a82b5d6b9a2f981159a232b597ae038af0805c76301cdd840e63d96ed'
CONFIG = P / 'config/continuous-elfnotes.json'
CONFIG_SHA = '7ba275dcb0402237a262b6080d1506faa11c3b549970cf8860b9d67c7a8dcf51'
D_ROOT = P / 'experiments/continuous-elfnotes-openings-d16384-rehearsal-ready-20261109'
ROOTS = [D_ROOT] + [P / 'experiments' / n for n in ['continuous-elfnotes-openings-d16384-20261109',
                                                   'continuous-elfnotes-openings-v2-20261109',
                                                   'continuous-elfnotes-openings-20261109']]
RESTORE = ['continuous-elfnotes.ps1', 'autostart-mode.txt']  # logs are kept as records only


def stopped():
    out = {}
    for root in ROOTS:
        state = C.read_json(root / 'state.json')
        out[str(root)] = dict(stop=(root / 'STOP').read_text().strip(), status=state['status'], phase=state.get('phase'),
                              pid=state.get('pid'), child_pid=state.get('child_pid'), pending=state.get('pending'))
        assert out[str(root)]['stop'] == 'now' and state['status'] == 'stopped' and state.get('pid') is None \
            and state.get('pending') is None and not state.get('child_pid'), (root, out[str(root)])
    procs = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines()
    learners = [l for l in procs if re.search(r'python[0-9.]* .*(continuous_learning|cleanba|train\.py|opening_rehearsal)', l)]
    assert not learners, learners
    return out


def file_info(path):
    st = path.stat()
    return dict(path=str(path), sha256=C.sha(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


# ---- preconditions: D production config, everything stopped
assert C.sha(CONFIG) == CONFIG_SHA, 'production config changed since the D-ready switch'
cfg = C.read_json(CONFIG)
assert Path(cfg['root']) == D_ROOT and cfg['learning_rate'] == 1.5e-5
assert cfg['rehearsal']['enabled'] and cfg['rehearsal']['split'] == 'train' and cfg['rehearsal']['lambda_bc'] == 5.0
assert (cfg['bootstrap_steps'], cfg['steps_per_cycle']) == (4096, 4096)
before_stop = stopped()

# ---- originals: the virtualized install and the Codex work-dir source must agree byte for byte
assert C.sha(VIRTUAL / 'continuous-elfnotes.ps1') == C.sha(WORKDIR / 'continuous-elfnotes.ps1') == SCRIPT_SHA
originals = {p.name: file_info(p) for p in sorted(VIRTUAL.iterdir()) if p.is_file()}
record_source = A / 'source'
record_source.mkdir(exist_ok=True)
for name in originals:
    copy = record_source / ('virtualized-' + name)
    if not copy.exists():
        shutil.copyfile(VIRTUAL / name, copy)
    assert C.sha(copy) == originals[name]['sha256']
lnk_copy = record_source / 'startup-shortcut.lnk'
if not lnk_copy.exists():
    shutil.copyfile(STARTUP, lnk_copy)
assert C.sha(lnk_copy) == C.sha(STARTUP)

# ---- what the script does (static reading only; nothing is executed)
text = (VIRTUAL / 'continuous-elfnotes.ps1').read_bytes().decode('utf-8')
consts = dict(re.findall(r"^\$(LinuxConfig|LinuxScript|Python)='([^']+)'", text, re.M))
assert consts == {'LinuxScript': str(P / 'scripts/continuous_learning.py'), 'LinuxConfig': str(CONFIG),
                  'Python': '/opt/ygo-ai/.venv/bin/python'}, consts
assert "@('-d','Ubuntu-22.04','-u','hatk8','--exec',$Python,'-B','-u',$LinuxScript)" in text
run_block = text[text.index("if($Action -eq 'Run')"):text.index("if($Action -eq 'Start')")]
assert "'run','--config',$LinuxConfig" in run_block and 'resume' not in run_block and 'STOP' not in run_block
assert text.count('resume --config') == 1 and text.index('resume --config') > text.index("if($Action -eq 'Start')")
lnk = STARTUP.read_bytes().decode('utf-16-le', errors='ignore')
assert r'C:\Users\hatk8\AppData\Local\YugiohAI\continuous-elfnotes.ps1" -Action Run' in lnk

# ---- restore to the real path (byte copy; refuse to overwrite a different file)
target_existed = TARGET.exists()
TARGET.mkdir(exist_ok=True)
restored = {}
for name in RESTORE:
    source, target = VIRTUAL / name, TARGET / name
    if target.exists():
        assert C.sha(target) == C.sha(source), f'refusing to overwrite a different {target}'
    else:
        partial = TARGET / (name + '.restoring')
        shutil.copyfile(source, partial)
        assert C.sha(partial) == C.sha(source)
        os.replace(partial, target)
    restored[name] = file_info(target)
    assert restored[name]['sha256'] == originals[name]['sha256'] and restored[name]['bytes'] == originals[name]['bytes']
assert sorted(p.name for p in TARGET.iterdir()) == sorted(RESTORE)
assert C.sha(STARTUP) == C.sha(lnk_copy) and C.sha(CONFIG) == CONFIG_SHA
after_stop = stopped()

record = dict(
    restored_jst=C.now(),
    cause='The 2026-11-08 Install was run from the Codex desktop app (MSIX package OpenAI.Codex_2p2nqsd0c76g0); MSIX '
          'file-system virtualization redirected %LOCALAPPDATA%\\YugiohAI into the package LocalCache, so the real '
          'C:\\Users\\hatk8\\AppData\\Local\\YugiohAI never existed (scheduled task last result 0x8007010B, directory name invalid).',
    target_directory='C:\\Users\\hatk8\\AppData\\Local\\YugiohAI', target_directory_existed_before=target_existed,
    restored=restored, originals_virtualized=originals,
    workdir_source=file_info(WORKDIR / 'continuous-elfnotes.ps1'),
    script_modified=False, logs_restored=False,
    script_references=dict(consts, wsl_distro='Ubuntu-22.04', wsl_user='hatk8'),
    production_config=dict(path=str(CONFIG), sha256=CONFIG_SHA, root=cfg['root'], learning_rate=cfg['learning_rate'],
                           rehearsal=cfg['rehearsal'], bootstrap_steps=cfg['bootstrap_steps'], steps_per_cycle=cfg['steps_per_cycle'],
                           verified_policy_config=cfg['verified_policy_config']),
    actions=dict(Run='wsl continuous_learning.py run --config <production config>; with STOP present run_loop() records status '
                     'stopped and returns before any training (scripts/continuous_learning.py run_loop); does not remove STOP',
                 Start='calls resume (REMOVES STOP) then launches Run -- not executed',
                 Stop='stop --now', Status='status (read only)', Install='recreates the shortcut -- not executed',
                 DisableAutostart='removes the Startup shortcut (needs autostart-mode.txt) -- not executed'),
    startup_shortcut=dict(path=str(STARTUP), sha256=C.sha(STARTUP), touched=False,
                          runs='powershell -File C:\\Users\\hatk8\\AppData\\Local\\YugiohAI\\continuous-elfnotes.ps1 -Action Run'),
    scheduled_task_touched=False, script_executed=False, training_performed=False, stop_removed=False,
    roots_before=before_stop, roots_after=after_stop)
C.atomic_json(A / 'restore-record.json', record)
print('RESTORED', json.dumps({k: v['sha256'] for k, v in restored.items()}), flush=True)
