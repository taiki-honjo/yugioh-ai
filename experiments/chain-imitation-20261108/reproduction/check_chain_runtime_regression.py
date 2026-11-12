from pathlib import Path
import runpy
P=Path('/home/hatk8/projects/yugioh-ai');B=P/'experiments/chain-imitation-20261108'
W=Path('/mnt/c/Users/hatk8/Documents/Codex/2026-10-27/new-chat/work')
source=(W/'check_combo_runtime_regression.py').read_text()
source=source.replace("b=p/'experiments/combo-benchmark-20261107'","b=p/'experiments/chain-imitation-20261108'")
path=B/'check-normal-regression-used.py';path.write_text(source)
runpy.run_path(str(path),run_name='__main__')
