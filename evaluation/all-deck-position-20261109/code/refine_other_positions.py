"""Finalize public-text guards; retain the initial regression report."""
import hashlib,json,shutil
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai');E=P/'evaluation/all-deck-position-20261109';C=E/'code';W=Path(__file__).parent
cfg=json.loads((E/'registry.json').read_text());(E/'registry-initial-guard.json').write_text(json.dumps(cfg,indent=2)+'\n')
f=C/'battle_position.py';f.write_text(f.read_text().replace('def reset(self):self.base.reset();self.positions.reset()', 'def reset(self):self.base.reset();self.positions.reset();self.attack_intent=None'))
f=C/'position_cards.json';data=json.loads(f.read_text())
for code in [34541940,19491080,91237821,27204311]:data[str(code)][4]=0
f.write_text(json.dumps(data,separators=(',',':')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg['adapter_sha256']=sha(C/'battle_position.py');cfg['card_data_sha256']=sha(C/'position_cards.json')
cfg['reviewed_fixed_entry_stat_cards']=[34541940,19491080,91237821,27204311]
for f in [E/'registry.json',P/'config/latest-deck-position-policies.json']:f.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
for name in ['prepare_other_positions.py','refine_other_positions.py','check_other_positions.py']:shutil.copy2(W/name,C/name)
shutil.copy2(W/'deck_policy.py',P/'scripts/deck_policy.py');shutil.copy2(W/'deck_policy.py',C/'deck_policy.py')
print('Updated shared adapter guards; original Elfnotes code untouched.')
