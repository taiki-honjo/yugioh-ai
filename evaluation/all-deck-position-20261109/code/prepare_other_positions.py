"""Freeze a shared adapter and the four current opponent policies, without training."""
import hashlib,json,shutil,sqlite3
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai');W=Path(__file__).parent
E=P/'evaluation/all-deck-position-20261109';C=E/'code';C.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
source=P/'evaluation/battle-position-20261108/code/battle_position.py'
code=source.read_text().split('\ndef install_position_teacher_preferences():')[0]
code=code.replace("elif meta[3]:reason='position-dependent-effect-model-choice'", "elif meta[3]:reason='position-dependent-effect-model-choice'\n        elif len(meta)>4 and meta[4]:reason='unresolved-stat-effect-model-choice'")
code=code.replace('if self.base.memory.events:', "if getattr(self.base,'memory',None) and self.base.memory.events:")
code=code.replace('def reset(self):self.base.reset();self.positions.reset()', 'def reset(self):self.base.reset();self.positions.reset();self.attack_intent=None')
# Do not change the already verified Elfnotes-specific implementation.
(C/'battle_position.py').write_text(code)
db=sqlite3.connect(P/'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
data={str(i):[a,d,t,int(bool(t&1) and ('攻撃表示' in text or '守備表示' in text)),int('攻撃力' in text or '守備力' in text)]
      for i,a,d,t,text in db.execute('select datas.id,atk,def,type,desc from datas join texts using(id)')}
# Reviewed text: these modify other cards, not their own summon-entry stats.
# Finmel halves opposing ATK; Hat lowers opposing DEF; Devils boosts A-summoned
# monsters (not its own Xyz stats); Nibiru sets the separate Token's stats.
for code in [34541940,19491080,91237821,27204311]:data[str(code)][4]=0
(C/'position_cards.json').write_text(json.dumps(data,separators=(',',':')))
M=Path((P/'experiments/meta-opponents-20261107/latest-run.txt').read_text().strip());meta=read(M/'manifest.json')
B=Path((P/'experiments/opponent-article-training-20261108/latest-run.txt').read_text().strip())
registry=dict(version=1,weights_retrained=False,code_directory=str(C),adapter_sha256=sha(C/'battle_position.py'),
    card_data_sha256=sha(C/'position_cards.json'),model_code_directory=str(B/'code'),
    model_code_sha256=sha(B/'code/opponent_article_bc_common.py'),vocabulary=meta['vocabulary'],vocabulary_sha256=sha(meta['vocabulary']),
    runtime=str(P/'evaluation/opponent-article-basics-20261108/runtime'),decks={},
    scope='Four registered opponent decks; Elfnotes remains on its separately verified loader. Old frozen evaluators are unchanged.',
    policy_rule='Compare public ATK/DEF for legal face-up summon positions when no attack is planned; ties defend. Preserve uncertain attack plans, forced positions, concealed choices and unresolved card/stat effects.')
native=Path(registry['runtime'])/'agent/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so'
registry['native_sha256']=sha(native)
for r in meta['recipes']:
    key=r['key'];head=None
    if key in ['Artmage','ClownTroupe']:
        selected=Path(read(B/'training'/key/'result.json')['selected_checkpoint'])
        base=selected/'base.flax_model';head=selected/'chain-head.flax_model'
    else:base=Path(read(M/'training'/key/'phase2-65536/result.json')['trained_checkpoint'])
    registry['decks'][key]=dict(label=r['label'],base=str(base),base_sha256=sha(base),
        head=str(head) if head else None,head_sha256=sha(head) if head else None,
        deck=r['ydk'],deck_sha256=sha(r['ydk']),positions_enabled=True)
write(E/'registry.json',registry)
shutil.copy2(W/'deck_policy.py',P/'scripts/deck_policy.py')
shutil.copy2(W/'prepare_other_positions.py',C/'prepare_other_positions.py')
write(P/'config/latest-deck-position-policies.json',registry)
print(json.dumps(dict(prepared=str(E),decks=list(registry['decks'])),ensure_ascii=False))
