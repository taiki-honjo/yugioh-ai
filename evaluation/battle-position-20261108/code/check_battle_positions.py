"""Compare current placement helper against stats-based battle positions, same fixed cases."""
import collections,copy,json,os,shutil,sqlite3,sys
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.sched_setaffinity(0,{0,2});os.nice(10)
P=Path('/home/hatk8/projects/yugioh-ai');OLD=P/'evaluation/center-placement-20261108-v4';E=P/'evaluation/battle-position-20261108';E.mkdir(exist_ok=False);C=E/'code'
shutil.copytree(OLD/'code',C,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['battle_position.py','check_battle_positions.py']:shutil.copy2(Path(__file__).parent/name,C/name)
db=sqlite3.connect(P/'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
data={str(code):[atk,defense,kind,int(bool(kind&1) and ('攻撃表示' in text or '守備表示' in text))] for code,atk,defense,kind,text in db.execute('select datas.id,atk,def,type,desc from datas join texts on texts.id=datas.id')}
(C/'position_cards.json').write_text(json.dumps(data,separators=(',',':')))
f=C/'all_openings_common.py';f.write_text(f.read_text()+'\nfrom battle_position import install_position_teacher_preferences\ninstall_position_teacher_preferences()\n')
sys.path.insert(0,str(C));from all_openings_common import *
from center_placement import PlacementPolicy
from battle_position import PositionPolicy,PositionDecision
import all_opening_cases as bank
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
cfg=read(P/'config/latest-verified-opening-policy.json');checkpoint=Path(cfg['selected_checkpoint']);neural=Policy(checkpoint/'base.flax_model',checkpoint/'chain-head.flax_model');base=PlacementPolicy(neural,v.v.engine.ID_CODES);after=PositionPolicy(base,v.v.engine.ID_CODES,data)
m=read(B/'manifest.json');teacher=[]
for proto in bank.bank(include_baronne=True):
    c=next(c for c in m['decks']['ElfnotesKewlTune']['cases']['train'] if c['family']==proto['family']);r,_=v.run(bank.normalize(c));teacher.append(dict(family=c['family'],passed=r['passed'],failure=r['failure']));write(E/'teachers'/(c['case_id']+'.json'),r)
    print('TEACHER',c['family'],r['passed'],flush=True)
assert all(x['passed'] for x in teacher)
cases=read(OLD/'cases.json');write(E/'cases.json',cases);rows=[]
for c in cases:
    for mode,policy in [('before',base),('after',after)]:
        r,_=v.run(bank.normalize(c),policy=policy)
        if mode=='after':r['position_preferences']=dict(events=copy.deepcopy(after.positions.events),placement_events=copy.deepcopy(base.memory.events),weights_retrained=False)
        write(E/'evaluation'/mode/(c['case_id']+'.json'),r)
        rows.append(dict(mode=mode,case=c['case_id'],family=c['eval_family'],passed=r['passed'],failure=r['failure']))
        print('ROLLOUT',mode,c['case_id'],r['passed'],flush=True)
# Critical behavioral distinctions, including explicit later-turn attack intentions.
d=PositionDecision(data);tests=[]
for label,code,positions,turn,own,late,intent,tuner,zone,expected in [
 ('track-no-attack',42781164,[1,4],1,True,False,None,False,1,4),
 ('crystal-higher-attack',50954680,[1,4],1,True,False,None,False,1,1),
 ('explicit-later-no-attack',42781164,[1,4],3,True,False,False,False,1,4),
 ('explicit-attack',42781164,[1,4],3,True,False,True,False,1,1),
 ('uncertain-later-main1-keeps-choice',42781164,[1,4],3,True,False,None,False,1,1),
 ('remix-public-tuner-buff',88170262,[1,4],1,True,False,None,True,1,1),
 ('remix-no-public-buff',88170262,[1,4],1,True,False,None,False,1,4),
 ('forced-defense-even-higher-atk',84815190,[4],1,True,False,None,False,1,4),
 ('tie-defense',94145021,[1,4],1,True,False,None,False,1,4),
 ('strel-center-equal-stats',42302563,[1,4],1,True,False,None,False,3,4)]:
    result=d.choose(code,positions,0,turn,own,late,tuner,zone,intent);assert positions[result]==expected,label;tests.append(dict(label=label,passed=True,event=d.events[-1]))
write(E/'decision-checks.json',tests)
events=[ev for f in (E/'evaluation/after').glob('*.json') for ev in read(f)['position_preferences']['events']]
summary=dict(before=sum(x['passed'] for x in rows if x['mode']=='before'),after=sum(x['passed'] for x in rows if x['mode']=='after'),total=44,
    teacher_passed=sum(x['passed'] for x in teacher),teacher_total=26,position_changes=sum(e['changed'] for e in events),
    track_choices=sum(e['code']==42781164 for e in events),track_defense=sum(e['code']==42781164 and e['chosen_position']==4 for e in events),cases=rows,
    lua_errors=v.native.get_lua_error_count(),weights_retrained=False,same_regression_hands_as_center_check=True,base_sha256=sha(checkpoint/'base.flax_model'),head_sha256=sha(checkpoint/'chain-head.flax_model'),source_checkpoint=str(checkpoint),server_replays_saved=0)
assert summary['lua_errors']==0;write(E/'summary.json',summary);print('COMPLETE',json.dumps({k:x for k,x in summary.items() if k!='cases'}),flush=True)
