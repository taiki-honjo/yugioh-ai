import collections,hashlib,importlib.util,json,sys
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai');ROOT=P/'evaluation/all-deck-position-20261109';E=ROOT/'verified'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(ROOT/'registry.json');spec=importlib.util.spec_from_file_location('shared_position',ROOT/'code/battle_position.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);data=m.load_card_data()
d=m.PositionDecision(data);checks=[]
for label,code,positions,turn,own,late,intent,expected in [
    ('Litera-first-turn',97434754,[1,4],1,True,False,None,4),
    ('Meteor-opponent-turn',93172951,[1,4],2,False,False,None,4),
    ('Devils-fixed-own-stats',91237821,[1,4],1,True,False,None,4),
    ('Faimena-no-attack-plan',1498449,[1,4],3,True,False,False,4),
    ('Faimena-explicit-attack',1498449,[1,4],3,True,False,True,1),
    ('Faimena-uncertain-later-main1',1498449,[1,4],3,True,False,None,1),
    ('Faimena-main2',1498449,[1,4],3,True,True,None,4),
    ('Roze-tie',37351133,[1,4],1,True,False,None,4),
    ('Kagari-forced-attack',63288573,[1],1,True,False,None,1),
    ('Nerva-forced-defense',53589300,[4],1,True,False,None,4),
    ('position-dependent-effect-preserved',42759961,[1,4],1,True,False,None,1),
    ('variable-attack-effect-preserved',89851827,[1,4],1,True,False,None,1)]:
    result=d.choose(code,positions,0,turn,own,late,attack_intent=intent)
    assert positions[result]==expected,label;checks.append(dict(label=label,passed=True,event=d.events[-1]))
assert [1,4,6][d.choose(1498449,[1,4,6],2,1,True,False)]==6
class Dummy:
    def reset(self):pass
p=m.PositionPolicy(Dummy(),{},data);p.set_attack_intent(True);p.reset();assert p.attack_intent is None
write(E/'decision-checks.json',dict(cases=checks,concealed_choice_preserved=True,attack_intent_reset=True))
summary=read(E/'summary.json');counts={};audit=[]
for key in cfg['decks']:
    paths=list((E/'regression'/key/'after').glob('*.json')) if key in ['Artmage','ClownTroupe'] else [E/'controlled'/(key+'.json')]
    events=[];checked=0
    for f in paths:
        r=read(f);events.extend(r['events']);trace=r['trace']
        for index,t in enumerate(trace):
            if t['snapshot']['message']!='select_position' or t['snapshot']['to_play']!=0:continue
            a=t['selected'];code=a['code'];position=a['position']
            old=sum(c['controller']==0 and c['location']==4 and c['code']==code and c['position']==position for c in t['snapshot']['cards'])
            snapshots=[x['snapshot'] for x in trace[index+1:]]
            if 'final' in r:snapshots.append(r['final'])
            if r.get('arrival'):snapshots.append(dict(cards=[r['arrival']]))
            arrived=any(sum(c['controller']==0 and c['location']==4 and c['code']==code and c['position']==position for c in s['cards'])>old for s in snapshots)
            assert arrived,(key,f.name,code,position)
            checked+=1;audit.append(dict(deck=key,case=f.stem,code=code,chosen_position=position,native_arrival_verified=True))
    codes=collections.Counter((e['code'],e['chosen_position']) for e in events)
    counts[key]=dict(position_choices=len(events),changed=sum(e['changed'] for e in events),native_arrivals_verified=checked,
        selections=[dict(code=c,position=p,count=n) for (c,p),n in sorted(codes.items())])
    if key in ['Artmage','ClownTroupe']:
        for mode in ['before','after']:
            results=[r for r in summary['regression'] if r['deck']==key and r['mode']==mode]
            assert len(results)==4 and all(r['passed'] for r in results)
            counts[key][mode]=sum(r['passed'] for r in results)
        teacher=read(E/'teachers'/(key+'.json'));assert teacher['passed'];counts[key]['teacher_passed']=True
assert all(r['passed'] for r in summary['controlled']) and summary['lua_errors']==0
assert summary['resources']['live_duels']==0
for entry in cfg['decks'].values():
    assert sha(Path(entry['base']))==entry['base_sha256'];assert sha(Path(entry['deck']))==entry['deck_sha256']
    if entry['head']:assert sha(Path(entry['head']))==entry['head_sha256']
assert sha(ROOT/'code/battle_position.py')==cfg['adapter_sha256'];assert sha(ROOT/'code/position_cards.json')==cfg['card_data_sha256']
old_root=Path('/mnt/c/Users/hatk8/Documents/Codex/2026-10-27/new-chat/outputs');preserved={}
for name in ['all-openings-training-2026-11-08','center-placement-2026-11-08','battle-position-2026-11-08']:
    old=old_root/name;files=read(old/'SHA256-manifest.json')
    for rel,meta in files.items():assert sha(old/rel)==meta['sha256'],(name,rel)
    preserved[name]=len(files)
elf=read(P/'config/latest-verified-opening-policy.json');checkpoint=Path(elf['selected_checkpoint'])
assert sha(checkpoint/'base.flax_model')==elf['base_sha256'];assert sha(checkpoint/'chain-head.flax_model')==elf['head_sha256']
write(E/'native-position-audit.json',audit)
report=dict(passed=True,decks=counts,decision_checks=len(checks),native_position_arrivals=len(audit),
    weights_retrained=False,source_weights_and_decks_preserved=True,old_deliveries_preserved=preserved,
    original_elfnotes_policy_preserved=True,lua_errors=0,server_replays_saved=0,
    scope=summary['scope'])
write(E/'verification.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
