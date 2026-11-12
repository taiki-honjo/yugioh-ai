import sys,json,itertools,random,copy,collections,gc,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from chain_imitation_common import *
from chain_counter_cases import definition,run_counter
OUT=B/'dataset';OUT.mkdir(exist_ok=False)
POOL=[14558127,97268402,94145021,23434538,59438930,52038441]
def groups(n):
    limits=collections.Counter(c.main)
    v=[list(x) for x in itertools.combinations_with_replacement(POOL,n)
       if all(k<=limits[card] for card,k in collections.Counter(x).items())]
    old=sorted(c.FILL if n==4 else c.FILL[:3]);v=[x for x in v if sorted(x)!=old]
    random.Random(811800+n).shuffle(v);return v
G={3:groups(3),4:groups(4)}
split_groups={'train':range(0,10),'validation':range(10,13),'holdout':range(13,19)}
templates=c.fixture_definitions();specs={s:[] for s in split_groups}
for split,indices in split_groups.items():
    for template in templates:
        n=len(template['protected'])
        for index in indices:
            for perm in range(2 if split=='train' else 1):
                case=copy.deepcopy(template);fill=G[n][index]
                starters=case['hand'][:-len(case['protected'])]
                case['hand']=starters+fill;random.Random(90000+index*7+perm).shuffle(case['hand'])
                case['protected']=list(fill);case['seed']=90000+index*7+perm
                if case.get('field'):
                    case['field'][0]['sequence']=[0,2,4][index%3] if split=='train' else [1,3][index%2]
                case['case_id']=case['id']+f'-g{index:02d}-p{perm}'
                case['group_id']=f'fill{n}-'+','.join(map(str,sorted(fill)))
                specs[split].append(case)
def fingerprint(case):
    return (tuple(sorted(collections.Counter(case['hand']).items())),tuple((x['code'],x['location'],x['sequence'],x['position']) for x in case.get('field',[])))
sets={s:{fingerprint(ca) for ca in cases} for s,cases in specs.items()}
assert all(not sets[a]&sets[b] for a,b in itertools.combinations(sets,2))
old_cases=json.loads((P/'experiments/combo-benchmark-20261107/evaluation-v1/cases.json').read_text())
assert all(not {fingerprint(x) for x in old_cases}&sets[s] for s in sets)
counter_specs={s:[] for s in specs};seen=set()
train_kinds=['self_rukina','opp_rukina','self_regina','opp_regina_offturn','self_regina_offturn']
for split in specs:
    kinds=train_kinds if split!='holdout' else train_kinds+['self_mix','opp_mix','self_track','opp_track']
    for kind in kinds:
        wanted=4 if split=='train' else 2
        index=0 if split=='train' else 100 if split=='validation' else 1000
        made=0
        while made<wanted:
            case=definition(kind,index);index+=1
            fp=(tuple(tuple(sorted(collections.Counter(case['hands'][p]).items())) for p in [0,1]),
                tuple((x['code'],x['controller'],x['sequence']) for x in case['field']))
            if fp in seen:continue
            seen.add(fp);counter_specs[split].append(case);made+=1
manifest=dict(combo_cases=specs,counter_cases=counter_specs,split_rule='disjoint filler multisets; all hand orders and duplicate goal prefixes grouped; counter initial states disjoint',
              original_24_diagnostic_cases_not_training=True,heldout_sources=['Kewl Tune Mix','Kewl Tune Trackmaker'],
              labels='machine-replayed feasible basic witnesses; no claim of expert-optimal play',
              training_plan=dict(updates=600,batch_size=4,base_learning_rate=3e-5,head_learning_rate=3e-4,
                                 kl_weight=0.01,checkpoint_selection='minimum validation teacher-forcing CE, never holdout',
                                 seed=8118),runtime=json.loads((B/'runtime.json').read_text()),original_model=original_path() if 'original_path' in globals() else None)
(OUT/'split-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
def collect_combo(case):
    env,obs,info,_=c.create_env(case);teacher=c.Witness(case);examples=[];trace=[]
    try:
        for step in range(128):
            s=c.native.benchmark_snapshot();collect_snapshot_checks(obs,info,s)
            if c.goal_check(case,s)['pass_']:return examples,trace
            assert s['turn_player']==0 and s['phase'] in [1,2,4]
            a=teacher(s) if s['to_play']==0 else c.passive(s)
            if s['to_play']==0:
                examples.append(dict(obs={k:np.asarray(obs[k])[0].copy() for k in ['cards_','global_','actions_','h_actions_']},
                                     chain=chain_features(info)[0].copy(),action=int(a),weight=1.0))
            trace.append(dict(message=s['message'],player=s['to_play'],selected=s['actions'][a],public_chain=s['public_chain']))
            obs,_,term,trunc,info=env.step(np.array([a],dtype=np.int32));assert not term[0] and not trunc[0]
        raise RuntimeError('Witness did not finish '+case['case_id'])
    finally:env.close();del env;gc.collect();assert c.native.get_env_resource_stats()['live_duels']==0
def pack(trajectories,ids,path):
    lengths=[len(t) for t in trajectories];T=max(lengths);assert T<=48,T
    data={}
    for key in ['cards_','global_','actions_','h_actions_']:
        data[key]=np.stack([np.stack([t[min(i,len(t)-1)]['obs'][key] for i in range(T)]) for t in trajectories])
    data['chain']=np.stack([np.stack([t[min(i,len(t)-1)]['chain'] for i in range(T)]) for t in trajectories])
    data['labels']=np.array([[t[min(i,len(t)-1)]['action'] for i in range(T)] for t in trajectories],np.int32)
    data['weights']=np.array([[t[i]['weight'] if i<len(t) else 0 for i in range(T)] for t in trajectories],np.float32)
    data['lengths']=np.array(lengths,np.int32)
    np.savez_compressed(path,**data)
    (path.with_suffix('.json')).write_text(json.dumps(dict(ids=ids,trajectories=len(ids),max_length=T,
                  supervised_decisions=int((data['weights']>0).sum()),sha256=sha(path)),indent=2)+'\n')
for split in specs:
    trajectories=[];ids=[];proof=[]
    for i,case in enumerate(specs[split]):
        examples,trace=collect_combo(case);trajectories.append(examples);ids.append(case['case_id'])
        proof.append(dict(id=case['case_id'],passed=True,trace=trace))
        if i%20==0:print('COMBO',split,i+1,len(specs[split]),flush=True)
    for case in counter_specs[split]:
        d=run_counter(case,collect=True);assert d['passed']
        trajectories.append(d.pop('examples'));ids.append(case['id']);proof.append(d)
    pack(trajectories,ids,OUT/(split+'.npz'))
    (OUT/(split+'-witnesses.json')).write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print('SPLIT_COMPLETE',split,len(trajectories),flush=True)
assert c.native.get_lua_error_count()==0
manifest.update(status='complete',no_cross_split_initial_state_overlap=True,lua_errors=0,
                files={s:json.loads((OUT/(s+'.json')).read_text()) for s in specs})
(OUT/'split-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('DATASET_COMPLETE',json.dumps({s:len(specs[s])+len(counter_specs[s]) for s in specs}),flush=True)
