import sys,json,itertools,random,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from endboard_witness import *
D=S/'dataset';D.mkdir(exist_ok=False)
old=json.loads((B/'dataset/split-manifest.json').read_text());observed=json.loads((S/'cases.json').read_text())
seen={tuple(sorted(x['hand'])) for x in observed}
for cases in old['combo_cases'].values():seen.update(tuple(sorted(x['hand'])) for x in cases)
for cases in old['counter_cases'].values():seen.update(tuple(sorted(h)) for x in cases for h in x['hands'].values())
pool=[14558127,97268402,94145021,23434538,59438930,52038441];limits=collections.Counter(c.main)
groups={}
for n in [3,4]:
    used={tuple(sorted(x['protected'])) for vs in old['combo_cases'].values() for x in vs if len(x['protected'])==n}
    candidates=[]
    for x in itertools.combinations_with_replacement(pool,n):
        if any(v>limits[k] for k,v in collections.Counter(x).items()) or tuple(sorted(x)) in used:continue
        families=['regina'] if n==3 else ['rukina','dina']
        if any(tuple(sorted(definition(f,x,'pass',0,'')['hand'])) in seen for f in families):continue
        candidates.append(list(x))
    random.Random(34000+n).shuffle(candidates);assert len(candidates)>=21;groups[n]=candidates
splits={'train':range(0,12),'validation':range(12,15),'holdout':range(15,21)}
specs={s:[] for s in splits}
for split,indices in splits.items():
    for family in ['rukina','dina','regina']:
        for i in indices:
            fill=groups[3 if family=='regina' else 4][i]
            for arm in ['pass','ash_once']:
                for perm in range(2 if split=='train' else 1):
                    case=definition(family,fill,arm,34000+i*5+perm,f'{family}-{arm}-g{i:02d}-p{perm}')
                    random.Random(case['seed']).shuffle(case['hand']);case['group_id']=f'fill{len(fill)}-'+','.join(map(str,sorted(fill)))
                    specs[split].append(case)
sets={s:{tuple(sorted(x['hand'])) for x in cs} for s,cs in specs.items()}
assert all(not sets[a]&sets[b] for a,b in itertools.combinations(sets,2))
assert all(not sets[s]&seen for s in sets)
def pack(trajectories):
    T=max(map(len,trajectories));assert T<=64
    data={k:np.stack([np.stack([t[min(i,len(t)-1)]['obs'][k] for i in range(T)]) for t in trajectories]) for k in ['cards_','global_','actions_','h_actions_']}
    data['chain']=np.stack([np.stack([t[min(i,len(t)-1)]['chain'] for i in range(T)]) for t in trajectories])
    data['labels']=np.array([[t[min(i,len(t)-1)]['action'] for i in range(T)] for t in trajectories],np.int32)
    data['weights']=np.array([[t[i]['weight'] if i<len(t) else 0 for i in range(T)] for t in trajectories],np.float32)
    data['lengths']=np.array(list(map(len,trajectories)),np.int32);return data
def merge(a,b):
    T=max(a['labels'].shape[1],b['labels'].shape[1])
    def pad(x,key):
        if key=='lengths':return x
        pad=T-x.shape[1]
        if not pad:return x
        more=np.repeat(x[:,-1:],pad,axis=1)
        if key=='weights':more*=0
        return np.concatenate([x,more],axis=1)
    return {k:np.concatenate([pad(a[k],k),pad(b[k],k)],axis=0) for k in a}
manifest=dict(status='precommitted',new_cases=specs,split_rule='All permutations and both interruption arms grouped by filler multiset; new initial hands disjoint from every previous combo split and inspection case.',
    goals='No interruption: central Elfnotes plus off-center Rukina or Dina, active Parallels, protected hand and normal summon preserved, end turn. Ash: explicit observed-negation residual-resource baseline; not recovery-combo success.',
    expert_optimality_claimed=False,reference_bounce_probes=[str(x) for x in (S/'reference-probes').glob('*-pass.json')],
    training_plan=dict(updates=600,batch_size=4,base_learning_rate=3e-5,head_learning_rate=3e-4,kl_weight=.01,seed=34008,
        checkpoint_selection='minimum combined validation teacher-forcing CE; no holdout or battle selection'),
    source_training_sha256=sha(B/'training/result.json'),old_training_replay=True,old_holdout_replayed=False)
write(D/'split-manifest.json',manifest)
for split,cases in specs.items():
    trajectories=[];proof=[]
    for case in cases:
        d=run_reference(case,collect=True);trajectories.append(d.pop('examples'));proof.append(d)
    data=pack(trajectories);new_n=len(trajectories)
    if split!='holdout':data=merge(data,{k:v for k,v in np.load(B/'dataset'/(split+'.npz')).items()})
    np.savez_compressed(D/(split+'.npz'),**data)
    write(D/(split+'-witnesses.json'),proof)
    write(D/(split+'.json'),dict(new_trajectories=new_n,replayed_trajectories=len(data['labels'])-new_n,total_trajectories=len(data['labels']),
        max_length=data['labels'].shape[1],supervised_decisions=int((data['weights']>0).sum()),sha256=sha(D/(split+'.npz'))))
    print('SPLIT',split,new_n,len(data['labels']),flush=True)
manifest.update(status='complete',lua_errors=c.native.get_lua_error_count(),all_reference_goals_passed=True,
    files={s:json.loads((D/(s+'.json')).read_text()) for s in splits})
write(D/'split-manifest.json',manifest)
print('DATASET_COMPLETE',flush=True)
