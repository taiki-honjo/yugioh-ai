"""Build disjoint masked trajectories for all verified references; old runs stay frozen."""
import argparse,collections,copy,hashlib,itertools,json,random,shutil,sys
from pathlib import Path
from datetime import datetime
import numpy as np
P=Path('/home/hatk8/projects/yugioh-ai');W=Path(__file__).parent
parser=argparse.ArgumentParser();parser.add_argument('--bar-alternative',action='store_true');args=parser.parse_args()
root_name='all-openings-bar-alternative-20261108' if args.bar_alternative else 'all-openings-training-20261108'
ROOT=P/'experiments'/root_name;ROOT.mkdir(exist_ok=True)
cache_root=P/'experiments/all-openings-training-20261108' if args.bar_alternative else ROOT
cached_run=Path((cache_root/'latest-run.txt').read_text().strip()) if (cache_root/'latest-run.txt').exists() else None
B=ROOT/('run-'+datetime.now().strftime('%Y%m%d-%H%M%S'));B.mkdir();(ROOT/'latest-run.txt').write_text(str(B)+'\n')
C=B/'code';C.mkdir()
for name in ['all_opening_witness.py','all_opening_cases.py','prepare_all_opening_training.py','rukina_recovery_witness.py','verify_opponent_article_basics.py','verify_spirit_and_ash_lines.py','verify_jj_ash_release.py']:
    shutil.copy2(W/name,C/name)
kt=(W/'verify_kewltune_current_recipe.py').read_text().split("out=v.P/'evaluation/kewltune-current-recipe-20261108'")[0]
(C/'kewl_teacher.py').write_text(kt)
common=(W/'opponent_article_bc_common.py').read_text().replace('import article_witness as v','import all_opening_witness as v').replace('experiments/opponent-article-training-20261108','experiments/'+root_name)
(C/'all_openings_common.py').write_text(common)
trainer=(W/'train_opponent_article_bc.py').read_text().replace('from opponent_article_bc_common import *','from all_openings_common import *').replace("choices=['ClownTroupe','Artmage']","choices=['ElfnotesKewlTune']")
trainer=trainer.replace("base=load(spec['source_model']);head=init_head();params=dict(base=base,head=head)","base=load(spec['source_model']);head=load(spec['source_head']);params=dict(base=base,head=head)")
trainer=trainer.replace("assert np.all(np.asarray(HEAD.apply(head,jnp.ones((1,4),jnp.int32),jnp.zeros((1,128,12),jnp.uint8)))==0)","assert sha(spec['source_head'])==spec['source_head_sha256']")
trainer=trainer.replace('head_started_neutral=True',"head_started_neutral=False,source_head=spec['source_head'],source_head_sha256=spec['source_head_sha256']")
trainer=trainer.replace("ce=-jnp.take_along_axis(logp,batch['labels'][...,None],axis=-1)[...,0]","ce=-jax.nn.logsumexp(jnp.where(batch['targets']>0,logp,-1e9),axis=-1)")
trainer=trainer.replace("((logits.argmax(-1)==batch['labels'])*w).sum()/denom","(jnp.take_along_axis(batch['targets'],logits.argmax(-1)[...,None],axis=-1)[...,0]*w).sum()/denom")
trainer=trainer.replace("def batch_of(d,indices):return {k:jnp.swapaxes(jnp.asarray(a[indices]),0,1) for k,a in d.items() if k!='lengths'}",
    "def batch_of(d,indices):\n    length=min(d['labels'].shape[1],int(np.ceil(d['lengths'][indices].max()/32)*32))\n    return {k:jnp.swapaxes(jnp.asarray(a[indices,:length]),0,1) for k,a in d.items() if k!='lengths'}")
trainer=trainer.replace("idx=rng.integers(0,len(data['train']['labels']),size=plan['batch_size'])",
    "first=int(rng.integers(0,len(data['train']['labels'])));bucket=np.ceil(data['train']['lengths']/32).astype(int)\n        same=np.flatnonzero(bucket==bucket[first]);idx=np.r_[first,rng.choice(same,size=plan['batch_size']-1)]")
trainer=trainer.replace('probs.append(probabilities)',"if probabilities.shape[1]<d['labels'].shape[1]:probabilities=np.pad(probabilities,((0,0),(0,d['labels'].shape[1]-probabilities.shape[1]),(0,0)),mode='edge')\n        probs.append(probabilities)")
(C/'train_all_openings.py').write_text(trainer)
sys.path.insert(0,str(C));import all_opening_witness as w;import all_opening_cases as cases
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
prior=Path((P/'experiments/elfnotes-rukina-recovery-20261108/latest-run.txt').read_text().strip())
old=cases.read(prior/'manifest.json');oldhands={tuple(sorted(c['hand'])) for s in ['train','validation','holdout'] for c in old['decks']['ElfnotesKewlTune']['cases'][s]}
source=Path(cases.read(prior/'training/ElfnotesKewlTune/result.json')['selected_checkpoint'])
manifest=dict(status='building',grading_policy='normal-summon-agnostic-v2',normal_summon_usage_scored=False,publish_new_replays=False,
    task_scope='Verified Elfnotes, Kewl Tune and mixed references, plus six existing Rukina/pass/interruption branches. Multiple correct branches are accepted without feeding goals to the model.',
    prior_run=str(prior),cached_witness_run=str(cached_run) if cached_run else None,crystal_wing_first_preferred=True,
    runtime=old['runtime'],vocabulary=old['vocabulary'],vocabulary_sha256=old['vocabulary_sha256'],
    await_main_training_source=args.bar_alternative,baronne_and_crystal_spirit_endings_accepted=True,
    training_plan=dict(updates=400 if args.bar_alternative else 1600,batch_size=2,base_learning_rate=3e-5,head_learning_rate=3e-4,kl_weight=0.02,seed=15109 if args.bar_alternative else 15108,
        checkpoint_selection='Minimum validation acceptable-action cross entropy; final holdout never used.'),decks={})
write(B/'manifest.json',manifest)
bank=cases.bank(include_baronne=args.bar_alternative);limits=collections.Counter(w.v.CARDS['main']);groups={};specs={s:[] for s in ['train','validation','holdout']}
for i,proto in enumerate(bank):
    group_key=tuple(sorted(proto['required']))
    # Pure Rukina-Regina references use hands without Veiler, keeping the prior recovery cue distinct.
    no_veiler=proto['family']=='elf-rukina-regina'
    key=(group_key,no_veiler)
    if key not in groups:
        pool=[w.v.VEILER,w.v.ASH,w.v.DROLL,w.v.G,w.v.OGRE,52038441,70088809]
        if proto.get('recover'):pool.remove(w.v.ASH)
        if no_veiler:pool.remove(w.v.VEILER)
        hands=[]
        for tail in itertools.combinations_with_replacement(pool,5-len(proto['required'])):
            hand=list(proto['required'])+list(tail)
            if tuple(sorted(hand)) in oldhands:continue
            if all(n<=limits[c] for c,n in collections.Counter(hand).items()):hands.append(hand)
        random.Random(151080+i).shuffle(hands);assert len(hands)>=15,(proto['family'],len(hands));groups[key]=hands[:15]
    hands=groups[key]
    for split,indices in [('train',range(8)),('validation',range(8,11)),('holdout',range(11,15))]:
        for index in indices:
            for perm in range(2 if split=='train' else 1):
                hand=list(hands[index]);random.Random(151080+index*7+perm).shuffle(hand)
                c=copy.deepcopy(proto);c.update(hand=hand,seed=151080+index*7+perm,split=split,group_id=index,
                    case_id=proto['family']+f'-g{index:02d}-p{perm}',arm=proto.get('arm','pass'))
                specs[split].append(c)
# Shared starter pairs and alternative endings share a split. Prevent any other accidental cross-split duplication.
sets={s:{tuple(sorted(c['hand'])) for c in cs} for s,cs in specs.items()}
assert all(not sets[a]&sets[b] for a,b in itertools.combinations(sets,2))
assert not sets['holdout']&oldhands
# Only one free evaluation per identical starting condition; it may reach either requested JJ ending.
seen=set();hold=[]
for c in specs['holdout']:
    key=(c['eval_family'],tuple(sorted(c['hand'])),c['seed'])
    if key not in seen:seen.add(key);hold.append(c)
specs['holdout']=hold
record=dict(source_model=str(source/'base.flax_model'),source_model_sha256=sha(source/'base.flax_model'),source_head=str(source/'chain-head.flax_model'),source_head_sha256=sha(source/'chain-head.flax_model'),
    deck=str(w.v.DECK),deck_sha256=sha(w.v.DECK),cases=specs,split_rule='Complete initial hand multisets are globally disjoint. All outcomes/permutations for a hand grouped. New data exclude all previous run hands.',
    teacher_families=len(bank),evaluation_contexts=len({c['eval_family'] for c in hold}))
D=B/'dataset/ElfnotesKewlTune';D.mkdir(parents=True);write(D/'cases.json',specs)
for split,cs in specs.items():
    trajectories=[];proof=[]
    cache_data=None;cache_cases={}
    if cached_run and (cached_run/'dataset/ElfnotesKewlTune/cases.json').exists():
        cache_cases={c['case_id']:(i,c) for i,c in enumerate(json.loads((cached_run/'dataset/ElfnotesKewlTune/cases.json').read_text())[split])}
        cache_data=np.load(cached_run/'dataset/ElfnotesKewlTune'/(split+'.npz'))
        cache_arrays={k:cache_data[k] for k in ['cards_','global_','actions_','h_actions_','chain','labels','lengths']}
    for i,c in enumerate(cs):
        cached=cache_cases.get(c['case_id']);reused=bool(cached and cached[1]==c)
        if reused:
            ix=cached[0];n=int(cache_arrays['lengths'][ix]);r=json.loads((cached_run/'witnesses'/split/(c['case_id']+'.json')).read_text())
            arrays={k:cache_arrays[k][ix,:n].copy() for k in cache_arrays if k!='lengths'}
        else:
            r,ex=w.run(cases.normalize(c),collect=True)
            arrays={k:np.stack([x['obs'][k] for x in ex]) for k in ex[0]['obs']} if ex else {}
            if ex:arrays.update(chain=np.stack([x['chain'] for x in ex]),labels=np.array([x['label'] for x in ex],np.int32))
        path=B/'witnesses'/split/(c['case_id']+'.json');write(path,r)
        if not r['passed']:raise RuntimeError('Teacher failed '+c['case_id']+' '+str(r['failure']))
        trajectories.append(arrays);proof.append(dict(case_id=c['case_id'],family=c['family'],passed=True,decisions=len(arrays['labels']),sha256=sha(path),visibility_audit=r.get('visibility_audit'),reused_frozen_witness=reused))
        if i%16==0:print('WITNESS',split,i+1,len(cs),c['family'],flush=True)
    # At identical public observations, every demonstrated continuation is a valid label.
    union={}
    for t in trajectories:
        for j,label in enumerate(t['labels']):
            key=hashlib.sha256(b''.join(t[k][j].tobytes() for k in ['cards_','global_','actions_','h_actions_','chain'])).digest()
            union.setdefault(key,set()).add(int(label))
    alternate_states=0
    for t in trajectories:
        masks=[]
        for j in range(len(t['labels'])):
            key=hashlib.sha256(b''.join(t[k][j].tobytes() for k in ['cards_','global_','actions_','h_actions_','chain'])).digest();mask=np.zeros(128,np.uint8)
            for label in union[key]:mask[label]=1
            alternate_states+=int(mask.sum()>1);masks.append(mask)
        t['targets']=np.stack(masks)
    lengths=np.array([len(t['labels']) for t in trajectories],np.int32);T=int(lengths.max())
    data={k:np.stack([np.stack([t[k][min(j,len(t[k])-1)] for j in range(T)]) for t in trajectories]) for k in trajectories[0]}
    data.update(lengths=lengths,weights=np.array([[float(j<n) for j in range(T)] for n in lengths],np.float32))
    np.savez_compressed(D/(split+'.npz'),**data);write(D/(split+'-witnesses.json'),proof)
    record[split]=dict(trajectories=len(cs),decisions=int(lengths.sum()),max_length=T,acceptable_alternative_states=alternate_states,sha256=sha(D/(split+'.npz')))
    print('SPLIT_COMPLETE',split,record[split],flush=True)
manifest['decks']['ElfnotesKewlTune']=record;manifest.update(status='complete',lua_errors=w.native.get_lua_error_count(),visibility_checks_passed=True,goals_and_plans_in_model_inputs=False)
assert manifest['lua_errors']==0;write(B/'manifest.json',manifest);print('DATASET_COMPLETE',B,flush=True)
