"""Verify interruption branches and build globally disjoint hand-composition splits."""
import collections,copy,hashlib,itertools,json,random,shutil,sys
from pathlib import Path
from datetime import datetime
import numpy as np
P=Path('/home/hatk8/projects/yugioh-ai');W=Path(__file__).parent
ROOT=P/'experiments/elfnotes-rukina-recovery-20261108';ROOT.mkdir(exist_ok=True)
B=ROOT/('run-'+datetime.now().strftime('%Y%m%d-%H%M%S'));B.mkdir();(ROOT/'latest-run.txt').write_text(str(B)+'\n')
C=B/'code';C.mkdir()
for name in ['rukina_recovery_witness.py','verify_opponent_article_basics.py','prepare_rukina_recovery_training.py']:
    shutil.copy2(W/name,C/name)
# Reuse the already exercised architecture, optimizer and masked model interface.
common=(W/'opponent_article_bc_common.py').read_text().replace('import article_witness as v','import rukina_recovery_witness as v').replace('experiments/opponent-article-training-20261108','experiments/elfnotes-rukina-recovery-20261108')
(C/'elfnotes_recovery_common.py').write_text(common)
trainer=(W/'train_opponent_article_bc.py').read_text().replace('from opponent_article_bc_common import *','from elfnotes_recovery_common import *').replace("choices=['ClownTroupe','Artmage']","choices=['ElfnotesKewlTune']")
trainer=trainer.replace('base=load(spec[\'source_model\']);head=init_head();params=dict(base=base,head=head)',"base=load(spec['source_model']);head=load(spec['source_head']);params=dict(base=base,head=head)")
trainer=trainer.replace("assert np.all(np.asarray(HEAD.apply(head,jnp.ones((1,4),jnp.int32),jnp.zeros((1,128,12),jnp.uint8)))==0)","assert sha(spec['source_head'])==spec['source_head_sha256']")
trainer=trainer.replace('head_started_neutral=True','head_started_neutral=False,source_head=spec[\'source_head\'],source_head_sha256=spec[\'source_head_sha256\']')
(C/'train_elfnotes_recovery.py').write_text(trainer)
sys.path.insert(0,str(C));import rukina_recovery_witness as v
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
source=Path(json.loads((P/'experiments/endboard-20261108/training/result.json').read_text())['selected_checkpoint'])
meta=v.engine.meta
manifest=dict(status='building',task_scope='Train the own Elfnotes Kewl Tune AI on the user Rukina line and five interruption branches; no full battle training.',
    grading_policy='normal-summon-agnostic-v2',normal_summon_usage_scored=False,publish_new_replays=False,
    source_url='https://duel-simulator.miyauchidp.dev/replay/AXEBgcKV',
    runtime=json.loads((v.engine.E/'runtime.json').read_text()),vocabulary=meta['vocabulary'],vocabulary_sha256=sha(meta['vocabulary']),
    training_plan=dict(updates=1000,batch_size=2,base_learning_rate=3e-5,head_learning_rate=3e-4,kl_weight=0.02,seed=12108,
        checkpoint_selection='minimum validation teacher-forcing CE; final holdout never used'),decks={})
write(B/'manifest.json',manifest)
def groups(prefix,n,pool,seed):
    limits=collections.Counter(v.CARDS['main']);out=[]
    for tail in itertools.combinations_with_replacement(pool,n):
        hand=prefix+list(tail)
        if all(count<=limits[code] for code,count in collections.Counter(hand).items()):out.append(hand)
    random.Random(seed).shuffle(out);assert len(out)>=12
    return out[:12]
pool=[v.VEILER,v.ASH,v.DROLL,v.G,v.OGRE]
main_groups=groups([v.RUK,v.VEILER],3,pool,12108)
review_groups={
 'dina_review':groups([v.DINA],4,pool,12109),
 'regina_review':groups([v.REG,v.PAR],3,pool,12110)}
review_plans={
 'dina_review':[v.a('SpSummon',v.DINA),v.a('Activate',v.DINA),v.choose(v.PAR,8),v.a('Activate',v.PAR),v.choose(v.DINA,16),v.choose(v.REG,4),v.choose(v.RUK,4),v.a('Activate',v.RUK),v.choose(v.FORT)],
 'regina_review':[v.a('Activate',v.REG),v.choose(v.PAR,16),v.choose(v.DINA,4),v.a('Activate',v.DINA),v.choose(v.PAR,8),v.a('Activate',v.PAR),v.choose(v.DINA,16),v.choose(v.RUK,4),v.a('Activate',v.RUK),v.choose(v.FORT)]}
specs={s:[] for s in ['train','validation','holdout']}
for split,indices in [('train',range(6)),('validation',range(6,8)),('holdout',range(8,12))]:
    for family in v.ARMS+list(review_groups):
        for i in indices:
            for perm in range(2 if split=='train' else 1):
                hand=list(main_groups[i] if family in v.ARMS else review_groups[family][i]);random.Random(121080+i*7+perm).shuffle(hand)
                case=dict(case_id=f'{family}-g{i:02d}-p{perm}',arm=family if family in v.ARMS else 'pass',family=family,
                    hand=hand,seed=121080+i*7+perm,split=split,group_id=i)
                if family in review_plans:case['plan_override']=review_plans[family]
                specs[split].append(case)
sets={s:{tuple(sorted(c['hand'])) for c in cases} for s,cases in specs.items()}
assert all(not sets[a]&sets[b] for a,b in itertools.combinations(sets,2))
D=B/'dataset/ElfnotesKewlTune';D.mkdir(parents=True);write(D/'cases.json',specs)
record=dict(source_model=str(source/'base.flax_model'),source_model_sha256=sha(source/'base.flax_model'),
    source_head=str(source/'chain-head.flax_model'),source_head_sha256=sha(source/'chain-head.flax_model'),
    deck=str(v.DECK),deck_sha256=sha(v.DECK),cases=specs,split_rule='Complete initial hand multisets disjoint globally; all arms and permutations grouped.')
for split,cases in specs.items():
    trajectories=[];proof=[]
    for i,case in enumerate(cases):
        result,examples=v.run(case,collect=True)
        path=B/'witnesses'/split/(case['case_id']+'.json');write(path,result)
        if not result['passed']:raise RuntimeError('Teacher failed '+case['case_id']+' '+str(result['failure']))
        arrays={k:np.stack([x['obs'][k] for x in examples]) for k in examples[0]['obs']}
        arrays.update(chain=np.stack([x['chain'] for x in examples]),labels=np.array([x['label'] for x in examples],np.int32))
        trajectories.append(arrays)
        proof.append(dict(case_id=case['case_id'],family=case['family'],passed=True,interruption_injected=result['interruption_injected'],
            interruption_countered=result['interruption_countered'],scope=result['scope'],trace_path=str(path),trace_sha256=sha(path)))
        if i%12==0:print('WITNESS',split,i+1,len(cases),case['family'],flush=True)
    lengths=np.array([len(t['labels']) for t in trajectories],np.int32);T=int(lengths.max())
    data={k:np.stack([np.stack([t[k][min(i,len(t[k])-1)] for i in range(T)]) for t in trajectories]) for k in trajectories[0]}
    data.update(lengths=lengths,weights=np.array([[float(i<n) for i in range(T)] for n in lengths],np.float32))
    np.savez_compressed(D/(split+'.npz'),**data);write(D/(split+'-witnesses.json'),proof)
    record[split]=dict(trajectories=len(cases),decisions=int(lengths.sum()),max_length=T,sha256=sha(D/(split+'.npz')))
    print('SPLIT_COMPLETE',split,record[split],flush=True)
manifest['decks']['ElfnotesKewlTune']=record
manifest.update(status='complete',lua_errors=v.native.get_lua_error_count(),private_information_checks_passed=True)
assert manifest['lua_errors']==0;write(B/'manifest.json',manifest);print('DATASET_COMPLETE',B,flush=True)
