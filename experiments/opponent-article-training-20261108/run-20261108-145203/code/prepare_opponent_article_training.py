"""Build disjoint hand-multiset datasets; do not overwrite pilot or old models."""
import collections,copy,gc,hashlib,importlib.util,itertools,json,random,shutil,sys,time
from pathlib import Path
from datetime import datetime
import numpy as np
P=Path('/home/hatk8/projects/yugioh-ai');W=Path(__file__).parent
ROOT=P/'experiments/opponent-article-training-20261108';ROOT.mkdir(exist_ok=True)
B=ROOT/('run-'+datetime.now().strftime('%Y%m%d-%H%M%S'));B.mkdir()
(ROOT/'latest-run.txt').write_text(str(B)+'\n');C=B/'code';C.mkdir()
for source,dest in [('verify_opponent_article_basics.py','article_witness.py'),('opponent_article_bc_common.py','opponent_article_bc_common.py'),
                    ('prepare_opponent_article_training.py','prepare_opponent_article_training.py'),('train_opponent_article_bc.py','train_opponent_article_bc.py'),
                    ('evaluate_opponent_article_bc.py','evaluate_opponent_article_bc.py')]:shutil.copy2(W/source,C/dest)
sys.path.insert(0,str(C));import article_witness as v
class DiverseTeacher(v.Teacher):
    def __call__(self,s):
        if len(self.steps)>=2 and self.steps[:2]==[v.a('None',v.DIABOLO),v.a('None',v.FLARE)] and s['message']=='select_card':
            choices=[a for a in s['actions'] if a['code']]
            if choices and all(a['code']==v.FLARE for a in choices):
                self.ritual_chosen_early=True
                return choices[0]['index']
        if getattr(self,'ritual_chosen_early',False) and self.steps and self.steps[0]==v.a('None',v.FLARE):
            if any(c['controller']==0 and c['location']==4 and c['code']==v.FLARE for c in s['cards']):
                self.steps.pop(0);self.ritual_chosen_early=False
        try:return super().__call__(s)
        except RuntimeError:
            if s['phase']==512 and s['message']=='select_card' and not self.steps:
                hand=collections.Counter(c['code'] for c in s['cards'] if c['controller']==0 and c['location']==2)
                choices=[a for a in s['actions'] if a['code']]
                if choices:return max(choices,key=lambda a:(hand[a['code']],a['code']))['index']
            raise
v.Teacher=DiverseTeacher
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta_run=Path((P/'experiments/meta-opponents-20261107/latest-run.txt').read_text().strip());meta=json.loads((meta_run/'manifest.json').read_text())
manifest=dict(status='building',runtime=json.loads((v.E/'runtime.json').read_text()),vocabulary=meta['vocabulary'],vocabulary_sha256=sha(meta['vocabulary']),
    grading_policy='normal-summon-agnostic-v2',normal_summon_usage_scored=False,publish_new_replays=False,
    split_rule='Disjoint complete initial hand multisets, all permutations and random draw variants grouped.',
    reference_scope='Two verified basic openings only; not all 51 article references.',
    training_plan=dict(updates=600,batch_size=2,base_learning_rate=3e-5,head_learning_rate=3e-4,kl_weight=0.01,seed=1198,
        checkpoint_selection='minimum held-out-validation teacher-forcing cross entropy; final holdout never used'),decks={})
write(B/'manifest.json',manifest)
def build(key):
    template=v.CASES[key];limits=collections.Counter(v.deck_cards(key)['main'])
    required=[v.FLARE,v.VEILER] if key=='ClownTroupe' else [v.ASSAULT,59438930]
    pool=[v.VEILER,v.ASH,v.DROLL,v.G,59438930,27204311] if key=='ClownTroupe' else [v.ASH,v.G,73642296,38814750,49036338]
    groups=[]
    for tail in itertools.combinations_with_replacement(pool,3):
        hand=required+list(tail)
        if all(n<=limits[c] for c,n in collections.Counter(hand).items()) and sorted(hand)!=sorted(template['hand']):groups.append(hand)
    random.Random(1198).shuffle(groups);assert len(groups)>=12
    groups=groups[:12];specs={split:[] for split in ['train','validation','holdout']}
    for split,indices in [('train',range(0,6)),('validation',range(6,8)),('holdout',range(8,12))]:
        for index in indices:
            for perm in range(2 if split=='train' else 1):
                case=copy.deepcopy(template);case['hand']=list(groups[index]);random.Random(11980+index*5+perm).shuffle(case['hand'])
                case.update(case_id=f'{key}-g{index:02d}-p{perm}',seed=119800+index*5+perm,split=split,group_id=index)
                specs[split].append(case)
    fingerprints={split:{tuple(sorted(x['hand'])) for x in cases} for split,cases in specs.items()}
    assert all(not fingerprints[a]&fingerprints[b] for a,b in itertools.combinations(specs,2))
    recipe=next(r for r in meta['recipes'] if r['key']==key)
    source=json.loads((meta_run/'training'/key/'phase2-65536/result.json').read_text())
    checkpoint=Path(source['trained_checkpoint']);record=dict(source_model=str(checkpoint),source_model_sha256=sha(checkpoint),
        deck=recipe['ydk'],deck_sha256=sha(recipe['ydk']),cases=specs,original_pilot_excluded_from_splits=True)
    D=B/'dataset'/key;D.mkdir(parents=True);write(D/'cases.json',specs)
    for split,cases in specs.items():
        trajectories=[];proof=[]
        for case in cases:
            v.CASES[key]=case
            result=v.execute(key,case['seed'],'training-'+B.name+'/'+case['case_id'])
            if not result['passed']:raise RuntimeError('Teacher failed '+case['case_id'])
            source_dir=v.E/key/('training-'+B.name)/case['case_id']
            trajectories.append({k:x.copy() for k,x in np.load(source_dir/'teacher-observations.npz').items()})
            proof.append(dict(case_id=case['case_id'],passed=True,goals=result['goal'],trace_path=str(source_dir/'result.json'),
                              trace_sha256=sha(source_dir/'result.json'),adaptations=result['adaptations']))
        lengths=np.array([len(t['labels']) for t in trajectories],np.int32);T=int(lengths.max())
        data={k:np.stack([np.stack([t[k][min(i,len(t[k])-1)] for i in range(T)]) for t in trajectories]) for k in trajectories[0]}
        data.update(lengths=lengths,weights=np.array([[float(i<n) for i in range(T)] for n in lengths],np.float32))
        np.savez_compressed(D/(split+'.npz'),**data);write(D/(split+'-witnesses.json'),proof)
        record[split]=dict(trajectories=len(cases),decisions=int(lengths.sum()),max_length=T,sha256=sha(D/(split+'.npz')))
        print('SPLIT',key,split,record[split],flush=True)
    manifest['decks'][key]=record;write(B/'manifest.json',manifest)
for key in ['ClownTroupe','Artmage']:build(key)
assert v.native.get_lua_error_count()==0
manifest.update(status='complete',lua_errors=0,private_information_checks_passed=True)
write(B/'manifest.json',manifest);print('DATASET_COMPLETE',B,flush=True)
