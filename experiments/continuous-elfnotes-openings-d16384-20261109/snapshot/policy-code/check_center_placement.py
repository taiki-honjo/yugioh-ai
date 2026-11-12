"""Fresh paired placement regression; preserve prior model/evaluation and all server replays."""
import collections,copy,hashlib,json,os,random,shutil,sys
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.sched_setaffinity(0,{0,2});os.nice(10)
P=Path('/home/hatk8/projects/yugioh-ai');B=Path((P/'experiments/all-openings-bar-alternative-20261108/latest-run.txt').read_text().strip())
E=P/'evaluation/center-placement-20261108-v4';E.mkdir(exist_ok=False);C=E/'code'
shutil.copytree(B/'code',C,ignore=shutil.ignore_patterns('__pycache__'))
for n in ['center_placement.py','check_center_placement.py']:shutil.copy2(Path(__file__).parent/n,C/n)
sys.path.insert(0,str(C));from all_openings_common import *
from center_placement import PlacementPolicy,PlacementTeacher
import all_opening_cases as bank
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
m=read(B/'manifest.json');spec=m['decks']['ElfnotesKewlTune'];selected=read(B/'training/ElfnotesKewlTune/result.json')
checkpoint=Path(selected['selected_checkpoint']);base=Policy(checkpoint/'base.flax_model',checkpoint/'chain-head.flax_model');adapted=PlacementPolicy(base,v.v.engine.ID_CODES)
teacher_class=v.v.Teacher;old_teacher=v.teacher
v.v.Teacher=lambda *a,**kw:PlacementTeacher(teacher_class(*a,**kw))
def new_teacher(c):
    if c['kind']=='elf':return v.v.Teacher(c,v.v.Interruption('pass'))
    return PlacementTeacher(old_teacher(c))
v.teacher=new_teacher
teacher_rows=[]
for proto in bank.bank(include_baronne=True):
    c=next(c for c in spec['cases']['train'] if c['family']==proto['family'])
    r,_=v.run(bank.normalize(c));write(E/'teachers'/(c['case_id']+'.json'),r)
    teacher_rows.append(dict(family=c['family'],passed=r['passed'],failure=r['failure']))
    print('TEACHER',c['family'],r['passed'],r['failure'],flush=True)
assert all(r['passed'] for r in teacher_rows)
used={tuple(sorted(c['hand'])) for cs in spec['cases'].values() for c in cs}
prior=read(Path(m['prior_run'])/'manifest.json');used|={tuple(sorted(c['hand'])) for cs in prior['decks']['ElfnotesKewlTune']['cases'].values() for c in cs}
pool=[97268402,14558127,94145021,23434538,59438930,52038441,70088809];stock=collections.Counter(v.v.CARDS['main']);rng=random.Random(811032)
protos={}
for c in spec['cases']['validation']:protos.setdefault(c['eval_family'],c)
fresh=[]
for family,proto in protos.items():
    for j in range(2):
        for attempt in range(10000):
            hand=list(proto['required'])+rng.choices(pool,k=5-len(proto['required']));key=tuple(sorted(hand))
            if key in used or collections.Counter(hand)-stock:continue
            if proto.get('recover') and 14558127 in hand:continue
            if family=='elf-rukina-regina' and 97268402 in hand:continue
            break
        else:raise RuntimeError('No fresh fixture '+family)
        used.add(key);rng.shuffle(hand);c=copy.deepcopy(proto);c.update(hand=hand,case_id=family+f'-center-{j:02d}',group_id=j,seed=8110320+len(fresh),split='fresh-placement-regression');fresh.append(c)
write(E/'cases.json',fresh)
rows=[]
for c in fresh:
    for mode,policy in [('before',base),('after',adapted)]:
        r,_=v.run(bank.normalize(c),policy=policy)
        if mode=='after':r['placement_adapter']=dict(events=copy.deepcopy(adapted.memory.events),uses_benchmark_snapshot=False,weights_retrained=False)
        write(E/'evaluation'/mode/(c['case_id']+'.json'),r)
        rows.append(dict(mode=mode,case=c['case_id'],family=c['eval_family'],passed=r['passed'],failure=r['failure'],
            changed=sum(e['changed'] for e in adapted.memory.events) if mode=='after' else 0))
        print('ROLLOUT',mode,c['case_id'],r['passed'],r['failure'],flush=True)
summary=dict(teacher_passed=sum(x['passed'] for x in teacher_rows),teacher_total=len(teacher_rows),fresh_hands=44,
    before=sum(x['passed'] for x in rows if x['mode']=='before'),after=sum(x['passed'] for x in rows if x['mode']=='after'),
    placement_changes=sum(x['changed'] for x in rows if x['mode']=='after'),cases=rows,lua_errors=v.native.get_lua_error_count(),
    hands_disjoint_from_prior_training_validation_and_holdout=True,weights_retrained=False,base_sha256=sha(checkpoint/'base.flax_model'),head_sha256=sha(checkpoint/'chain-head.flax_model'),
    source_checkpoint=str(checkpoint),normal_summon_usage_scored=False,server_replays_saved=0)
assert summary['lua_errors']==0;write(E/'summary.json',summary);print('COMPLETE',json.dumps(summary),flush=True)
