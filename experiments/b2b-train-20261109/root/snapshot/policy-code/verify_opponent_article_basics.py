"""Replay public article witnesses in the pinned local engine; no server writes."""
import argparse,collections,gc,hashlib,json,os,random,sqlite3,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
P=Path('/home/hatk8/projects/yugioh-ai')
E=P/'evaluation/opponent-article-basics-20261108'
E.mkdir(parents=True,exist_ok=True)
R=E/'runtime'
run=Path('/home/hatk8/projects/yugioh-ai/experiments/meta-opponents-20261107/run-20261107-224631')
meta=json.loads(Path('/home/hatk8/projects/yugioh-ai/experiments/b2b-train-20261109/root/snapshot/meta-run-manifest.json').read_text())
sys.path[:0]=[str(R/'agent'),str(R/'agent/ygoenv')]
os.chdir(R/'agent/scripts')
import numpy as np
import ygoenv
from ygoai.utils import init_ygopro
from ygoai.rl.utils import EnvPreprocess
from ygoenv.ygopro import ygopro_ygoenv as native
init_ygopro('YGOPro-v1','english',str(run/'decks'),meta['vocabulary'])
db=sqlite3.connect(R/'agent/assets/locale/ja/cards.cdb')
names={code:name for code,name in db.execute('select id,name from texts')}
ID_CODES={i+1:int(line.split()[0]) for i,line in enumerate(Path(meta['vocabulary']).read_text().splitlines())}
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def a(act,code,**extra):return dict(act=act,code=code,**extra)
FLARE=42759961;WHITE=82159583;REHEARSAL=20448151;EXTREME=83232904
METEOR=93172951;DEVILS=91237821;OPENING=57847269;DIABOLO=31533473;PHANTASM=70058649
VEILER=97268402;ASH=14558127;DROLL=94145021;G=23434538
ASSAULT=1122030;MEDIUS=97556336;BEAST=23829452;MAIDEN=74631897;FIELD=74733322
AWAKENING=23599634;GRAPH=60946049;VARNISH=74011784;LITERA=97434754;FINEMELT=34541940;CROSS=50277355
CASES={
 'ClownTroupe':dict(id='clown-CRWQjPaN',source='https://duel-simulator.miyauchidp.dev/replay/CRWQjPaN',
   hand=[FLARE,VEILER,ASH,DROLL,G],plan=[
    a('Activate',FLARE),a('None',REHEARSAL),a('None',VEILER),
    a('Activate',REHEARSAL),a('None',FLARE),a('None',WHITE),a('None',EXTREME),
    a('Activate',EXTREME),a('None',METEOR),a('None',DEVILS),a('Summon',WHITE),a('Activate',0,effect=1),
    a('None',METEOR),a('None',DEVILS),a('Activate',WHITE),a('Activate',METEOR),a('Activate',METEOR,effect=10012),a('None',FLARE),
    a('Activate',DEVILS),a('Activate',DEVILS,effect=10012),a('None',OPENING),
    a('Activate',OPENING),a('Activate',OPENING),a('None',DIABOLO),
    a('Activate',REHEARSAL),a('None',DIABOLO),a('None',FLARE),a('Activate',DIABOLO),a('Activate',DIABOLO,effect=10012),a('None',PHANTASM),
    a('None',0,phase='End'),a('Activate',OPENING)]),
 'Artmage':dict(id='artmage-g2rN57Xb-1',source='https://duel-simulator.miyauchidp.dev/replay/g2rN57Xb',
   hand=[ASSAULT,ASH,G,59438930,73642296],plan=[
    a('Activate',ASSAULT),a('Activate',ASSAULT),a('None',MEDIUS),a('Summon',MEDIUS),a('Activate',MEDIUS),a('None',BEAST),a('Activate',0,effect=1152),
    a('Activate',ASSAULT),a('Activate',BEAST,message='select_idlecmd'),a('None',MAIDEN),
    a('Activate',MAIDEN),a('Activate',BEAST),a('None',AWAKENING),a('None',FIELD),a('Activate',FIELD),a('Activate',FIELD),a('None',GRAPH),
    a('Activate',GRAPH),a('Activate',GRAPH),a('None',VARNISH),a('Activate',VARNISH),a('None',ASSAULT),
    a('Activate',FIELD),a('None',LITERA),a('Activate',LITERA),a('Activate',LITERA),a('None',ASSAULT),
    a('Activate',FIELD),a('None',FINEMELT),a('None',FINEMELT),a('SpSummon',CROSS),a('None',GRAPH),a('None',LITERA),
    a('Activate',FINEMELT),a('Activate',FINEMELT),a('Activate',CROSS),a('None',LITERA),a('Activate',MEDIUS),a('None',59438930),
    a('None',0,phase='End'),a('Activate',MAIDEN),a('None',27184601),a('None',MAIDEN),a('Activate',CROSS)]),
}
def deck_cards(key):
    recipe=next(x for x in meta['recipes'] if x['key']==key)
    return {section:[c['code'] for c in recipe['entries'] if c['section']==section for _ in range(c['count'])] for section in ['main','extra']}
def fixture(case,key,seed):
    deck=deck_cards(key);rows=[];rng=random.Random(seed)
    for player in [0,1]:
        remaining=list(deck['main'])
        for i,code in enumerate(case['hand']):
            remaining.remove(code);rows.append([code,player,player,2,i,8])
        rng.shuffle(remaining)
        rows.extend([code,player,player,1,i,8] for i,code in enumerate(remaining))
        rows.extend([code,player,player,64,i,8] for i,code in enumerate(deck['extra']))
    assert collections.Counter((x[1],x[0]) for x in rows)==collections.Counter((player,code) for player in [0,1] for code in deck['main']+deck['extra'])
    return rows
def passive(s):
    for a in s['actions']:
        if a['act']=='Cancel' or a['finish']:return a['index']
    for phase in ['End','Main2','Battle']:
        for a in s['actions']:
            if a['phase']==phase:return a['index']
    if len(s['actions'])==1:return 0
    raise RuntimeError('Unresolved passive choice')
class Teacher:
    def __init__(self,case):self.steps=list(case['plan']);self.adaptations=[];self.summon_context=None
    def __call__(self,s):
        acts=s['actions'];msg=s['message']
        if self.steps and self.steps[0]==a('Activate',DIABOLO) and any(x['controller']==0 and x['location']==2 and x['code']==PHANTASM for x in s['cards']):
            self.steps[:3]=[a('Set',PHANTASM)]
            self.adaptations.append('Phantasm drawn by Whiteface: decline Diabolo search and set the drawn copy from hand.')
        if self.steps:
            target=self.steps[0]
            match=next((x for x in acts if all((msg if k=='message' else x.get(k))==v for k,v in target.items())),None)
            if match:
                self.steps.pop(0)
                if match['act'] in ['Summon','SpSummon','Activate']:self.summon_context=match['code']
                return match['index']
        if msg in ['select_chain','select_effectyn','select_yesno']:return passive(s)
        if msg in ['select_place','select_disfield']:
            prefer={MAIDEN:1,CROSS:6,FINEMELT:3}.get(self.summon_context)
            return next((x['index'] for x in acts if x['place']==prefer),acts[0]['index'])
        if msg=='select_position':return next((x['index'] for x in acts if x['position']==1),0)
        if len(acts)==1:return 0
        if not self.steps and msg=='select_idlecmd':return passive(s)
        if s['phase']==512 and msg=='select_card':
            for code in [DROLL,59438930,VEILER,ASH,G]:
                choice=next((x for x in acts if x['code']==code),None)
                if choice:return choice['index']
        if s['phase']!=4:return passive(s)
        raise RuntimeError('Unresolved witness choice')
def compact(s):
    d=dict(s)
    d['actions']=[dict(x,name=names.get(x['code'],'')) for x in s['actions']]
    d['cards']=[dict(x,name=names.get(x['code'],'')) for x in s['cards']]
    return d
def check_goal(key,s):
    own=lambda location:[x for x in s['cards'] if x['controller']==0 and x['location']==location]
    monsters={x['code'] for x in own(4) if x['position']&5}
    active={x['code'] for x in own(8) if x['position']&5}
    set_spells={x['code'] for x in own(8) if not x['position']&5}
    if key=='ClownTroupe':
        return dict(monsters={WHITE,FLARE}<=monsters,opening_active=OPENING in active,
                    phantasm_set=PHANTASM in set_spells,hand_limit_respected=len(own(2))<=6)
    return dict(monsters={27184601,LITERA,FINEMELT,BEAST,CROSS}<=monsters,
                field_and_assault_active={FIELD,ASSAULT}<=active,
                medius_redirected=any(x['code']==MEDIUS for x in own(32)))
def check_observation(obs,info,s):
    actions=np.asarray(obs['actions_'])[0]
    for action in s['actions']:
        row=actions[action['index']];cid=int(row[1])*256+int(row[2])
        assert ID_CODES.get(cid,0)==action['code'],('Action identity mismatch',s['message'],action,cid)
    cards=np.asarray(obs['cards_'])[0]
    hidden=(cards[:,4]==1)&np.isin(cards[:,2],[1,2])
    assert np.all(cards[hidden,:2]==0),'Opponent private cards exposed'
    assert int(info['public_chain_length'][0])==len(s['public_chain'])
def execute(key,seed,attempt):
    case=CASES[key];out=E/key/attempt;out.mkdir(parents=True,exist_ok=True)
    native.set_benchmark_fixture(fixture(case,key,seed))
    env=ygoenv.make('YGOPro-v1',env_type='gymnasium',num_envs=1,num_threads=1,seed=seed,deck1=key,deck2=key,
        player=-1,play_mode='self',max_options=128,n_history_actions=32,oppo_info=False,
        async_reset=False,max_steps=2048,max_episode_steps=2048,timeout=10)
    env.num_envs=1;env=EnvPreprocess(env,skip_mask=True)
    obs,info=env.reset();trace=[];teacher=Teacher(case);failure=None;dataset=[];normal_peak=0;pre_opponent=None
    try:
        for step in range(400):
            s=native.benchmark_snapshot()
            if s['turn_player']==1 and s['chain_size']==0 and not teacher.steps:break
            assert int(info['raw_num_options'][0])<=128,'Action set truncated'
            assert len(s['actions'])==int(info['num_options'][0])
            check_observation(obs,info,s)
            if s['turn_player']==0:normal_peak=max(normal_peak,s['normal_summons']);pre_opponent=compact(s)
            idx=teacher(s) if s['to_play']==0 else passive(s)
            if s['to_play']==0:
                length=int(info['public_chain_length'][0]);owner=int(info['public_chain_player'][0])
                chain=np.array([length,0 if not length else 1 if owner==0 else 2,int(info['public_chain_card_id'][0]),int(info['public_chain_effect'][0])],np.int32)
                dataset.append(dict(obs={k:np.asarray(obs[k])[0].copy() for k in ['cards_','global_','actions_','h_actions_']},label=idx,chain=chain))
            trace.append(dict(step=step,snapshot=compact(s),selected=dict(s['actions'][idx],name=names.get(s['actions'][idx]['code'],'')),remaining_plan=teacher.steps[:]))
            obs,_,term,trunc,info=env.step(np.array([idx],np.int32))
            assert not term[0] and not trunc[0]
        else:raise RuntimeError('Decision limit exceeded')
    except Exception as exc:failure=str(exc)
    final=compact(native.benchmark_snapshot())
    goals=check_goal(key,final);passed=failure is None and not teacher.steps and all(goals.values()) and native.get_lua_error_count()==0
    if passed:
        arrays={k:np.stack([x['obs'][k] for x in dataset]) for k in dataset[0]['obs']}
        arrays.update(labels=np.array([x['label'] for x in dataset],np.int32),chain=np.stack([x['chain'] for x in dataset]))
        np.savez_compressed(out/'teacher-observations.npz',**arrays)
    result=dict(case=case,seed=seed,trace=trace,final=final,pre_opponent=pre_opponent,goal=goals,passed=passed,
        normal_summons_peak_own_turn=normal_peak,teacher_observations=len(dataset) if passed else 0,
        remaining_plan=teacher.steps,adaptations=teacher.adaptations,failure=failure,
        native_lua_errors=native.get_lua_error_count(),normal_summon_usage_scored=False,
        engine_path=str(R),runtime=json.loads((E/'runtime.json').read_text()),trained_from_this_line=False,publish_new_replays=False)
    write(out/'result.json',result)
    print(json.dumps(dict(out=str(out),steps=len(trace),passed=passed,goal=goals,failure=failure,remaining=teacher.steps,
        prompt=None if passed else {k:final[k] for k in ['message','phase','turn_player','to_play','actions']},
        adaptations=teacher.adaptations,teacher_observations=len(dataset) if passed else 0),ensure_ascii=False),flush=True)
    env.close();del env;gc.collect()
    assert native.get_env_resource_stats()['live_duels']==0
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deck',choices=list(CASES),default='ClownTroupe')
    p.add_argument('--seed',type=int,default=81108);p.add_argument('--attempt',default='probe-01')
    p.add_argument('--control',choices=['skip-assault-conversion'])
    args=p.parse_args()
    if args.control:
        assert args.deck=='Artmage'
        assert CASES['Artmage']['plan'][7]==a('Activate',ASSAULT)
        CASES['Artmage']['plan'].pop(7)
    result=execute(args.deck,args.seed,args.attempt)
    if args.control:
        assert result['passed'] is False and result['native_lua_errors']==0
        assert any(c['code']==27184601 and c['location']==4 and c['controller']==0 for c in result['final']['cards'])
        print('NEGATIVE_CONTROL_PASSED: Maiden unavailable without converting Medius before fusion',flush=True)
    elif not result['passed']:raise SystemExit(1)
