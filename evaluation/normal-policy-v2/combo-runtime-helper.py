"""Fixed-fixture milestone tests. Goals are not fed into the learned policies."""
import argparse,collections,ctypes,gc,hashlib,json,os,random,sqlite3,sys,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
P=Path('/home/hatk8/projects/yugioh-ai');B=P/'experiments/chain-imitation-20261108';R=B/'runtime'
sys.path[:0]=[str(R/'agent'),str(R/'agent/ygoenv')];os.chdir(R/'agent/scripts')
import numpy as np
import ygoenv
from ygoai.utils import init_ygopro
from ygoai.rl.utils import EnvPreprocess
from ygoenv.ygopro import ygopro_ygoenv as native
DECK=P/'assets/deck/ElfnotesKewlTune.ydk'
VOCAB=P/'experiments/elfnotes-20261106/runtime/agent/scripts/code_list.txt'
NAME=init_ygopro('YGOPro-v1','english',str(DECK),str(VOCAB))
REG=56651978;RUK=13597785;DINA=59581480;FORT=85976588;PAR=64491754;RHAP=24092792
CLIP=43904702;MIX=16509007;TRACK=42781164;SYNC=78058681
FILL=[14558127,97268402,94145021,23434538]
OPP_HAND=FILL+[52038441]
data=json.loads((P/'assets/deck/elfnotes-kewltune-decklist.json').read_text())
NAMES={c['code']:c['db_name'] for c in data['cards']}
ELF={c['code'] for c in data['cards'] if c['section']=='monster' and ('耀聖' in c['db_name'] or c['code']==12375297)}
KEWL={c['code'] for c in data['cards'] if 'キラーチューン' in c['db_name'] or c['code']==14442329}
main=[];extra=[];section='main'
for line in DECK.read_text().splitlines():
    if line=='#extra':section='extra'
    elif line=='!side':section='side'
    elif line.isdigit():(main if section=='main' else extra).append(int(line))
def action(act,code):return dict(act=act,code=code)
def choose(code,after_location=2):return dict(act='None',code=code,after_location=after_location)
def fixture_definitions():
    return [
      dict(id='rukina_search',label='ルキナを中央に出し、エルフェンノーツをサーチ',kind='opening',hand=[RUK]+FILL,
           protected=FILL,goal='rukina_search',plan=[action('SpSummon',RUK),action('Activate',RUK),choose(REG)]),
      dict(id='dina_place',label='ディーナからパラレリズムを表側設置',kind='opening',hand=[DINA]+FILL,
           protected=FILL,goal='dina_place',plan=[action('SpSummon',DINA),action('Activate',DINA),choose(PAR,8)]),
      dict(id='fortuna_place',label='フォルトナからラプソディアを表側設置',kind='opening',hand=[FORT]+FILL,
           protected=FILL,goal='fortuna_place',plan=[action('SpSummon',FORT),action('Activate',FORT),choose(RHAP,8)]),
      dict(id='regina_recruit',label='レギナを中央に出し、別のエルフェンノーツをリクルート',kind='opening',hand=[REG,PAR]+FILL[:3],
           protected=FILL[:3],goal='regina_recruit',plan=[action('Activate',REG),choose(PAR,16),choose(DINA,4)]),
      dict(id='dina_chain',label='ディーナ初動からレギナ＋ルキナ＋永続魔法＋サーチ',kind='opening_combo',hand=[DINA]+FILL,
           protected=FILL,goal='dina_chain',plan=[action('SpSummon',DINA),action('Activate',DINA),choose(PAR,8),
                     action('Activate',PAR),choose(DINA,16),choose(REG,4),choose(RUK,4),action('Activate',RUK),choose(FORT)]),
      dict(id='rukina_chain',label='ルキナ初動からレギナ＋ディーナ＋永続魔法',kind='opening_combo',hand=[RUK]+FILL,
           protected=FILL,goal='rukina_chain',plan=[action('SpSummon',RUK),action('Activate',RUK),choose(REG),
                     action('Activate',REG),choose(RUK,16),choose(DINA,4),action('Activate',DINA),choose(PAR,8)]),
      dict(id='clip_hand_synchro',label='場のクリップ＋手札ミクスでS召喚・サーチ・EX除外',kind='synthetic_midturn',hand=[MIX]+FILL,
           field=[dict(code=CLIP,location=4,sequence=0,position=1)],protected=FILL,goal='hand_synchro',
           plan=[action('SpSummon',TRACK),choose(SYNC)]),
      dict(id='mix_hand_synchro',label='場のミクス＋手札クリップでS召喚・サーチ・EX除外',kind='synthetic_midturn',hand=[CLIP]+FILL,
           field=[dict(code=MIX,location=4,sequence=4,position=1)],protected=FILL,goal='hand_synchro',
           plan=[action('SpSummon',TRACK),choose(SYNC)]),
    ]
def fixture_cards(case):
    rows=[]
    for player in [0,1]:
        remaining=list(main)
        if player==0:
            for i,code in enumerate(case['hand']):remaining.remove(code);rows.append([code,0,0,2,i,8])
            for c in case.get('field',[]):
                remaining.remove(c['code']);rows.append([c['code'],0,0,c['location'],c['sequence'],c['position']])
        else:
            for i,code in enumerate(OPP_HAND):remaining.remove(code);rows.append([code,1,1,2,i,8])
        for i,code in enumerate(remaining):rows.append([code,player,player,1,i,8])
        for i,code in enumerate(extra):rows.append([code,player,player,64,i,8])
    assert collections.Counter((r[1],r[0]) for r in rows)==collections.Counter((p,c) for p in [0,1] for c in main+extra)
    return rows
def own(snap,location):return [c for c in snap['cards'] if c['controller']==0 and c['location']==location]
def goal_check(case,s):
    hand=collections.Counter(c['code'] for c in own(s,2));field=own(s,4);spells=own(s,8)
    has=lambda code:any(c['code']==code and c['position']&5 for c in field)
    center=lambda code:any(c['code']==code and c['sequence']==2 and c['position']&5 for c in field)
    active=lambda code:any(c['code']==code and c['position']&5 for c in spells)
    protected=all(hand[c]>=n for c,n in collections.Counter(case['protected']).items())
    stable=s['message'] in ['select_idlecmd','select_chain'] and s['chain_size']==0 and s['turn_player']==0 and s['phase']==4
    resources=protected
    g=case['goal'];target=False
    if g=='rukina_search':target=center(RUK) and any(hand[c]>0 for c in ELF if c!=RUK)
    if g=='dina_place':target=center(DINA) and active(PAR)
    if g=='fortuna_place':target=center(FORT) and active(RHAP)
    if g=='regina_recruit':target=center(REG) and any(c['code'] in ELF and c['code']!=REG for c in field)
    if g=='dina_chain':target=center(REG) and has(RUK) and active(PAR) and any(hand[c]>0 for c in ELF)
    if g=='rukina_chain':target=center(REG) and has(DINA) and active(PAR)
    if g=='hand_synchro':
        target=has(TRACK) and any(hand[c]>0 for c in KEWL) and len([c for c in s['cards'] if c['controller']==1 and c['location']==64])==14
    return dict(pass_=bool(stable and target and resources),target=bool(target),stable=bool(stable),
                protected=bool(protected),normal_summons=s['normal_summons'],normal_summons_scored=False)
def passive(s):
    acts=s['actions']
    for a in acts:
        if a['act']=='Cancel' or a['finish']:return a['index']
    if len(acts)==1:return 0
    for a in acts:
        if a['phase']=='End':return a['index']
    return 0
class Witness:
    def __init__(self,case):self.steps=list(case['plan']);self.materials=set()
    def __call__(self,s):
        acts=s['actions'];msg=s['message']
        if s['phase']!=4:return passive(s)
        if msg in ['select_chain','select_effectyn']:
            for a in acts:
                if a['act']=='Activate' and (a['code'] in [TRACK,CLIP] or (a['code']==REG and a['spec'].startswith('m'))):return a['index']
        while self.steps:
            target=self.steps[0]
            for a in acts:
                if all(a[k]==v for k,v in target.items() if k!='after_location'):
                    self.steps.pop(0)
                    if a['act']=='SpSummon':self.materials.clear()
                    return a['index']
            if 'after_location' in target and any(c['code']==target['code'] for c in own(s,target['after_location'])):
                self.steps.pop(0);continue
            break
        if msg=='select_idlecmd':raise RuntimeError('Witness plan stalled: '+json.dumps(dict(pending=self.steps,snapshot=s)))
        if msg in ['select_chain','select_effectyn','select_yesno']:return passive(s)
        if msg in ['select_place','select_disfield']:
            return next((a['index'] for a in acts if a['place']==3),0)
        if msg=='select_position':return next((a['index'] for a in acts if a['position']==1),0)
        if msg in ['select_card','select_unselect_card','select_sum','select_tribute']:
            if self.materials:
                for a in acts:
                    if a['act']=='Cancel' or a['finish']:return a['index']
            for code in [CLIP,MIX]:
                for a in acts:
                    if a['code']==code and code not in self.materials:
                        self.materials.add(code);return a['index']
            for a in acts:
                if a['finish'] or a['act']=='Cancel':return a['index']
        if len(acts)==1:return 0
        raise RuntimeError('Witness has unresolved choice: '+json.dumps(s))
class Learned:
    def __init__(self,sample):
        import jax,jax.numpy as jnp,flax.serialization
        from dataclasses import asdict
        from ygoai.rl.jax.agent import RNNAgent,ModelArgs
        self.jax=jax;self.agent=RNNAgent(**asdict(ModelArgs()),embedding_shape=None)
        key=jax.random.PRNGKey(random.Random(0).randint(0,int(1e8)))
        state=self.agent.init_rnn_state(1)
        self.base=jax.jit(self.agent.init)(key,sample,state)
        @jax.jit
        def infer(params,state,obs):
            out=self.agent.apply(params,obs,state)
            return out[0],jnp.argmax(out[1],axis=-1)
        self.infer=infer;self.cache={}
    def reset(self,path):
        import flax.serialization
        if str(path) not in self.cache:self.cache[str(path)]=self.jax.device_put(flax.serialization.from_bytes(self.base,Path(path).read_bytes()))
        self.params=self.cache[str(path)];self.state=self.jax.device_put(self.agent.init_rnn_state(1))
    def act(self,obs):
        self.state,actions=self.infer(self.params,self.state,obs)
        return int(np.asarray(actions)[0])
def snapshot_compact(s):
    result={k:v for k,v in s.items() if k!='actions'}
    result['cards']=[c for c in result['cards'] if c['location']!=64 or c['controller']==0]
    return result
def create_env(case):
    native.set_benchmark_fixture(fixture_cards(case))
    raw=ygoenv.make('YGOPro-v1',env_type='gymnasium',num_envs=1,num_threads=1,seed=case['seed'],deck1=NAME,deck2=NAME,
                   player=-1,play_mode='self',max_options=128,n_history_actions=32,oppo_info=False,
                   async_reset=False,max_steps=128,max_episode_steps=128,timeout=10)
    raw.num_envs=1;env=EnvPreprocess(raw,skip_mask=True)
    obs,info=env.reset();initial=native.benchmark_snapshot()
    assert collections.Counter(c['code'] for c in own(initial,2))==collections.Counter(case['hand'])
    assert len(own(initial,4))==len(case.get('field',[]))
    assert collections.Counter(c['code'] for c in initial['cards'] if c['controller']==1 and c['location']==2)==collections.Counter(OPP_HAND)
    assert initial['normal_summons']==0 and initial['chain_size']==0
    assert not goal_check(case,initial)['pass_']
    return env,obs,info,[]
def main_run():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',choices=['reference','all'],default='reference')
    parser.add_argument('--variants',type=int,default=3);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    templates=fixture_definitions();cases=[]
    for template in templates:
        for v in range(args.variants):
            c=dict(template);c['hand']=list(c['hand'])
            if v==1:c['hand']=c['hand'][2:]+c['hand'][:2]
            if v==2:c['hand']=list(reversed(c['hand']))
            c['case_id']=c['id']+f'-hand{v}';c['seed']=7100+v;c['review_status']='rule_backed_machine_witness; human_strategy_review_pending'
            cases.append(c)
    (args.out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    previous=json.loads((P/'experiments/elfnotes-20261106/expansion-revised-20261106-205114/additional-training-result.json').read_text())
    policies=[('reference',None)]
    if args.policy=='all':policies += [('model4096',previous['source_checkpoint']),('model69632',previous['trained_checkpoint'])]
    rows=[];learned=None;started=time.perf_counter()
    for label,path in policies:
        for case in cases:
            env=None;trace=[];reason='decision_limit';counts=collections.Counter()
            try:
                env,obs,info,setup=create_env(case)
                start_snapshot=native.benchmark_snapshot()
                if label=='reference':witness=Witness(case)
                else:
                    if learned is None:learned=Learned(obs)
                    learned.reset(path)
                peak=0;max_rep=0
                for step in range(128):
                    s=native.benchmark_snapshot();g=goal_check(case,s)
                    if g['pass_']:reason='goal_reached';break
                    if s['done']:reason='engine_ended';break
                    if s['turn_player']!=0 or s['phase'] not in [1,2,4]:reason='left_main_phase_1';break
                    peak=max(peak,int(info['raw_num_options'][0]))
                    if peak>128:raise RuntimeError('Action set truncated; fixture not accepted')
                    digest=hashlib.sha256(b''.join(np.asarray(obs[k]).tobytes() for k in ['cards_','global_','actions_','h_actions_'])).hexdigest()
                    counts[digest]+=1;max_rep=max(max_rep,counts[digest])
                    if s['to_play']!=0:a=passive(s)
                    elif label=='reference':a=witness(s)
                    else:a=learned.act(obs)
                    assert 0<=a<len(s['actions'])
                    trace.append(dict(step=step,player=s['to_play'],message=s['message'],selected=s['actions'][a],
                                      available_actions=s['actions'],snapshot=snapshot_compact(s),obs_sha256=digest))
                    obs,_,terminated,truncated,info=env.step(np.array([a],dtype=np.int32))
                    if terminated[0] or truncated[0]:reason='engine_ended_or_limit';break
                final=native.benchmark_snapshot();g=goal_check(case,final)
                if g['pass_']:reason='goal_reached'
                assert native.get_lua_error_count()==0
                row=dict(case_id=case['case_id'],family=case['id'],policy=label,passed=g['pass_'],reason=reason,
                         steps=len(trace),goal=g,peak_raw_options=peak,max_repeated_observation=max_rep,
                         loop_suspected=max_rep>=12,setup=setup,start=start_snapshot,final=snapshot_compact(final),trace=trace,
                         model_path=str(path) if path else None,model_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None)
                rows.append(row)
                (args.out/(label+'-'+case['case_id']+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
                (args.out/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(cases)*len(policies),
                    last_case=case['case_id'],policy=label,passed=row['passed'],reason=reason),indent=2)+'\n')
                print('CASE',label,case['case_id'],row['passed'],reason,len(trace),flush=True)
                if label=='reference' and not row['passed']:raise RuntimeError('Reference witness failed; do not score models')
            except Exception:
                (args.out/'failure-context.json').write_text(json.dumps(dict(case=case,policy=label,trace=trace,
                    snapshot=native.benchmark_snapshot() if env is not None else None),ensure_ascii=False,indent=2)+'\n')
                raise
            finally:
                if env is not None:env.close();del env
                gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
                assert native.get_env_resource_stats()['live_duels']==0
    summaries={label:dict(passed=sum(row['passed'] for row in rows if row['policy']==label),
                         cases=sum(row['policy']==label for row in rows)) for label,_ in policies}
    (args.out/'summary.json').write_text(json.dumps(dict(status='complete',results=summaries,
        elapsed_seconds=time.perf_counter()-started,case_count=len(cases),human_strategy_review='pending',
        native_lua_errors=native.get_lua_error_count(),runtime=json.loads((B/'runtime.json').read_text())),indent=2)+'\n')
    print('COMPLETE',json.dumps(summaries),flush=True)
if __name__=='__main__':main_run()
