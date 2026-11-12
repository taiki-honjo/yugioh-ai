"""Native regression of the common position loader, no server access or training."""
import collections,copy,gc,json,os,random,sys
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
os.sched_setaffinity(0,{0,2});os.nice(10)
P=Path('/home/hatk8/projects/yugioh-ai');E=P/'evaluation/all-deck-position-20261109'
sys.path.insert(0,str(P/'scripts'));from deck_policy import create_policy,create_teacher
cfg=json.loads((E/'registry.json').read_text());sys.path.insert(0,cfg['model_code_directory'])
E=E/'verified'
from opponent_article_bc_common import v,np,B
def read(p):return json.loads(Path(p).read_text())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def env_for(key,case,rows=None):
    v.native.set_benchmark_fixture(rows if rows is not None else v.fixture(case,key,case['seed']))
    env=v.ygoenv.make('YGOPro-v1',env_type='gymnasium',num_envs=1,num_threads=1,seed=case['seed'],deck1=key,deck2=key,
        player=-1,play_mode='self',max_options=128,n_history_actions=32,oppo_info=False,async_reset=False,
        max_steps=2048,max_episode_steps=2048,timeout=10)
    env.num_envs=1;env=v.EnvPreprocess(env,skip_mask=True);return env,*env.reset()
def close(env):
    env.close()
class DiverseTeacher(v.Teacher):
    def __call__(self,s):
        if len(self.steps)>=2 and self.steps[:2]==[v.a('None',v.DIABOLO),v.a('None',v.FLARE)] and s['message']=='select_card':
            choices=[a for a in s['actions'] if a['code']]
            if choices and all(a['code']==v.FLARE for a in choices):self.ritual_chosen_early=True;return choices[0]['index']
        if getattr(self,'ritual_chosen_early',False) and self.steps and self.steps[0]==v.a('None',v.FLARE):
            if any(c['controller']==0 and c['location']==4 and c['code']==v.FLARE for c in s['cards']):self.steps.pop(0);self.ritual_chosen_early=False
        return super().__call__(s)
manifest=read(B/'manifest.json');regression=[]
for key in ['Artmage','ClownTroupe']:
    cases=manifest['decks'][key]['cases']['holdout']
    policies={'before':create_policy(key,positions=False),'after':create_policy(key)}
    for case in cases:
        for mode,policy in policies.items():
            env,obs,info=env_for(key,case);policy.reset();trace=[];reason='decision_limit'
            try:
                for step in range(320):
                    s=v.native.benchmark_snapshot();goal=v.check_goal(key,s)
                    if s['turn_player']==1 and s['chain_size']==0 and all(goal.values()):reason='goal_reached';break
                    if s['turn_player']==1 and s['phase']>4 and s['chain_size']==0:reason='opponent_main_phase_passed';break
                    if s['turn_count']>2:reason='two_turns_passed';break
                    assert int(info['raw_num_options'][0])<=128
                    v.check_observation(obs,info,s)
                    idx=policy.act(obs,info) if s['to_play']==0 else v.passive(s)
                    assert idx in [a['index'] for a in s['actions']]
                    trace.append(dict(step=step,snapshot=v.compact(s),selected=s['actions'][idx]))
                    obs,_,term,trunc,info=env.step(np.array([idx],np.int32))
                    if term[0] or trunc[0]:reason='episode_finished';break
                final=v.compact(v.native.benchmark_snapshot());goal=v.check_goal(key,final)
                row=dict(deck=key,mode=mode,case_id=case['case_id'],passed=all(goal.values()) and final['turn_player']==1,
                    goal=goal,reason=reason,final=final,trace=trace,events=copy.deepcopy(policy.positions.events) if mode=='after' else [])
                write(E/'regression'/key/mode/(case['case_id']+'.json'),row)
                regression.append({k:row[k] for k in ['deck','mode','case_id','passed','reason']})
                print('REGRESSION',key,mode,case['case_id'],row['passed'],flush=True)
            finally:
                close(env);del env;gc.collect();assert v.native.get_env_resource_stats()['live_duels']==0
    # Correct the stored witness interface without requiring Elfnotes memory.
    case=cases[0];env,obs,info=env_for(key,case);teacher=create_teacher(DiverseTeacher(case));trace=[]
    try:
        for step in range(400):
            s=v.native.benchmark_snapshot()
            if s['turn_player']==1 and s['chain_size']==0 and not teacher.steps:break
            idx=teacher(s) if s['to_play']==0 else v.passive(s)
            trace.append(dict(snapshot=v.compact(s),selected=s['actions'][idx]));obs,_,_,_,info=env.step(np.array([idx],np.int32))
        final=v.compact(v.native.benchmark_snapshot());passed=not teacher.steps and all(v.check_goal(key,final).values())
        write(E/'teachers'/(key+'.json'),dict(passed=passed,trace=trace,final=final,events=teacher.positions.events));assert passed
        print('TEACHER',key,passed,flush=True)
    finally:
        close(env);del env;gc.collect();assert v.native.get_env_resource_stats()['live_duels']==0
    del policies;gc.collect()

controlled=[]
for key,target in [('BrandedDracotail',1498449),('SkyStriker',37351133)]:
    cards=v.deck_cards(key);required=[14558127,94145021] if key=='BrandedDracotail' else [26077387,37351133]
    hand=list(required)
    for code in cards['main']:
        if len(hand)==5:break
        if hand.count(code)<cards['main'].count(code):hand.append(code)
    case=dict(hand=hand,seed=1190901);rows=v.fixture(case,key,case['seed'])
    if key=='BrandedDracotail':
        for code,location,seq in [(89851827,4,0),(1498449,16,0),(70871153,16,1)]:
            row=next(x for x in rows if x[1]==0 and x[0]==code and x[3] in [1,64]);row[3:]=[location,seq,1]
        for player in [0,1]:
            for i,x in enumerate(x for x in rows if x[1]==player and x[3]==1):x[4]=i
    env,obs,info=env_for(key,case,rows);policy=create_policy(key);policy.reset();trace=[];arrival=None;link_arrival=False;failure=None
    try:
        for step in range(100):
            s=v.native.benchmark_snapshot();acts=s['actions'];msg=s['message']
            found=next((c for c in s['cards'] if c['controller']==0 and c['location']==4 and c['code']==target),None)
            if found:arrival=copy.deepcopy(found)
            link=next((c for c in s['cards'] if c['controller']==0 and c['location']==4 and c['code']==63288573),None)
            if link:link_arrival=copy.deepcopy(link)
            if arrival and (key=='BrandedDracotail' or link_arrival):break
            if s['to_play']!=0:idx=v.passive(s)
            elif msg=='select_position':idx=policy.act(obs,info)
            elif msg in ['select_place','select_disfield']:idx=next((a['index'] for a in acts if a['place'] in [1,2,4,5,6,7]),0)
            elif key=='SkyStriker':
                if msg=='select_idlecmd':idx=next(a['index'] for a in acts if (not arrival and a['act']=='Summon' and a['code']==26077387) or (arrival and a['act']=='SpSummon' and a['code']==63288573))
                elif msg in ['select_chain','select_effectyn']:idx=next((a['index'] for a in acts if a['code']==37351133 and a['act']=='Activate'),None);idx=v.passive(s) if idx is None else idx
                elif msg in ['select_card','select_unselect_card','select_sum']:
                    idx=next((a['index'] for a in acts if a['code']==26077387 and a['spec'].startswith('m')),None)
                    idx=v.passive(s) if idx is None else idx
                else:idx=v.passive(s)
            else:
                if msg=='select_idlecmd':idx=next(a['index'] for a in acts if a['act']=='Activate' and a['code']==89851827)
                elif msg in ['select_card','select_unselect_card']:
                    # Prefer Faimena when selecting the monster to revive.
                    idx=next((a['index'] for a in acts if a['code']==1498449),None)
                    if idx is None:idx=next(a['index'] for a in acts if a['code']==70871153)
                else:idx=v.passive(s)
            trace.append(dict(snapshot=v.compact(s),selected=acts[idx]));obs,_,_,_,info=env.step(np.array([idx],np.int32))
        assert arrival and arrival['position']==4,(key,arrival,policy.positions.events)
        if key=='SkyStriker':assert link_arrival and link_arrival['position']==1
    except Exception as exc:failure=str(exc)
    finally:
        row=dict(deck=key,passed=failure is None,fixture='Controlled public field/grave board; recipe counts preserved.' if key=='BrandedDracotail' else 'Raye normal summon, Roze special summon, then Kagari link summon.',
            target=target,arrival=arrival,link_arrival=link_arrival,events=policy.positions.events,trace=trace,failure=failure)
        write(E/'controlled'/(key+'.json'),row);controlled.append({k:row[k] for k in ['deck','passed','failure','arrival','link_arrival']});close(env)
        del env;gc.collect();assert v.native.get_env_resource_stats()['live_duels']==0
    print('CONTROLLED',key,row['passed'],failure,flush=True)

summary=dict(regression=regression,controlled=controlled,weights_retrained=False,normal_summon_usage_scored=False,
    server_replays_saved=0,lua_errors=v.native.get_lua_error_count(),resources=v.native.get_env_resource_stats(),
    scope='Regression reuses the two previously tested four-hand opening sets; the other two decks use controlled legal summon tests, not opening/win-rate certification.')
write(E/'summary.json',summary);print('COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='regression'}),flush=True)
assert summary['lua_errors']==0 and all(r['passed'] for r in controlled)
