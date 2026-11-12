"""Observe complete first turns; no success labels are inferred from the learner."""
import argparse,collections,gc,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from chain_policy import *
S=P/'experiments/endboard-20261108'
ASH=14558127
OPP_HAND=[c.RUK,ASH,97268402,94145021,23434538]
def write(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def compact(s):
    return {**{k:v for k,v in s.items() if k not in ['cards','actions']},
        'cards':[x for x in s['cards'] if x['controller']==0 and x['location']!=64]}
def prepare():
    S.mkdir(parents=True,exist_ok=True)
    path=S/'cases.json'
    if path.exists():return json.loads(path.read_text())
    seen=set();data=json.loads((B/'dataset/split-manifest.json').read_text())
    for cases in data['combo_cases'].values():
        seen.update(tuple(sorted(x['hand'])) for x in cases if not x.get('field'))
    cases=[]
    for x in c.fixture_definitions():
        v=dict(x,case_id='canonical-'+x['id'],seed=31700+len(cases),suite='canonical')
        cases.append(v);seen.add(tuple(sorted(v['hand'])))
    rng=random.Random(31800)
    for i in range(16):
        while True:
            hand=rng.sample(c.main,5);key=tuple(sorted(hand))
            if key not in seen:break
        seen.add(key);cases.append(dict(id='random',case_id=f'random-{i:02d}',hand=hand,field=[],protected=[],
            seed=31800+i,suite='random',kind='opening',label='固定乱数で無作為に選んだ初手'))
    write(path,cases)
    write(S/'plan.json',dict(stage1='Inspect complete turn with opponent passing; no strategic quality labels.',
        stage2='Replay same initial states with opponent using Ash once at the first legal selected response.',
        stage3='Train only after target endboards and feasible continuation witnesses are specified and verified.',
        outer_decision_budget=512,native_budget=2048,case_count=len(cases),canonical=8,random=16,
        old_short_goals_are_diagnostic_only=True,opponent_hand=OPP_HAND,
        source_training_sha256=sha(B/'training/result.json'),runtime=json.loads((B/'runtime.json').read_text()),
        cases_sha256=sha(path)))
    return cases
def create(case):
    c.OPP_HAND=OPP_HAND
    rows=c.fixture_cards(case)
    if case['suite']=='random':
        rng=random.Random(case['seed']+200000)
        for player in [0,1]:
            indices=[i for i,row in enumerate(rows) if row[1]==player and row[3]==1]
            codes=[rows[i][0] for i in indices];rng.shuffle(codes)
            for i,code in zip(indices,codes):rows[i][0]=code
    c.native.set_benchmark_fixture(rows)
    env=c.ygoenv.make('YGOPro-v1',env_type='gymnasium',num_envs=1,num_threads=1,seed=case['seed'],deck1=c.NAME,deck2=c.NAME,
        player=-1,play_mode='self',max_options=128,n_history_actions=32,oppo_info=False,async_reset=False,
        max_steps=2048,max_episode_steps=2048,timeout=10)
    env.num_envs=1;env=c.EnvPreprocess(env,skip_mask=True);obs,info=env.reset()
    return env,obs,info
def run(case,policy,arm):
    env,obs,info=create(case);policy.reset();trace=[];repeats=collections.Counter();milestone=None
    first=c.native.benchmark_snapshot();normal_peak=0;ash_used=False;reason='decision_budget';peak=0
    try:
        for step in range(512):
            s=c.native.benchmark_snapshot();collect_snapshot_checks(obs,info,s)
            peak=max(peak,int(info['raw_num_options'][0]));assert peak<=128
            if s['turn_player']==0:normal_peak=max(normal_peak,s['normal_summons'])
            if s['turn_player']==1 and s['chain_size']==0:reason='turn_completed';break
            if s['done']:reason='duel_ended';break
            if 'goal' in case and milestone is None and c.goal_check(case,s)['pass_']:
                milestone=dict(step=step,snapshot=compact(s))
            digest=hashlib.sha256(b''.join(np.asarray(obs[k]).tobytes() for k in ['cards_','global_','actions_','h_actions_'])).hexdigest()
            repeats[digest]+=1
            if s['to_play']==0:a=policy.act(obs,info)
            else:
                a=c.passive(s)
                if arm=='ash_once' and not ash_used and s['turn_player']==0 and s['public_chain'] and s['public_chain'][-1]['player']==0:
                    choices=[x for x in s['actions'] if x['act']=='Activate' and x['code']==ASH]
                    if choices:a=choices[0]['index'];ash_used=True
            chosen=s['actions'][a]
            trace.append(dict(step=step,player=s['to_play'],message=s['message'],phase=s['phase'],selected=chosen,
                public_chain=s['public_chain'],legal_actions=s['actions'],observation_sha256=digest,
                own_before=compact(s)))
            obs,_,term,trunc,info=env.step(np.array([a],np.int32))
            if term[0] or trunc[0]:reason='native_terminated_or_truncated';break
        final=c.native.benchmark_snapshot()
        ended=reason=='turn_completed'
        if final['turn_player']==0:normal_peak=max(normal_peak,final['normal_summons'])
        self_ash=[r['step'] for r in trace if r['player']==0 and r['selected']['code']==ASH and r['selected']['act']=='Activate'
            and r['public_chain'] and r['public_chain'][-1]['player']==0]
        assert c.native.get_lua_error_count()==0
        return dict(case=case,arm=arm,turn_completed=ended,reason=reason,steps=len(trace),
            milestone=milestone,initial=compact(first),final=compact(final),normal_summons_used=normal_peak,
            ash_injected=ash_used,self_ash_against_own_chain=self_ash,peak_raw_options=peak,
            max_repeated_observation=max(repeats.values(),default=0),trace=trace,
            quality_label=None,quality_label_note='Observation only; the learner output is not a gold endboard.')
    finally:
        env.close();del env;gc.collect();assert c.native.get_env_resource_stats()['live_duels']==0
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['pass','ash_once'],required=True)
    ap.add_argument('--policy',choices=['baseline','current'],required=True);args=ap.parse_args()
    cases=prepare();out=S/args.arm/args.policy;out.mkdir(parents=True,exist_ok=False)
    training=json.loads((B/'training/result.json').read_text());selected=Path(training['selected_checkpoint'])
    policy=Policy() if args.policy=='baseline' else Policy(selected/'base.flax_model',selected/'chain-head.flax_model')
    results=[];start=time.perf_counter()
    for case in cases:
        result=run(case,policy,args.arm);write(out/(case['case_id']+'.json'),result);results.append(result)
        write(out/'progress.json',dict(completed=len(results),total=len(cases),last=case['case_id'],reason=result['reason']))
        print('CASE',args.arm,args.policy,case['case_id'],result['reason'],result['steps'],flush=True)
    write(out/'summary.json',dict(status='complete',arm=args.arm,policy=args.policy,cases=len(results),
        turn_completed=sum(x['turn_completed'] for x in results),ash_injected=sum(x['ash_injected'] for x in results),
        self_ash_actions=sum(len(x['self_ash_against_own_chain']) for x in results),
        cases_with_milestone=sum(x['milestone'] is not None for x in results),elapsed_seconds=time.perf_counter()-start,
        lua_errors=c.native.get_lua_error_count(),resources=c.native.get_env_resource_stats(),
        base_sha256=sha(original_path() if args.policy=='baseline' else selected/'base.flax_model'),
        head_sha256=sha(selected/'chain-head.flax_model') if args.policy=='current' else None,
        case_manifest_sha256=sha(S/'cases.json'),no_quality_labels_assigned=True))
    print('COMPLETE',args.arm,args.policy,flush=True)
if __name__=='__main__':main()
