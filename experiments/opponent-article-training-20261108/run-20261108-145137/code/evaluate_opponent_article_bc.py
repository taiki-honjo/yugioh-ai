"""Free engine rollouts on holdout hands; no witness actions supplied to models."""
import argparse,gc,sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from opponent_article_bc_common import *
p=argparse.ArgumentParser();p.add_argument('--deck',required=True);p.add_argument('--policy',choices=['before','after'],required=True);args=p.parse_args()
manifest=json.loads((B/'manifest.json').read_text());deck=manifest['decks'][args.deck]
if args.policy=='before':policy=Policy(deck['source_model'])
else:
    train=json.loads((B/'training'/args.deck/'result.json').read_text());selected=Path(train['selected_checkpoint'])
    policy=Policy(selected/'base.flax_model',selected/'chain-head.flax_model')
OUT=B/'evaluation'/args.deck/args.policy;OUT.mkdir(parents=True,exist_ok=False)
rows=[]
for case in deck['cases']['holdout']:
    v.native.set_benchmark_fixture(v.fixture(case,args.deck,case['seed']))
    env=v.ygoenv.make('YGOPro-v1',env_type='gymnasium',num_envs=1,num_threads=1,seed=case['seed'],deck1=args.deck,deck2=args.deck,
        player=-1,play_mode='self',max_options=128,n_history_actions=32,oppo_info=False,async_reset=False,
        max_steps=2048,max_episode_steps=2048,timeout=10)
    env.num_envs=1;env=v.EnvPreprocess(env,skip_mask=True);obs,info=env.reset();policy.reset();trace=[];reason='decision_limit';normal=0
    try:
        for step in range(320):
            s=v.native.benchmark_snapshot();goal=v.check_goal(args.deck,s)
            if s['turn_player']==0:normal=max(normal,s['normal_summons'])
            if s['turn_player']==1 and s['chain_size']==0 and all(goal.values()):reason='goal_reached';break
            if s['turn_player']==1 and s['phase']>4 and s['chain_size']==0:reason='opponent_main_phase_passed';break
            if s['turn_count']>2:reason='two_turns_passed';break
            assert int(info['raw_num_options'][0])<=128
            v.check_observation(obs,info,s)
            idx=policy.act(obs,info) if s['to_play']==0 else v.passive(s)
            trace.append(dict(step=step,snapshot=v.compact(s),selected=s['actions'][idx]))
            obs,_,term,trunc,info=env.step(np.array([idx],np.int32))
            if term[0] or trunc[0]:reason='episode_finished';break
        final=v.compact(v.native.benchmark_snapshot());goal=v.check_goal(args.deck,final)
        row=dict(case_id=case['case_id'],hand=case['hand'],seed=case['seed'],passed=all(goal.values()) and final['turn_player']==1,
            goal=goal,reason=reason,steps=len(trace),normal_summons_diagnostic=normal,normal_summon_usage_scored=False,
            final=final,trace=trace,lua_errors=v.native.get_lua_error_count(),model_received_goals_or_plans=False)
        assert row['lua_errors']==0;write(OUT/(case['case_id']+'.json'),row);rows.append(row)
        print('ROLLOUT',args.deck,args.policy,case['case_id'],row['passed'],reason,len(trace),flush=True)
    finally:
        env.close();del env;gc.collect();assert v.native.get_env_resource_stats()['live_duels']==0
summary=dict(deck=args.deck,policy=args.policy,passed=sum(x['passed'] for x in rows),total=len(rows),
    cases=[{k:x[k] for k in ['case_id','passed','goal','reason','steps']} for x in rows],lua_errors=v.native.get_lua_error_count(),
    scope='Four unseen initial hand multisets for the single trained opening; not all possible hands, disruption lines or win rate.')
write(OUT/'summary.json',summary);print('EVALUATION_COMPLETE',json.dumps(summary),flush=True)
