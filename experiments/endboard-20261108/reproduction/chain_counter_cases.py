import sys,json,gc,itertools,random
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from chain_imitation_common import *
ASH=14558127;STONE=72323266;RECO=89392810
def definition(kind,variant=0):
    source=1 if 'regina' in kind and 'offturn' in kind else 0
    role_self=kind.startswith('self')
    ai=source if role_self else 1-source
    card=c.RUK if 'rukina' in kind else c.REG if 'regina' in kind else c.MIX if 'mix' in kind else c.TRACK
    def fill(prefix,player):
        limits=collections.Counter(c.main);limits.subtract(prefix)
        pool=[STONE,c.PAR,c.RHAP,94145021]
        choices=[v for v in itertools.combinations_with_replacement(pool,5-len(prefix))
                 if all(n<=limits[k] for k,n in collections.Counter(v).items())]
        seed=variant*113+player*17+sum(map(ord,kind));rng=random.Random(seed)
        hand=list(prefix)+list(rng.choice(choices));rng.shuffle(hand);return hand
    hands={0:fill([],0),1:fill([],1)}
    hands[ai]=fill([ASH],ai)
    prefix=[card if card!=c.TRACK else c.MIX]+([c.PAR] if card==c.REG else [])+([ASH] if ai==source else [])
    hands[source]=fill(prefix,source)
    fields=[]
    if card==c.TRACK:fields=[dict(code=c.CLIP,controller=source,location=4,sequence=(variant%5),position=1)]
    for player in [0,1]:
        shift=variant%5;hands[player]=hands[player][shift:]+hands[player][:shift]
    if card==c.RUK:plan=[c.action('SpSummon',c.RUK),c.action('Activate',c.RUK),c.choose(c.REG)]
    elif card==c.REG:plan=[c.action('Activate',c.REG),c.choose(c.PAR,16),c.choose(c.DINA,4)]
    elif card==c.MIX:plan=[c.action('Summon',c.MIX),c.choose(RECO)]
    else:plan=[c.action('SpSummon',c.TRACK),c.choose(c.SYNC)]
    return dict(id=kind+f'-v{variant}',kind=kind,source=source,ai=ai,card=card,hands=hands,field=fields,
                seed=21000+variant,plan=plan,expected='pass' if role_self else 'ash',synthetic=True)
def make_env(case):
    entries=[]
    for player in [0,1]:
        remain=list(c.main)
        for i,code in enumerate(case['hands'][player]):remain.remove(code);entries.append([code,player,player,2,i,8])
        for f in case['field']:
            if f['controller']==player:
                remain.remove(f['code']);entries.append([f['code'],player,player,f['location'],f['sequence'],f['position']])
        entries += [[code,player,player,1,i,8] for i,code in enumerate(remain)]
        entries += [[code,player,player,64,i,8] for i,code in enumerate(c.extra)]
    c.native.set_benchmark_fixture(entries)
    env=c.ygoenv.make('YGOPro-v1',env_type='gymnasium',num_envs=1,num_threads=1,seed=case['seed'],deck1=c.NAME,deck2=c.NAME,
                     player=-1,play_mode='self',max_options=128,n_history_actions=32,oppo_info=False,
                     async_reset=False,max_steps=128,max_episode_steps=128,timeout=10)
    env.num_envs=1;env=c.EnvPreprocess(env,skip_mask=True);obs,info=env.reset()
    return env,obs,info
def run_counter(case,policy=None,collect=False):
    # JSON object keys are strings; preserve the same cards after loading a split.
    case=dict(case,hands={int(k):v for k,v in case['hands'].items()})
    env,obs,info=make_env(case);teacher=c.Witness(dict(plan=case['plan']))
    examples=[];trace=[];critical=None;chosen_good=False;before_hand=None;before_field=None
    if policy:policy.reset()
    try:
        for step in range(128):
            s=c.native.benchmark_snapshot();collect_snapshot_checks(obs,info,s)
            acts=s['actions'];turn=s['to_play'];chain=s['public_chain']
            if critical is not None and not chain:
                hand=collections.Counter(x['code'] for x in s['cards'] if x['controller']==case['source'] and x['location']==2)
                field=sum(x['controller']==case['source'] and x['location']==4 for x in s['cards'])
                resolved=(field>before_field) if case['card']==c.REG else bool(hand-before_hand)
                success=chosen_good and resolved==(case['expected']=='pass')
                return dict(id=case['id'],kind=case['kind'],expected=case['expected'],passed=bool(success),
                            action_correct=bool(chosen_good),search_or_recruit_resolved=bool(resolved),
                            critical=critical,trace=trace,examples=examples)
            ash=[a for a in acts if a['act']=='Activate' and a['code']==ASH]
            cancel=[a for a in acts if a['act']=='Cancel']
            is_critical=(critical is None and turn==case['ai'] and s['message']=='select_chain' and ash and cancel
                         and len(chain)==1 and chain[-1]['player']==case['source'] and chain[-1]['code']==case['card'])
            if turn==case['source']:
                view=dict(s);view['cards']=[dict(x,controller=0 if x['controller']==case['source'] else 1) for x in s['cards']]
                if s['message'] in ['select_chain','select_effectyn'] and case['card'] in [c.MIX,c.TRACK] and any(a['code']==case['card'] and a['act']=='Activate' for a in acts):
                    a=next(a['index'] for a in acts if a['code']==case['card'] and a['act']=='Activate')
                elif s['message'] in ['select_chain','select_effectyn'] and any(a['code']==c.CLIP and a['act']=='Activate' for a in acts):
                    a=c.passive(s)
                elif s['message'] in ['select_chain','select_effectyn'] and any(a['code']==c.MIX and a['act']=='Activate' for a in acts):
                    a=next(a['index'] for a in acts if a['code']==c.MIX and a['act']=='Activate')
                elif s['phase']==4 and s['message']=='select_idlecmd' and not teacher.steps:
                    a=c.passive(s)
                else:a=teacher(view)
            else:a=c.passive(s)
            if is_critical:
                expected_action=cancel[0]['index'] if case['expected']=='pass' else ash[0]['index']
                if policy:a=policy.act(obs,info)
                else:a=expected_action
                chosen_good=a==expected_action
                critical=dict(step=step,public_chain=chain,features=chain_features(info)[0].tolist(),
                              actions=acts,selected=acts[a],expected_index=expected_action)
                before_hand=collections.Counter(x['code'] for x in s['cards'] if x['controller']==case['source'] and x['location']==2)
                before_field=sum(x['controller']==case['source'] and x['location']==4 for x in s['cards'])
            elif turn==case['ai'] and policy:
                # Prefix decisions are fixed to construct the response question; only update recurrent state.
                policy.observe(obs,info)
            if turn==case['ai'] and collect:
                examples.append(dict(obs={k:np.asarray(obs[k])[0].copy() for k in ['cards_','global_','actions_','h_actions_']},
                                     chain=chain_features(info)[0].copy(),action=int(a),weight=4.0 if is_critical else 0.0))
            trace.append(dict(step=step,player=turn,message=s['message'],selected=acts[a],public_chain=chain,critical=bool(is_critical)))
            obs,_,term,trunc,info=env.step(np.array([a],dtype=np.int32))
            if term[0] or trunc[0]:break
        raise RuntimeError('Counter fixture did not finish '+case['id']+' '+json.dumps(trace[-8:]))
    finally:
        env.close();del env;gc.collect();assert c.native.get_env_resource_stats()['live_duels']==0
if __name__=='__main__':
    B.joinpath('counter-probe').mkdir(exist_ok=True)
    kinds=['self_rukina','opp_rukina','self_regina','opp_regina_offturn','self_regina_offturn',
           'self_mix','opp_mix','self_track','opp_track']
    for kind in kinds:
        d=run_counter(definition(kind));d.pop('examples')
        (B/'counter-probe'/(kind+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
        assert d['passed'];print(kind,d['passed'],d['critical']['features'],flush=True)
