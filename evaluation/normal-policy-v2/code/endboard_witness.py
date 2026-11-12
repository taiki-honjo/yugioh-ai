"""Conservative, explicit endboard baselines, not expert-optimal combo labels."""
import sys,json,gc,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from inspect_full_turn import *

def definition(family,fill,arm,seed,case_id):
    if family=='rukina':
        hand=[c.RUK]+list(fill)
        plan=[c.action('SpSummon',c.RUK),c.action('Activate',c.RUK),c.choose(c.REG),
            c.action('Activate',c.REG),c.choose(c.RUK,16),c.choose(c.DINA,4),c.action('Activate',c.DINA),c.choose(c.PAR,8),
            c.action('Activate',c.PAR),c.choose(c.DINA,16),c.choose(c.RUK,4)]
    elif family=='dina':
        hand=[c.DINA]+list(fill);plan=c.fixture_definitions()[4]['plan']
    else:
        hand=[c.REG,c.PAR]+list(fill)
        plan=[c.action('Activate',c.REG),c.choose(c.PAR,16),c.choose(c.DINA,4),c.action('Activate',c.DINA),c.choose(c.PAR,8),
            c.action('Activate',c.PAR),c.choose(c.DINA,16),c.choose(c.RUK,4),c.action('Activate',c.RUK),c.choose(c.FORT)]
    return dict(id=family,case_id=case_id,hand=hand,field=[],protected=list(fill),plan=plan,
        seed=seed,suite='endboard',arm=arm,synthetic=False)

def end_goal(case,s,normal_used,ash_used):
    cards=[x for x in s['cards'] if x['controller']==0]
    hand=collections.Counter(x['code'] for x in cards if x['location']==2)
    field=[x for x in cards if x['location']==4 and x['position']&5]
    spells=[x for x in cards if x['location']==8 and x['position']&5]
    protected=all(hand[k]>=v for k,v in collections.Counter(case['protected']).items())
    center=any(x['sequence']==2 and x['code'] in c.ELF for x in field)
    usable_side=any(x['sequence']!=2 and x['code'] in [c.RUK,c.DINA] for x in field)
    engine_pair=center and usable_side and any(x['code']==c.PAR for x in spells)
    if case['arm']=='pass':target=engine_pair
    else:
        # These are residual-resource baselines after an observed negation, not recovery-combo success.
        target=ash_used and (any(x['code']==c.RUK for x in field) if case['id']=='rukina'
            else any(x['code']==c.REG for x in field) if case['id']=='regina' else len(field)==0)
    completed=s['turn_player']==1 and s['chain_size']==0 and not s['done']
    return dict(passed=bool(completed and target and protected),turn_completed=bool(completed),
        target=bool(target),protected=bool(protected),normal_summons=normal_used,normal_summons_scored=False,
        engine_pair=bool(engine_pair),scope='minimum endboard' if case['arm']=='pass' else 'post-negation residual-resource baseline')

def probe_rukina(env,obs,info):
    """Script an opponent level-6 summon and verify the reference can actually return it."""
    summoned=False;used=False;trace=[]
    for step in range(128):
        s=c.native.benchmark_snapshot();collect_snapshot_checks(obs,info,s)
        opp=[x for x in s['cards'] if x['controller']==1 and x['location']==4 and x['code']==c.RUK]
        if summoned and used and not opp and not s['public_chain']:
            return dict(passed=True,trace=trace)
        if s['turn_player']!=1 or s['done']:break
        a=c.passive(s)
        if s['to_play']==1:
            choices=[x for x in s['actions'] if x['act']=='SpSummon' and x['code']==c.RUK]
            if choices and not summoned:a=choices[0]['index'];summoned=True
            elif s['message']=='select_place':a=next((x['index'] for x in s['actions'] if x['place']==3),a)
        elif opp:
            choices=[x for x in s['actions'] if x['act']=='Activate' and x['code']==c.RUK]
            if choices:a=choices[0]['index'];used=True
            elif used and s['message'] in ['select_card','select_unselect_card']:
                choices=[x for x in s['actions'] if x['code']==c.RUK and x['spec'].startswith('o')]
                if choices:a=choices[0]['index']
        trace.append(dict(step=step,player=s['to_play'],message=s['message'],selected=s['actions'][a],actions=s['actions']))
        obs,_,term,trunc,info=env.step(np.array([a],np.int32))
        if term[0] or trunc[0]:break
    return dict(passed=False,trace=trace,final=compact(c.native.benchmark_snapshot()))

def run_reference(case,collect=False,probe=False):
    env,obs,info=create(case);teacher=c.Witness(case);ash_used=False;ash_resolved=False;trace=[];examples=[];normal=0
    try:
        for step in range(256):
            s=c.native.benchmark_snapshot();collect_snapshot_checks(obs,info,s)
            if s['turn_player']==0:normal=max(normal,s['normal_summons'])
            if s['turn_player']==1 and s['chain_size']==0:break
            if ash_used and not s['public_chain']:ash_resolved=True;teacher.steps=[]
            if s['to_play']==0:
                if ash_resolved or (s['message']=='select_idlecmd' and not teacher.steps):a=c.passive(s)
                else:a=teacher(s)
                if collect:
                    examples.append(dict(obs={k:np.asarray(obs[k])[0].copy() for k in ['cards_','global_','actions_','h_actions_']},
                        chain=chain_features(info)[0].copy(),action=int(a),weight=1.0))
            else:
                a=c.passive(s)
                if case['arm']=='ash_once' and not ash_used and s['public_chain'] and s['public_chain'][-1]['player']==0:
                    choices=[x for x in s['actions'] if x['act']=='Activate' and x['code']==ASH]
                    if choices:a=choices[0]['index'];ash_used=True
            trace.append(dict(step=step,player=s['to_play'],message=s['message'],selected=s['actions'][a],public_chain=s['public_chain']))
            obs,_,term,trunc,info=env.step(np.array([a],np.int32));assert not term[0] and not trunc[0]
        final=c.native.benchmark_snapshot();goal=end_goal(case,final,normal,ash_used)
        assert goal['passed'],(case,goal,trace)
        result=dict(case=case,goal=goal,normal_used=normal,ash_injected=ash_used,final=compact(final),trace=trace,examples=examples)
        if probe and case['arm']=='pass':result['response_probe']=probe_rukina(env,obs,info);assert result['response_probe']['passed'],result
        assert c.native.get_lua_error_count()==0
        return result
    finally:env.close();del env;gc.collect();assert c.native.get_env_resource_stats()['live_duels']==0
if __name__=='__main__':
    out=S/'reference-probes';out.mkdir(exist_ok=False)
    for family in ['rukina','dina','regina']:
        for arm in ['pass','ash_once']:
            case=definition(family,c.FILL if family!='regina' else c.FILL[:3],arm,33100,family+'-'+arm)
            d=run_reference(case,probe=True);d.pop('examples');write(out/(case['case_id']+'.json'),d)
            print('REFERENCE',family,arm,d['goal'],flush=True)
