"""Masked model interface and engine rollouts for all locally verified own-deck openings."""
import collections,gc,json
import numpy as np
import rukina_recovery_witness as v
import verify_spirit_and_ash_lines as spirit
import verify_jj_ash_release as jj
import kewl_teacher as kt
P=v.P;R=v.R;native=v.native;features=v.features;write=v.write
class VisibilityAudit:
    def __init__(self):self.allowed_deck=set();self.checks=0;self.hand_peeks=0;self.deck_peeks=0
    def check(self,obs,info,s):
        assert len(s['actions'])==int(info['num_options'][0]) and int(info['raw_num_options'][0])<=128
        actions=np.asarray(obs['actions_'])[0]
        for a in s['actions']:
            row=actions[a['index']];cid=int(row[1])*256+int(row[2]);assert v.engine.ID_CODES.get(cid,0)==a['code'],('action_id',s['message'],a,cid)
        chain=s['public_chain'];rotary=any(x['player']==0 and x['code']==spirit.ROTARY and (x['description']&15)==1 for x in chain)
        cue=any(x['player']==0 and x['code']==spirit.CUE and (x['description']&15)==1 for x in chain)
        if not chain:self.allowed_deck.clear()
        menus=[a for a in s['actions'] if a['code'] and a['act']=='None' and s['message'] in ['select_card','select_unselect_card','select_sum']]
        self.allowed_deck.update(a['code'] for a in menus if a['spec'].startswith('o'))
        own_menu={a['code'] for a in menus if not a['spec'].startswith('o')}
        cards=np.asarray(obs['cards_'])[0]
        for row in cards:
            cid=int(row[0])*256+int(row[1]);code=v.engine.ID_CODES.get(cid,0);loc=int(row[2]);opponent=int(row[4])==1
            if not cid:assert np.all(row[7:]==0),('masked_stats',loc);continue
            if loc==1:
                if opponent:
                    assert cue and code in self.allowed_deck,('unrevealed_opponent_deck',code,s['message'],chain);self.deck_peeks+=1
                else:assert code in own_menu,('unrevealed_own_deck',code,s['message'])
            if opponent and loc==2:
                assert rotary,('unrevealed_opponent_hand',code,s['message'],chain)
                assert code in {x['code'] for x in s['cards'] if x['controller']==1 and x['location']==2};self.hand_peeks+=1
            if opponent and loc==7:assert False,('opponent_facedown_extra_exposed',code)
        assert int(info['public_chain_length'][0])==len(chain);self.checks+=1
def teacher(case):
    if case['kind']=='elf':return v.Teacher(case,v.Interruption('pass'))
    if case['kind']=='tune':return kt.Teacher(case,v.Interruption('pass'))
    return spirit.Teacher(case)
def board_goal(case,s):
    ms={x['code'] for x in v.own(s,4) if x['position']&5};ss=v.own(s,8);active={x['code'] for x in ss if x['position']&5};hand=collections.Counter(x['code'] for x in v.own(s,2))
    if case['kind']=='elf':return {v.CRYSTAL,v.BARONNE}<=ms and {v.PAR,v.RHAP}<=active,'two_bosses'
    if case['kind']=='jj':
        for ending in case['accepted_endings']:
            if ms==set(ending['board']) and (not ending.get('retained') or hand[ending['retained']]>0):
                return spirit.JUKE in active and any(x['code']==spirit.SYNC and not x['position']&5 for x in ss),ending['id']
        return False,None
    if case['kind']=='spirit':return spirit.board_goal(case,s),case['family']
    good={spirit.TRACK,spirit.REMIX}<=ms and spirit.JUKE in active and any(x['code']==spirit.SYNC and not x['position']&5 for x in ss)
    if case.get('mixed'):good=good and {v.CRYSTAL,v.BARONNE}<=ms and {v.PAR,v.RHAP}<=active
    return bool(good),case['family']
def run(case,policy=None,collect=False):
    if case['kind']=='rukina':
        result,examples=v.run(case,policy=policy,collect=collect);result['scope']='Existing user Rukina line or targeted interruption branch.';return result,examples
    env,obs,info=v.create(case);t=None if policy else teacher(case);audit=VisibilityAudit();trace=[];examples=[];own_end=None;recovered=None;recruit=None;failure=None;normal=0
    ash_used=False;search_started=False;opphand_before=None;negated=False;variant=None;search_resolved=False;opponent_steps=[v.a('SpSummon',v.RUK),v.a('Activate',v.RUK)]
    if policy:policy.reset()
    try:
        for step in range(420):
            s=native.benchmark_snapshot()
            if s['turn_player']==0:normal=max(normal,s['normal_summons'])
            if s['turn_player']==1 and s['chain_size']==0 and own_end is None:
                own_end=v.compact(s)
                if not case.get('recover'):break
                if t:
                    if t.steps:raise RuntimeError('Own turn ended with unfinished plan')
                    t.steps=[v.a('Activate',spirit.REMIX),v.a('None',v.ASH),v.a('None',case['recovery_buddy']),v.a('None',v.ASH)]
            if recruit is None and any(x['code']==v.ASH for x in v.own(s,4)):recruit=v.compact(s)
            if own_end and case.get('recover') and recovered is None and s['chain_size']==0 and any(x['code']==v.ASH for x in v.own(s,2)):
                recovered=v.compact(s)
            if search_started and s['chain_size']==0:
                current=collections.Counter(x['code'] for x in s['cards'] if x['controller']==1 and x['location']==2)
                negated=ash_used and current==opphand_before;search_resolved=True;break
            if s['turn_count']>2:failure='opponent_turn_passed';break
            if s['to_play']==0:
                audit.check(obs,info,s)
                if policy:idx=policy.act(obs,info)
                elif recovered:
                    target=bool(s['public_chain'] and s['public_chain'][-1]['player']==1 and s['public_chain'][-1]['code']==v.RUK)
                    ash=next((a for a in s['actions'] if a['code']==v.ASH and a['act']=='Activate' and a['spec'].startswith('h')),None)
                    idx=ash['index'] if target and ash else v.passive(s)
                else:idx=t(s)
                chosen=s['actions'][idx]
                if search_started and chosen['code']==v.ASH and chosen['act']=='Activate' and chosen['spec'].startswith('h'):ash_used=True
                if collect:examples.append(dict(obs={k:np.asarray(obs[k])[0].copy() for k in ['cards_','global_','actions_','h_actions_']},chain=features(info)[0].copy(),label=idx))
            elif s['turn_player']==1 and s['phase']==4 and case.get('recover'):
                target=opponent_steps[0] if opponent_steps else None
                found=next((a for a in s['actions'] if target and all(a[k]==value for k,value in target.items())),None)
                if found:
                    idx=found['index'];opponent_steps.pop(0)
                    if found['act']=='Activate':search_started=True;opphand_before=collections.Counter(x['code'] for x in s['cards'] if x['controller']==1 and x['location']==2)
                elif s['message']=='select_place':idx=next(a['index'] for a in s['actions'] if a['place']==3)
                elif s['message']=='select_position':idx=0
                else:idx=v.passive(s)
            else:idx=v.passive(s)
            trace.append(dict(step=step,snapshot=v.compact(s),selected=s['actions'][idx]))
            obs,_,term,trunc,info=env.step(np.array([idx],np.int32))
            if term[0] or trunc[0]:failure='terminated_or_truncated';break
        else:failure='decision_limit'
    except Exception as exc:failure=str(exc)
    final=v.compact(native.benchmark_snapshot());board=False
    if own_end:board,variant=board_goal(case,own_end)
    release=None;ash_material=False
    for i,row in enumerate(trace):
        a=row['selected']
        if a['code']==v.ASH and a['act']=='None' and a['spec'].startswith('m'):
            previous=trace[i-1]['selected'] if i else {}
            if previous.get('code')==spirit.JUKE and previous.get('act')=='Activate' and previous.get('spec','').startswith('s'):release=row['step']
            elif row['snapshot']['message'] in ['select_sum','select_unselect_card']:ash_material=True
    recovered_ok=not case.get('recover') or recovered is not None and ash_used and negated
    jj_ok=case['kind']!='jj' or release is not None and not ash_material
    result=dict(case=case,passed=bool(board and recovered_ok and jj_ok and failure is None),board=bool(board),chosen_variant=variant,
        own_end=own_end,final=final,trace=trace,failure=failure,ash_recruited=recruit,ash_recovered=recovered,recovered_ash_activated=ash_used,
        opponent_search_negated=negated,opponent_search_resolved=search_resolved,jj_release_step=release,ash_used_as_synchro_material=ash_material,
        normal_summons_diagnostic=normal,normal_summon_usage_scored=False,lua_errors=native.get_lua_error_count(),
        visibility_audit=dict(checks=audit.checks,legal_hand_peek_rows=audit.hand_peeks,legal_deck_peek_rows=audit.deck_peeks),
        model_received_goals_or_plans=False if policy else None)
    result['passed']=result['passed'] and result['lua_errors']==0
    env.close();del env;gc.collect();assert native.get_env_resource_stats()['live_duels']==0
    return result,examples
