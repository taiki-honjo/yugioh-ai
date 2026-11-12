"""Current 40/15 witnesses for two-card Spirit starts and actual opponent-turn Ash recovery."""
import argparse,collections,gc,json,random
import rukina_recovery_witness as v
a=v.a;c=v.choose
CUE=16387555;RECO=89392810;ROTARY=17209452;MIX=16509007;TRACK=42781164;REMIX=88170262;JUKE=14442329;SYNC=78058681;LOUD=41069676
def elf_mid():
    return [a('SpSummon',v.DINA),a('Activate',v.DINA),c(v.PAR,8),a('Activate',v.PAR),c(v.DINA,16),c(v.REG,4),c(v.FORT,4),
        a('Activate',v.FORT),c(v.RHAP,8),a('Activate',v.RHAP),c(v.FORT,16),c(v.SPIRIT,4),a('Activate',v.SPIRIT),a('SpSummon',v.BARONNE)]
def juke_mix():return [a('Activate',JUKE),a('Activate',JUKE),c(TRACK,16),c(MIX,4),a('Activate',MIX)]
def make_cases():
    filler=[v.VEILER,v.DROLL,v.G]
    return [
      dict(case_id='reco-spirit-ash',hand=[RECO,v.SPIRIT]+filler,recover=True,mixed=True,board=[TRACK,REMIX,v.BARONNE],
        plan=[a('Summon',RECO),a('Activate',RECO),c(ROTARY),a('SpSummon',TRACK),a('Activate',TRACK),c(JUKE),c(v.DINA)]+elf_mid()+juke_mix()+
            [c(CUE),a('Summon',CUE),a('Activate',CUE),c(v.ASH,4),a('SpSummon',TRACK),a('Activate',CUE),a('Activate',ROTARY),c(SYNC),a('SpSummon',REMIX),a('Set',SYNC)],
        materials={TRACK:[[(RECO,'m'),(v.SPIRIT,'h')],[(CUE,'m'),(ROTARY,'h')]],REMIX:[[(MIX,'m'),(v.ASH,'m')]]}),
      dict(case_id='cue-spirit',hand=[CUE,v.SPIRIT]+filler,recover=False,mixed=True,board=[TRACK,REMIX,v.BARONNE],
        plan=[a('Summon',CUE),a('SpSummon',TRACK),a('Activate',CUE),a('Activate',TRACK),c(JUKE),c(v.DINA)]+elf_mid()+juke_mix()+
            [c(CUE),a('Summon',CUE),a('Activate',CUE),c(RECO,4),a('Activate',RECO),c(ROTARY),a('SpSummon',TRACK),a('Activate',ROTARY),c(SYNC),a('SpSummon',REMIX),a('Set',SYNC)],
        materials={TRACK:[[(CUE,'m'),(v.SPIRIT,'h')],[(CUE,'m'),(ROTARY,'h')]],REMIX:[[(MIX,'m'),(RECO,'m')]]}),
      dict(case_id='cue-reco-ash',hand=[CUE,RECO]+filler,recover=True,mixed=False,board=[LOUD,REMIX],
        plan=[a('Summon',RECO),a('Activate',RECO),c(ROTARY),a('SpSummon',TRACK),a('Activate',ROTARY),a('Activate',TRACK),c(JUKE),c(SYNC)]+juke_mix()+
            [c(RECO),a('Summon',CUE),a('Activate',CUE),c(v.ASH,4),a('SpSummon',REMIX),a('SpSummon',LOUD),a('Activate',CUE),a('Set',SYNC)],
        materials={TRACK:[[(RECO,'m'),(ROTARY,'h')]],REMIX:[[(MIX,'m'),(RECO,'h')]],LOUD:[[(CUE,'m'),(v.ASH,'m')]]}),
      dict(case_id='cue-mix-ash',hand=[CUE,MIX]+filler,recover=True,mixed=False,board=[TRACK,REMIX],
        plan=[a('Summon',MIX),a('Activate',MIX),c(ROTARY),a('Activate',ROTARY),c(CUE),a('Summon',CUE),a('Activate',CUE),c(v.ASH,4),
            a('SpSummon',TRACK),a('Activate',CUE),a('Activate',ROTARY),a('Activate',TRACK),c(SYNC),c(JUKE),a('Activate',JUKE),a('SpSummon',REMIX),a('Set',SYNC)],
        materials={TRACK:[[(CUE,'m'),(ROTARY,'h')]],REMIX:[[(MIX,'m'),(v.ASH,'m')]]}),
      dict(case_id='cue-rotary-ash',hand=[CUE,ROTARY]+filler,recover=True,mixed=False,board=[TRACK,REMIX],
        plan=[a('Summon',CUE),a('Activate',CUE),c(v.ASH,4),a('SpSummon',TRACK),a('Activate',CUE),a('Activate',ROTARY),a('Activate',TRACK),c(SYNC),c(JUKE)]+juke_mix()+
            [c(RECO),a('SpSummon',REMIX),a('Summon',RECO),a('Activate',RECO),c(ROTARY),a('SpSummon',TRACK),a('Set',SYNC)],
        materials={TRACK:[[(CUE,'m'),(ROTARY,'h')],[(RECO,'m'),(ROTARY,'h')]],REMIX:[[(MIX,'m'),(v.ASH,'m')]]}),
    ]
class Teacher:
    def __init__(self,case):
        self.case=case;self.steps=list(case['plan']);self.context=None;self.materials=set();self.current=[];self.summon_counts=collections.Counter()
    def __call__(self,s):
        acts=s['actions'];msg=s['message']
        if msg in ['select_chain','select_effectyn']:
            for x in acts:
                if x['act']=='Activate' and ((x['code']==v.SPIRIT and x['spec'].startswith('g')) or (x['code']==v.REG and x['spec'].startswith(('g','m')))):return x['index']
        if msg in ['select_card','select_unselect_card','select_sum','select_tribute'] and self.current:
            if any(x['code']==code and x['spec'].startswith(loc) for code,loc in self.current for x in acts):
                for code,loc in self.current:
                    found=next((x for x in acts if x['code']==code and x['spec'].startswith(loc) and code not in self.materials),None)
                    if found:self.materials.add(code);return found['index']
                finish=next((x for x in acts if x['finish'] or x['act']=='Cancel'),None)
                if finish:return finish['index']
        if msg=='select_card' and acts and all(x['spec'].startswith('o') for x in acts):return 0
        if msg=='select_yesno' and self.steps and self.steps[0]['code'] in [JUKE,SYNC]:return next(x['index'] for x in acts if x['act']!='Cancel')
        if msg=='select_option':return next((x['index'] for x in acts if x['effect']==1152),0)
        while self.steps:
            target=self.steps[0]
            found=next((x for x in acts if all(x.get(k)==value for k,value in target.items() if k!='after_location')),None)
            if found is None and msg=='select_card' and target['act']=='Summon':
                found=next((x for x in acts if x['act']=='None' and x['code']==target['code'] and x['spec'].startswith('h')),None)
            if found:
                self.steps.pop(0)
                if found['act'] in ['SpSummon','Summon']:
                    self.context=found['code'];self.materials.clear();self.current=[]
                    if self.context in self.case['materials']:
                        index=self.summon_counts[self.context];self.summon_counts[self.context]+=1;self.current=self.case['materials'][self.context][index]
                    elif self.context==v.BARONNE:self.current=[(v.SPIRIT,'m'),(v.REG,'m')]
                return found['index']
            if 'after_location' in target and any(x['code']==target['code'] for x in v.own(s,target['after_location'])):
                self.steps.pop(0);continue
            break
        if msg in ['select_chain','select_effectyn','select_yesno']:return v.passive(s)
        if msg in ['select_place','select_disfield']:
            preferred=[6,2,4,5,1,3] if self.context in [TRACK,REMIX,LOUD] else [1,2,4,5,6,3] if self.context==v.BARONNE else [3,2,4,1,5,6]
            for place in preferred:
                found=next((x for x in acts if x['place']==place),None)
                if found:return found['index']
            return 0
        if msg=='select_position':return next((x['index'] for x in acts if x['position']==1),0)
        if not self.steps and msg=='select_idlecmd':return v.passive(s)
        if len(acts)==1:return 0
        raise RuntimeError('Unresolved witness '+json.dumps(dict(message=msg,steps=self.steps,actions=acts)))
def board_goal(case,s):
    monsters={x['code'] for x in v.own(s,4) if x['position']&5};spells=v.own(s,8)
    board=set(case['board'])<=monsters and any(x['code']==JUKE and x['position']&5 for x in spells) and any(x['code']==SYNC and not x['position']&5 for x in spells)
    if case['mixed']:board=board and {v.PAR,v.RHAP}<={x['code'] for x in spells if x['position']&5} and any(x['code']==v.REG for x in v.own(s,2))
    return bool(board)
def run(case):
    env,obs,info=v.create(case);teacher=Teacher(case);trace=[];failure=None;own_end=None;recovered=None;ash_used=False;opphand_before=None;negated=None;recruit=None;normal=0
    opponent_steps=[a('SpSummon',v.RUK),a('Activate',v.RUK)]
    try:
        for step in range(600):
            s=v.native.benchmark_snapshot()
            if s['turn_player']==0:normal=max(normal,s['normal_summons'])
            if s['turn_player']==1 and s['chain_size']==0 and own_end is None:
                own_end=v.compact(s)
                if teacher.steps:raise RuntimeError('Own turn ended with plan unfinished')
                if not case['recover']:break
                teacher.steps=[a('Activate',REMIX),a('None',v.ASH),a('None',case.get('recovery_buddy',RECO)),a('None',v.ASH)]
            if recruit is None and any(x['code']==v.ASH for x in v.own(s,4)):
                recruit=v.compact(s)
            if own_end is not None and case['recover'] and recovered is None and s['chain_size']==0 and any(x['code']==v.ASH for x in v.own(s,2)) and any(x['code']==case.get('recovery_buddy',RECO) for x in v.own(s,4)):
                recovered=v.compact(s)
            if ash_used and s['chain_size']==0:
                now=collections.Counter(x['code'] for x in s['cards'] if x['controller']==1 and x['location']==2)
                negated=now==opphand_before and any(x['code']==v.ASH for x in v.own(s,16));break
            assert len(s['actions'])==int(info['num_options'][0]) and int(info['raw_num_options'][0])<=128
            for action in s['actions']:
                row=obs['actions_'][0][action['index']];cid=int(row[1])*256+int(row[2]);assert v.engine.ID_CODES.get(cid,0)==action['code']
            if s['to_play']==0:
                opposing_search=bool(recovered and s['public_chain'] and s['public_chain'][-1]['player']==1 and s['public_chain'][-1]['code']==v.RUK)
                ash=next((x for x in s['actions'] if x['code']==v.ASH and x['act']=='Activate' and x['spec'].startswith('h')),None)
                if opposing_search and ash and not ash_used:idx=ash['index'];ash_used=True
                elif recovered:idx=v.passive(s)
                else:idx=teacher(s)
            elif recovered and s['phase']==4:
                target=opponent_steps[0] if opponent_steps else None
                match=next((x for x in s['actions'] if target and all(x[k]==value for k,value in target.items())),None)
                if match:
                    idx=match['index'];opponent_steps.pop(0)
                    if match['act']=='Activate':opphand_before=collections.Counter(x['code'] for x in s['cards'] if x['controller']==1 and x['location']==2)
                elif s['message']=='select_place':idx=next(x['index'] for x in s['actions'] if x['place']==3)
                elif s['message']=='select_position':idx=0
                else:idx=v.passive(s)
            else:idx=v.passive(s)
            trace.append(dict(step=step,snapshot=v.compact(s),selected=s['actions'][idx],remaining_plan=list(teacher.steps)))
            obs,_,term,trunc,info=env.step(v.np.array([idx],v.np.int32))
            if term[0] or trunc[0]:failure='terminated_or_truncated';break
        else:failure='decision_limit'
    except Exception as exc:failure=str(exc)
    final=v.compact(v.native.benchmark_snapshot());board=own_end is not None and board_goal(case,own_end)
    recovered_ok=not case['recover'] or recruit is not None and recovered is not None and negated is True
    result=dict(case=case,passed=bool(board and recovered_ok and failure is None),first_turn_board_passed=bool(board),failure=failure,trace=trace,
        own_end=own_end,ash_recruited=recruit,ash_recovered=recovered,recovered_ash_activated=ash_used,opponent_search_negated=negated,
        final=final,remaining_plan=teacher.steps,normal_summons_diagnostic=normal,normal_summon_usage_scored=False,lua_errors=v.native.get_lua_error_count(),
        trained=False,privacy_certified_for_training=False,opponent_behaviour='Pass except a Rukina summon and search after recovery. No enemy disruption of the setup.')
    untouched=collections.Counter(case.get('unused_initial_cards',[]))
    result['unused_initial_cards_preserved']=own_end is not None and untouched<=collections.Counter(x['code'] for x in v.own(own_end,2))
    ash_choice=next((t for t in trace if t['selected']['code']==v.ASH and t['snapshot']['message']=='select_card' and t['snapshot']['public_chain'] and t['snapshot']['public_chain'][-1]['code']==CUE),None)
    # The benchmark snapshot omits the main deck. Cue's legal zones are hand/deck/graveyard:
    # if no Ash is in hand or graveyard at this native selection, the selected copy is from deck.
    result['ash_from_deck_verified']=not case['recover'] or recruit is not None and ash_choice is not None and not any(x['code']==v.ASH for loc in [2,16] for x in v.own(ash_choice['snapshot'],loc)) and sum(x['code']==v.ASH for x in v.own(recruit,4))==1
    result['ash_recruit_source_proof']='Cue selection; no Ash in own hand/graveyard before selection; one Ash appears on field.' if case['recover'] else None
    result['passed']=result['passed'] and result['unused_initial_cards_preserved'] and result['ash_from_deck_verified']
    result['passed']=result['passed'] and result['lua_errors']==0
    env.close();del env;gc.collect();assert v.native.get_env_resource_stats()['live_duels']==0
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--repeat',type=int,default=1);args=p.parse_args()
    out=v.P/'evaluation/spirit-ash-openings-20261108';out.mkdir(exist_ok=True,parents=True);rows=[]
    for i,base in enumerate(make_cases()):
        if args.case and args.case!=base['case_id']:continue
        for repeat in range(args.repeat):
            fillers=[[v.VEILER,v.DROLL,v.G],[v.OGRE,v.VEILER,v.VEILER],[v.DROLL,v.DROLL,v.G]][repeat%3]
            hand=base['hand'][:2]+fillers;random.Random(63100+i*20+repeat).shuffle(hand)
            case=dict(base,hand=hand,required_starters=base['hand'][:2],unused_initial_cards=fillers,arm='pass',seed=63100+i*20+repeat,
                case_id=base['case_id']+f'-s{repeat}',recovery_buddy=MIX if base['case_id']=='cue-mix-ash' else RECO)
            result=run(case);v.write(out/(case['case_id']+'.json'),result)
            row={k:result[k] for k in ['passed','first_turn_board_passed','failure','remaining_plan','recovered_ash_activated','opponent_search_negated','lua_errors']};row.update(case_id=case['case_id'])
            display=dict(row);display['failure']=str(row['failure'])[:250] if row['failure'] else None;display['remaining_plan']=row['remaining_plan'][:3]
            print(json.dumps(display,ensure_ascii=False),flush=True);rows.append(row)
    v.write(out/('summary-'+(args.case or 'all')+'.json'),dict(cases=rows,passed=sum(x['passed'] for x in rows),total=len(rows),trained=False,normal_summon_usage_scored=False))
