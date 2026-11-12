"""User-corrected two-Kewl-Tune lines: tribute recruited Ash to JJ, keep Track/Remix or Zaren/Remix."""
import argparse,json,random
import verify_spirit_and_ash_lines as base
v=base.v;a=base.a;c=base.c
CUE=base.CUE;RECO=base.RECO;MIX=base.MIX;ROTARY=base.ROTARY;TRACK=base.TRACK;REMIX=base.REMIX;JUKE=base.JUKE;SYNC=base.SYNC
ZAREN=4891376
def reco_prefix():
    return [a('Summon',RECO),a('Activate',RECO),c(ROTARY),a('SpSummon',TRACK),a('Activate',ROTARY),a('Activate',TRACK),c(JUKE),c(SYNC),
        a('Activate',JUKE),a('Summon',CUE),a('Activate',CUE),c(v.ASH,4),a('Activate',JUKE),c(v.ASH,16),c(MIX,4),a('Activate',MIX),c(RECO)]
def rotary_prefix():
    return [a('Summon',CUE),a('Activate',CUE),c(v.ASH,4),a('SpSummon',TRACK),a('Activate',CUE),a('Activate',ROTARY),a('Activate',TRACK),c(SYNC),c(JUKE),
        a('Activate',JUKE),a('Activate',JUKE),c(v.ASH,16),c(RECO,4),a('Activate',RECO),c(MIX),a('Summon',MIX),a('Activate',MIX),c(CUE)]
def make_cases():
    filler=[v.VEILER,v.DROLL,v.G]
    return [
      dict(case_id='cue-reco-jj-zaren',hand=[CUE,RECO]+filler,board=[ZAREN,REMIX],recovery_buddy=RECO,
        plan=reco_prefix()+[a('SpSummon',REMIX),a('SpSummon',ZAREN),a('Activate',CUE),a('Set',SYNC)],
        materials={TRACK:[[(RECO,'m'),(ROTARY,'h')]],REMIX:[[(MIX,'m'),(RECO,'h')]],ZAREN:[[(TRACK,'m'),(CUE,'m')]]}),
      dict(case_id='cue-reco-jj-track',hand=[CUE,RECO]+filler,board=[TRACK,REMIX],recovery_buddy=MIX,retained=RECO,
        plan=reco_prefix()+[a('SpSummon',REMIX),a('Activate',CUE),a('Set',SYNC)],
        materials={TRACK:[[(RECO,'m'),(ROTARY,'h')]],REMIX:[[(MIX,'m'),(CUE,'m')]]}),
      dict(case_id='cue-mix-jj-track',hand=[CUE,MIX]+filler,board=[TRACK,REMIX],recovery_buddy=RECO,retained=ROTARY,
        plan=[a('Summon',MIX),a('Activate',MIX),c(ROTARY),a('Activate',ROTARY),c(CUE),a('Summon',CUE),a('Activate',CUE),c(v.ASH,4),
            a('SpSummon',TRACK),a('Activate',CUE),a('Activate',ROTARY),a('Activate',TRACK),c(SYNC),c(JUKE),
            a('Activate',JUKE),a('Activate',JUKE),c(v.ASH,16),c(RECO,4),a('Activate',RECO),c(ROTARY),a('SpSummon',REMIX),a('Set',SYNC)],
        materials={TRACK:[[(CUE,'m'),(ROTARY,'h')]],REMIX:[[(MIX,'m'),(RECO,'m')]]}),
      dict(case_id='cue-rotary-jj-zaren',hand=[CUE,ROTARY]+filler,board=[ZAREN,REMIX],recovery_buddy=RECO,
        plan=rotary_prefix()+[a('SpSummon',REMIX),a('SpSummon',ZAREN),a('Set',SYNC)],
        materials={TRACK:[[(CUE,'m'),(ROTARY,'h')]],REMIX:[[(MIX,'m'),(CUE,'h')]],ZAREN:[[(TRACK,'m'),(RECO,'m')]]}),
      dict(case_id='cue-rotary-jj-track',hand=[CUE,ROTARY]+filler,board=[TRACK,REMIX],recovery_buddy=RECO,retained=CUE,
        plan=rotary_prefix()+[a('SpSummon',REMIX),a('Set',SYNC)],
        materials={TRACK:[[(CUE,'m'),(ROTARY,'h')]],REMIX:[[(MIX,'m'),(RECO,'m')]]}),
    ]
def check_release(r):
    trace=r['trace'];release=None
    for i,t in enumerate(trace):
        choice=t['selected']
        if choice['code']==v.ASH and choice['act']=='None' and choice['spec'].startswith('m'):
            preceding=trace[i-1]['selected'] if i else {}
            # Release is the native cost selection immediately following JJ's field effect.
            if preceding.get('code')==JUKE and preceding.get('act')=='Activate' and preceding.get('spec','').startswith('s'):
                release=t;break
    ash_material=[t['step'] for t in trace if t['snapshot']['turn_player']==0 and t['selected']['code']==v.ASH and t['selected']['spec'].startswith('m') and t['snapshot']['message'] in ['select_sum','select_unselect_card']]
    r['jj_released_recruited_ash']=release is not None
    r['jj_release_step']=release['step'] if release else None
    r['ash_used_as_synchro_material']=bool(ash_material)
    r['retained_requested_card']=not r['case'].get('retained') or r['own_end'] is not None and any(x['code']==r['case']['retained'] for x in v.own(r['own_end'],2))
    r['requested_exact_monster_board']=r['own_end'] is not None and sorted(x['code'] for x in v.own(r['own_end'],4))==sorted(r['case']['board'])
    r['passed']=r['passed'] and release is not None and not ash_material and r['retained_requested_card'] and r['requested_exact_monster_board']
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--repeat',type=int,default=1);args=p.parse_args()
    out=v.P/'evaluation/jj-ash-release-20261108';out.mkdir(parents=True,exist_ok=True);rows=[]
    for i,spec in enumerate(make_cases()):
        if args.case and args.case!=spec['case_id']:continue
        for repeat in range(args.repeat):
            filler=[[v.VEILER,v.DROLL,v.G],[v.OGRE,v.VEILER,v.VEILER],[v.DROLL,v.DROLL,v.G]][repeat%3]
            hand=spec['hand'][:2]+filler;random.Random(64100+i*20+repeat).shuffle(hand)
            case=dict(spec,hand=hand,required_starters=spec['hand'][:2],unused_initial_cards=filler,arm='pass',mixed=False,recover=True,
                seed=64100+i*20+repeat,case_id=spec['case_id']+f'-s{repeat}')
            r=check_release(base.run(case));v.write(out/(case['case_id']+'.json'),r)
            row={k:r[k] for k in ['passed','first_turn_board_passed','jj_released_recruited_ash','ash_used_as_synchro_material','recovered_ash_activated','opponent_search_negated','retained_requested_card','lua_errors']}
            row['case_id']=case['case_id'];row['failure']=str(r['failure'])[:250] if r['failure'] else None
            print(json.dumps(row),flush=True);rows.append(row)
    v.write(out/('summary-'+(args.case or 'all')+'.json'),dict(cases=rows,passed=sum(r['passed'] for r in rows),total=len(rows),trained=False,normal_summon_usage_scored=False))
