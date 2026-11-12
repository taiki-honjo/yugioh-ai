"""Verified reference bank. Cases are teacher/scoring metadata and are never model inputs."""
import copy,json
import all_opening_witness as w
v=w.v;P=w.P
def read(p):return json.loads(p.read_text())
def crystal_core(search):
    a=v.a;c=v.choose
    return [c(v.SPIRIT,4),a('SpSummon',v.STREL),c(search),a('Activate',v.STREL),c(v.SPIRIT,4),a('SpSummon',v.CRYSTAL)]
def crystal_finish():
    a=v.a;c=v.choose
    return [a('Activate',v.RHAP),c(v.REG,16),c(v.SPIRIT,4),a('Activate',v.SPIRIT),a('SpSummon',v.BARONNE)]
def crystal_plans():
    a=v.a;c=v.choose
    start_ruk=[a('SpSummon',v.RUK),a('Activate',v.RUK),c(v.REG),a('Activate',v.REG),c(v.RUK,16)]
    dina_tail=[a('SpSummon',v.DINA),a('Activate',v.DINA),c(v.PAR,8),a('Activate',v.PAR),c(v.DINA,16),c(v.FORT,4),a('Activate',v.FORT),c(v.RHAP,8)]+crystal_finish()
    fort_tail=[a('SpSummon',v.FORT),a('Activate',v.FORT),c(v.RHAP,8)]+crystal_finish()
    return {
      'rukina-regina':start_ruk+crystal_core(v.DINA)+dina_tail,
      'dina':[a('SpSummon',v.DINA),a('Activate',v.DINA),c(v.PAR,8),a('Activate',v.PAR),c(v.DINA,16),c(v.RUK,4),a('Activate',v.RUK),c(v.REG),a('Activate',v.REG),c(v.RUK,16)]+crystal_core(v.FORT)+fort_tail,
      'par-light-cost':[a('Activate',v.PAR),a('Activate',v.PAR),c(v.VEILER,16),c(v.RUK,4),a('Activate',v.RUK),c(v.REG),a('Activate',v.REG),c(v.RUK,16)]+crystal_core(v.FORT)+fort_tail,
      'par-fire-cost':[a('Activate',v.PAR),a('Activate',v.PAR),c(v.ASH,16),c(v.REG,4)]+crystal_core(v.FORT)+fort_tail,
      'regina-par':[a('Activate',v.REG),c(v.PAR,16)]+crystal_core(v.DINA)+dina_tail}
def bank(include_baronne=False):
    out=[]
    for arm in v.ARMS:
        out.append(dict(kind='rukina',family='rukina-'+arm,eval_family='rukina-'+arm,case_id='rukina-'+arm,arm=arm,required=[v.RUK,v.VEILER]))
    required={'rukina-regina':[v.RUK],'dina':[v.DINA],'par-light-cost':[v.PAR,v.VEILER],'par-fire-cost':[v.PAR,v.ASH],'regina-par':[v.REG,v.PAR]}
    for key,prefix in required.items():
        case=read(P/'evaluation/elfnotes-other-starts-20261108'/(key+'.json'))['case']
        case.update(kind='elf',family='elf-'+key,eval_family='elf-'+key,required=prefix,plan_override=crystal_plans()[key],prefer_crystal_first=True,
            materials_override={str(v.STREL):[v.SPIRIT,v.REG],str(v.BARONNE):[v.SPIRIT,v.FORT]})
        out.append(case)
    required={'cue':[w.spirit.CUE],'reco':[w.spirit.RECO],'sync':[w.spirit.SYNC],'stone-sweeper':[72323266],
        'juke-veiler':[w.spirit.JUKE,v.VEILER],'rukina-cue':[v.RUK,w.spirit.CUE]}
    for key,prefix in required.items():
        case=read(P/'evaluation/kewltune-current-recipe-20261108'/(key+'.json'))['case']
        case.update(kind='tune',family=('mixed-' if key=='rukina-cue' else 'tune-')+key,required=prefix,mixed=key=='rukina-cue')
        if key=='rukina-cue':case.update(plan_override=crystal_plans()['rukina-regina']+w.kt.tune_plan(),prefer_crystal_first=True,
            materials_override={str(v.STREL):[v.SPIRIT,v.REG],str(v.BARONNE):[v.SPIRIT,v.FORT]})
        case['eval_family']=case['family'];out.append(case)
    for case in w.spirit.make_cases():
        if case['case_id'] not in ['reco-spirit-ash','cue-spirit']:continue
        case.update(kind='spirit',family='mixed-'+case['case_id'],eval_family='mixed-'+case['case_id'],required=case['hand'][:2],arm='pass',recovery_buddy=w.spirit.RECO)
        # Spirit has already supplied the first Trackmaker from hand. Reuse it through
        # Strelitzia to establish Crystal Wing before returning to the Tune engine.
        plan=case['plan'];at=next(i for i,x in enumerate(plan) if x['act']=='Activate' and x['code']==v.SPIRIT)
        case['plan']=plan[:at]+[v.a('SpSummon',v.STREL),v.a('Activate',v.STREL),v.choose(v.SPIRIT,4),v.a('SpSummon',v.CRYSTAL)]+plan[at+2:]
        case['materials'].update({v.STREL:[[(v.SPIRIT,'m'),(v.REG,'m')]],v.CRYSTAL:[[(v.SPIRIT,'m'),(v.STREL,'m')]]})
        case['board']=[v.CRYSTAL if code==v.BARONNE else code for code in case['board']]
        case['prefer_crystal_first']=True
        out.append(case)
    variants=w.jj.make_cases()
    for case in variants:
        prefix=case['hand'][:2]
        pair='jj-'+('cue-reco' if w.spirit.RECO in prefix else 'cue-mix' if w.spirit.MIX in prefix else 'cue-rotary')
        endings=[dict(id=x['case_id'],board=x['board'],retained=x.get('retained')) for x in variants if x['hand'][:2]==prefix]
        case.update(kind='jj',family=case['case_id'],eval_family=pair,required=prefix,accepted_endings=endings,arm='pass',mixed=False,recover=True)
        out.append(case)
    if include_baronne:
        for case in w.spirit.make_cases():
            if case['case_id'] not in ['reco-spirit-ash','cue-spirit']:continue
            name='mixed-'+case['case_id'];case.update(kind='spirit',family=name+'-baronne',eval_family=name,required=case['hand'][:2],arm='pass',
                recovery_buddy=w.spirit.RECO,prefer_crystal_first=False)
            out.append(case)
    assert len(out)==(26 if include_baronne else 24)
    return out
def normalize(case):
    case=copy.deepcopy(case)
    if 'materials' in case:case['materials']={int(k):value for k,value in case['materials'].items()}
    return case
