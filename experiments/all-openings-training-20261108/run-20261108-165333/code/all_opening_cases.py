"""Verified reference bank. Cases are teacher/scoring metadata and are never model inputs."""
import copy,json
import all_opening_witness as w
v=w.v;P=w.P
def read(p):return json.loads(p.read_text())
def bank():
    out=[]
    for arm in v.ARMS:
        out.append(dict(kind='rukina',family='rukina-'+arm,eval_family='rukina-'+arm,case_id='rukina-'+arm,arm=arm,required=[v.RUK,v.VEILER]))
    required={'rukina-regina':[v.RUK],'dina':[v.DINA],'par-light-cost':[v.PAR,v.VEILER],'par-fire-cost':[v.PAR,v.ASH],'regina-par':[v.REG,v.PAR]}
    for key,prefix in required.items():
        case=read(P/'evaluation/elfnotes-other-starts-20261108'/(key+'.json'))['case']
        case.update(kind='elf',family='elf-'+key,eval_family='elf-'+key,required=prefix)
        out.append(case)
    required={'cue':[w.spirit.CUE],'reco':[w.spirit.RECO],'sync':[w.spirit.SYNC],'stone-sweeper':[72323266],
        'juke-veiler':[w.spirit.JUKE,v.VEILER],'rukina-cue':[v.RUK,w.spirit.CUE]}
    for key,prefix in required.items():
        case=read(P/'evaluation/kewltune-current-recipe-20261108'/(key+'.json'))['case']
        case.update(kind='tune',family=('mixed-' if key=='rukina-cue' else 'tune-')+key,required=prefix,mixed=key=='rukina-cue')
        case['eval_family']=case['family'];out.append(case)
    for case in w.spirit.make_cases():
        if case['case_id'] not in ['reco-spirit-ash','cue-spirit']:continue
        case.update(kind='spirit',family='mixed-'+case['case_id'],eval_family='mixed-'+case['case_id'],required=case['hand'][:2],arm='pass',recovery_buddy=w.spirit.RECO)
        out.append(case)
    variants=w.jj.make_cases()
    for case in variants:
        prefix=case['hand'][:2]
        pair='jj-'+('cue-reco' if w.spirit.RECO in prefix else 'cue-mix' if w.spirit.MIX in prefix else 'cue-rotary')
        endings=[dict(id=x['case_id'],board=x['board'],retained=x.get('retained')) for x in variants if x['hand'][:2]==prefix]
        case.update(kind='jj',family=case['case_id'],eval_family=pair,required=prefix,accepted_endings=endings,arm='pass',mixed=False,recover=True)
        out.append(case)
    assert len(out)==24
    return out
def normalize(case):
    case=copy.deepcopy(case)
    if 'materials' in case:case['materials']={int(k):value for k,value in case['materials'].items()}
    return case
