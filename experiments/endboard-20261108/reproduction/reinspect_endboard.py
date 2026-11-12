import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from inspect_full_turn import *
selected=Path(json.loads((S/'training/result.json').read_text())['selected_checkpoint'])
policy=Policy(selected/'base.flax_model',selected/'chain-head.flax_model')
cases=json.loads((S/'cases.json').read_text());summary={}
for arm in ['pass','ash_once']:
    out=S/'reinspection'/arm;out.mkdir(parents=True,exist_ok=False);results=[]
    for case in cases:
        d=run(case,policy,arm);write(out/(case['case_id']+'.json'),d);results.append(d)
    codes={14558127,97268402,94145021,23434538,59438930,52038441}
    summary[arm]=dict(cases=len(results),turn_completed=sum(d['turn_completed'] for d in results),
        ash_injected=sum(d['ash_injected'] for d in results),
        facedown_handtrap_endboard_cases=sum(any(x['location']==4 and not x['position']&5 and x['code'] in codes for x in d['final']['cards']) for d in results),
        self_ash_actions=sum(len(d['self_ash_against_own_chain']) for d in results),
        caveat='Reused diagnostic starting states, not independent holdout.')
    print('REINSPECTION',arm,summary[arm],flush=True)
write(S/'reinspection/summary.json',dict(status='complete',results=summary,lua_errors=c.native.get_lua_error_count()))
