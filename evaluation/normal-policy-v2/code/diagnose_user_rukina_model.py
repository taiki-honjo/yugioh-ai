import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from inspect_full_turn import *
U=P/'evaluation/normal-policy-v2/user-reference'
target=json.loads((U/'target.json').read_text());case=dict(target['case'],suite='user_reference')
selected=Path(json.loads((S/'training/result.json').read_text())['selected_checkpoint'])
policy=Policy(selected/'base.flax_model',selected/'chain-head.flax_model')
d=run(case,policy,'pass')
field={x['code'] for x in d['final']['cards'] if x['location']==4 and x['position']&5}
spells={x['code'] for x in d['final']['cards'] if x['location']==8 and x['position']&5}
hand=collections.Counter(x['code'] for x in d['final']['cards'] if x['location']==2)
d['user_target']=dict(monsters=set(target['target']['monsters'])<=field,spells=set(target['target']['spells'])<=spells,
    regina_in_hand=hand[c.REG]>0,protected=all(hand[k]>=v for k,v in collections.Counter(c.FILL).items()),
    normal_summons=d['normal_summons_used'],turn_completed=d['turn_completed'])
d['passed_user_target']=all(d['user_target'][k] for k in ['monsters','spells','regina_in_hand','protected','turn_completed'])
d['model']=dict(base=str(selected/'base.flax_model'),base_sha256=sha(selected/'base.flax_model'),head_sha256=sha(selected/'chain-head.flax_model'))
write(U/'model-diagnostic.json',d);print('CURRENT_MODEL_USER_TARGET',d['passed_user_target'],d['user_target'],flush=True)
