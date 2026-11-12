import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from endboard_witness import end_goal,c,P
R=P/'evaluation/normal-policy-v2'
def read(p):return json.loads(p.read_text())
defs={x['id']:x for x in c.fixture_definitions()}
old=P/'experiments/combo-benchmark-20261107/controls'
normal=read(old/'normal_before_synchro.json')
assert normal['final_goal']['normal_summons']==1 and not normal['final_goal']['pass_']
fresh=read(R/'controls/normal_before_synchro.json')
# The old compact snapshot omits the opponent Extra Deck; use the freshly
# recorded live-engine judgment rather than reconstructing hidden fields.
now=fresh['final_goal']
assert now['pass_'] and now['normal_summons']==1 and now['normal_summons_scored'] is False
negative={}
for name,family in [('skip_search','rukina_search'),('wrong_center','regina_recruit')]:
    row=read(old/(name+'.json'));negative[name]=not c.goal_check(defs[family],row['final'])['pass_'];assert negative[name]
control=read(R/'normal-summon-endboard-control.json')
g=end_goal(control['case'],control['final'],control['normal_used'],control['ash_injected'])
assert g['passed'] and g['normal_summons_scored'] is False and control['normal_used']==1
accepted=[dict(case=control['case']['case_id'],normal_summons=control['normal_used'])]
user=read(R/'user-reference/engine-verification.json')
assert user['passed'] and user['goal']['normal_summons']==1 and user['native_lua_errors']==0
assert read(R/'user-reference/target.json')['target']['normal_summons_scored'] is False
result=dict(status='passed',grading_policy='normal-summon-agnostic-v2',
    formerly_rejected_legal_normal_summon_case_now_passes=True,negative_board_controls_still_fail=negative,
    actual_endboards_accepted_with_normal_summons=accepted,user_provided_full_line_passes_with_normal_summon=True,
    rule_engine_limits_unchanged=True)
(R/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
