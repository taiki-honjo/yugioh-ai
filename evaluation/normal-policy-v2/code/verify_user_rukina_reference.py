"""Replay the user's AXEBgcKV line in the pinned local rules engine; never publishes."""
import sys,json,gc,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from chain_policy import *
U=P/'evaluation/normal-policy-v2/user-reference';U.mkdir(parents=True,exist_ok=True)
SPIRIT=12375297;STREL=42302563;CRYSTAL=50954680;BARONNE=84815190
case=dict(id='user_rukina_one_card',case_id='AXEBgcKV',hand=[c.RUK]+c.FILL,protected=c.FILL,seed=35108,
    plan=[c.action('SpSummon',c.RUK),c.action('Activate',c.RUK),c.choose(SPIRIT),c.action('Summon',SPIRIT),
          c.action('SpSummon',STREL),c.choose(c.DINA),c.action('Activate',STREL),c.choose(SPIRIT,4),
          c.action('SpSummon',CRYSTAL),c.action('SpSummon',c.DINA),c.action('Activate',c.DINA),c.choose(c.PAR,8),
          c.action('Activate',c.PAR),c.choose(c.DINA,16),c.choose(c.REG,4),c.choose(c.FORT,4),
          c.action('Activate',c.FORT),c.choose(c.RHAP,8),c.action('Activate',c.RHAP),c.choose(c.FORT,16),c.choose(SPIRIT,4),
          c.action('Activate',SPIRIT),c.action('SpSummon',BARONNE)])
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
write(U/'target.json',dict(source_url='https://duel-simulator.miyauchidp.dev/replay/AXEBgcKV',source_title='ルキナ1枚',
    source_playback_observed=dict(state='ended',operations=36),case=case,
    target=dict(monsters=[CRYSTAL,BARONNE],spells=[c.PAR,c.RHAP],hand=[c.REG],observed_normal_summons=1,normal_summons_scored=False,protected=c.FILL),
    note='The source shows one starter. Four inert hand traps are added as protected fillers for a legal five-card starting hand. No server publication.',
    normal_summon_count_is_diagnostic_only=True))
class Teacher(c.Witness):
    def __init__(self,case):super().__init__(case);self.summon_context=None;self.material_choices=set()
    def __call__(self,s):
        acts=s['actions'];msg=s['message']
        if s['phase']!=4:return c.passive(s)
        if msg in ['select_chain','select_effectyn']:
            for a in acts:
                if a['act']=='Activate' and ((a['code']==SPIRIT and a['spec'].startswith('g')) or (a['code']==c.REG and a['spec'].startswith(('g','m')))):
                    return a['index']
        if msg in ['select_card','select_unselect_card','select_sum','select_tribute'] and self.summon_context in [STREL,CRYSTAL,BARONNE]:
            required={STREL:[SPIRIT,c.RUK],CRYSTAL:[SPIRIT,STREL],BARONNE:[SPIRIT,c.REG]}[self.summon_context]
            # Material prompts use field specifications; search/revival prompts are handled by the plan.
            material_prompt=any(a['code'] in required and a['spec'].startswith('m') for a in acts)
            if material_prompt:
                for code in required:
                    for a in acts:
                        if a['code']==code and a['spec'].startswith('m') and code not in self.material_choices:
                            self.material_choices.add(code);return a['index']
                finish=next((a for a in acts if a['finish'] or a['act']=='Cancel'),None)
                if finish:return finish['index']
        if msg in ['select_place','select_disfield']:
            preferred=6 if self.summon_context==CRYSTAL else 1 if self.summon_context in [STREL,BARONNE] else 3
            return next((a['index'] for a in acts if a['place']==preferred),0)
        if not self.steps and msg=='select_idlecmd':return c.passive(s)
        index=super().__call__(s)
        if acts[index]['act']=='SpSummon':
            self.summon_context=acts[index]['code'];self.material_choices.clear()
        elif acts[index]['act']=='Summon':self.summon_context=acts[index]['code']
        return index
def run():
    c.native.set_benchmark_fixture(c.fixture_cards(case))
    env=c.ygoenv.make('YGOPro-v1',env_type='gymnasium',num_envs=1,num_threads=1,seed=case['seed'],deck1=c.NAME,deck2=c.NAME,
        player=-1,play_mode='self',max_options=128,n_history_actions=32,oppo_info=False,async_reset=False,max_steps=2048,max_episode_steps=2048,timeout=10)
    env.num_envs=1;env=c.EnvPreprocess(env,skip_mask=True);obs,info=env.reset();teacher=Teacher(case);trace=[];normal=0
    try:
        for step in range(256):
            s=c.native.benchmark_snapshot();collect_snapshot_checks(obs,info,s)
            if s['turn_player']==0:normal=max(normal,s['normal_summons'])
            if s['turn_player']==1 and s['chain_size']==0:break
            a=teacher(s) if s['to_play']==0 else c.passive(s)
            trace.append(dict(step=step,player=s['to_play'],message=s['message'],selected=s['actions'][a],
                snapshot=s,remaining_plan=list(teacher.steps)))
            obs,_,term,trunc,info=env.step(np.array([a],np.int32));assert not term[0] and not trunc[0]
        final=c.native.benchmark_snapshot();own=lambda loc:[x for x in final['cards'] if x['controller']==0 and x['location']==loc]
        hand=__import__('collections').Counter(x['code'] for x in own(2))
        field={x['code'] for x in own(4) if x['position']&5};spells={x['code'] for x in own(8) if x['position']&5}
        goal=dict(monsters={CRYSTAL,BARONNE}<=field,spells={c.PAR,c.RHAP}<=spells,regina_in_hand=hand[c.REG]>=1,
            protected=all(hand[k]>=v for k,v in __import__('collections').Counter(c.FILL).items()),
            normal_summons=normal,turn_completed=final['turn_player']==1 and final['chain_size']==0)
        passed=all(goal[k] for k in ['monsters','spells','regina_in_hand','protected','turn_completed'])
        result=dict(status='complete',passed=passed,goal=goal,trace=trace,final=final,source_url='https://duel-simulator.miyauchidp.dev/replay/AXEBgcKV',
            native_lua_errors=c.native.get_lua_error_count(),native=json.loads((B/'runtime.json').read_text()),trained_from_this_line=False)
        write(U/'engine-verification.json',result);assert passed,goal
        print('USER_REFERENCE_VERIFIED',json.dumps(goal),flush=True)
    except Exception:
        write(U/'failure-context.json',dict(case=case,trace=trace,snapshot=c.native.benchmark_snapshot(),remaining_plan=teacher.steps));raise
    finally:env.close();del env;gc.collect();assert c.native.get_env_resource_stats()['live_duels']==0
run()
