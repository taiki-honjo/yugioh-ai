"""Rule-backed Rukina opening and five explicit Ash/Veiler branches."""
import collections,gc,hashlib,json,os,random,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).parent))
import verify_opponent_article_basics as engine
import numpy as np
P=engine.P;R=engine.R;native=engine.native;ygoenv=engine.ygoenv;EnvPreprocess=engine.EnvPreprocess
names=engine.names;NAME='ElfnotesKewlTune';DECK=P/'assets/deck/ElfnotesKewlTune.ydk'
RUK=13597785;SPIRIT=12375297;DINA=59581480;REG=56651978;FORT=85976588
PAR=64491754;RHAP=24092792;STREL=42302563;CRYSTAL=50954680;BARONNE=84815190
ASH=14558127;VEILER=97268402;DROLL=94145021;G=23434538;OGRE=59438930
ARMS=['pass','ash_rukina','veiler_rukina','ash_spirit','ash_parallelism','veiler_strelitzia']
def a(act,code,**kw):return dict(act=act,code=code,**kw)
def choose(code,loc=2):return dict(act='None',code=code,after_location=loc)
def tail_plan():
    return [a('SpSummon',DINA),a('Activate',DINA),choose(PAR,8),a('Activate',PAR),choose(DINA,16),choose(REG,4),choose(FORT,4),
        a('Activate',FORT),choose(RHAP,8),a('Activate',RHAP),choose(FORT,16),choose(SPIRIT,4),a('Activate',SPIRIT),a('SpSummon',BARONNE)]
def plan():
    return [a('SpSummon',RUK),a('Activate',RUK),choose(SPIRIT),a('Summon',SPIRIT),a('SpSummon',STREL),choose(DINA),
            a('Activate',STREL),choose(SPIRIT,4),a('SpSummon',CRYSTAL)]+tail_plan()
def read_deck():
    result={'main':[],'extra':[]};section='main'
    for line in DECK.read_text().splitlines():
        if line=='#extra':section='extra'
        elif line=='!side':section='side'
        elif line.isdigit() and section in result:result[section].append(int(line))
    return result
CARDS=read_deck();OPP_HAND=[RUK,ASH,VEILER,DROLL,G]
def fixture(case):
    rows=[];rng=random.Random(case['seed'])
    for player in [0,1]:
        remaining=list(CARDS['main']);hand=case['hand'] if player==0 else OPP_HAND
        for i,code in enumerate(hand):remaining.remove(code);rows.append([code,player,player,2,i,8])
        rng.shuffle(remaining)
        rows.extend([code,player,player,1,i,8] for i,code in enumerate(remaining))
        rows.extend([code,player,player,64,i,8] for i,code in enumerate(CARDS['extra']))
    assert collections.Counter((row[1],row[0]) for row in rows)==collections.Counter((p,c) for p in [0,1] for c in CARDS['main']+CARDS['extra'])
    return rows
def create(case):
    native.set_benchmark_fixture(fixture(case))
    raw=ygoenv.make('YGOPro-v1',env_type='gymnasium',num_envs=1,num_threads=1,seed=case['seed'],deck1=NAME,deck2=NAME,
        player=-1,play_mode='self',max_options=128,n_history_actions=32,oppo_info=False,async_reset=False,
        max_steps=2048,max_episode_steps=2048,timeout=10)
    raw.num_envs=1;env=EnvPreprocess(raw,skip_mask=True);obs,info=env.reset();return env,obs,info
def own(s,loc):return [x for x in s['cards'] if x['controller']==0 and x['location']==loc]
def passive(s):return engine.passive(s)
def compact(s):return engine.compact(s)
def check_observation(obs,info,s):return engine.check_observation(obs,info,s)
def features(info):
    length=np.asarray(info['public_chain_length'],np.int32);owner=np.asarray(info['public_chain_player'],np.int32);player=np.asarray(info['to_play'],np.int32)
    return np.stack([length,np.where(length>0,np.where(owner==player,1,2),0),np.asarray(info['public_chain_card_id'],np.int32),np.asarray(info['public_chain_effect'],np.int32)],axis=-1)
def write(p,obj):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
class Interruption:
    def __init__(self,arm):self.arm=arm;self.used=False;self.target_pending=None;self.resolved=None;self.countered=False
    def act(self,s):
        chain=s['public_chain'];acts=s['actions']
        if self.target_pending and s['message'] in ['select_card','select_unselect_card']:
            found=next((a for a in acts if a['code']==self.target_pending),None)
            if found:self.target_pending=None;return found['index']
        target={'ash_rukina':RUK,'veiler_rukina':RUK,'ash_spirit':SPIRIT,'ash_parallelism':PAR,'veiler_strelitzia':STREL}.get(self.arm)
        handtrap=VEILER if self.arm.startswith('veiler') else ASH
        effect_ok=not (self.arm=='ash_spirit' and chain and (chain[-1]['description']&15)!=1)
        eligible=(not self.used and target is not None and chain and chain[-1]['player']==0 and chain[-1]['code']==target and effect_ok)
        choices=[a for a in acts if a['act']=='Activate' and a['code']==handtrap]
        if eligible and choices:
            self.used=True;self.target_pending=target if handtrap==VEILER else None;return choices[0]['index']
        return passive(s)
    def observe(self,s):
        if self.used and self.resolved is None and not s['public_chain'] and s['chain_size']==0:
            self.resolved=compact(s);self.target_pending=None
class Teacher:
    def __init__(self,case,interruption):
        self.case=case;self.interruption=interruption;self.steps=list(case.get('plan_override',plan()));self.context=None;self.materials=set();self.switched=False
    def __call__(self,s):
        acts=s['actions'];msg=s['message'];arm=self.case['arm']
        if self.interruption.resolved is not None and not self.switched:
            if arm in ['ash_rukina','veiler_rukina']:
                self.steps=[a('Summon',VEILER),a('SpSummon',STREL),a('Activate',STREL),choose(RUK,4)]
            elif arm=='ash_spirit':self.steps=[a('Activate',STREL),choose(SPIRIT,4),a('SpSummon',CRYSTAL)]
            elif arm=='veiler_strelitzia':self.steps=tail_plan()
            self.switched=True
        # Answer an opponent's Ash with the already established Crystal Wing.
        if s['public_chain'] and s['public_chain'][-1]['player']==1 and s['public_chain'][-1]['code']==ASH:
            counter=next((x for x in acts if x['code']==CRYSTAL and x['act']=='Activate'),None)
            if counter:self.interruption.countered=True;return counter['index']
        if s['phase']!=4:return passive(s)
        if msg in ['select_chain','select_effectyn']:
            for x in acts:
                if x['act']=='Activate' and ((x['code']==SPIRIT and x['spec'].startswith('g')) or (x['code']==REG and x['spec'].startswith(('g','m')))):
                    return x['index']
        if msg in ['select_card','select_unselect_card','select_sum','select_tribute'] and self.context in [STREL,CRYSTAL,BARONNE]:
            tuner=VEILER if arm in ['ash_rukina','veiler_rukina'] else SPIRIT
            required=self.case.get('materials_override',{}).get(str(self.context),{STREL:[tuner,RUK],CRYSTAL:[SPIRIT,STREL],BARONNE:[SPIRIT,REG]}[self.context])
            if any(x['code'] in required and x['spec'].startswith('m') for x in acts):
                for code in required:
                    found=next((x for x in acts if x['code']==code and x['spec'].startswith('m') and code not in self.materials),None)
                    if found:self.materials.add(code);return found['index']
                finish=next((x for x in acts if x['finish'] or x['act']=='Cancel'),None)
                if finish:return finish['index']
        while self.steps:
            target=self.steps[0]
            found=next((x for x in acts if all(x.get(k)==value for k,value in target.items() if k!='after_location')),None)
            if found:
                self.steps.pop(0)
                if found['act'] in ['SpSummon','Summon']:self.context=found['code'];self.materials.clear()
                return found['index']
            if 'after_location' in target and any(x['code']==target['code'] for x in own(s,target['after_location'])):
                self.steps.pop(0);continue
            break
        if msg in ['select_chain','select_effectyn','select_yesno']:return passive(s)
        if msg in ['select_place','select_disfield']:
            preferred=3 if arm in ['ash_rukina','veiler_rukina'] else 6 if self.context==CRYSTAL else 1 if self.context in [STREL,BARONNE] else 3
            return next((x['index'] for x in acts if x['place']==preferred),0)
        if msg=='select_position':return next((x['index'] for x in acts if x['position']==1),0)
        if not self.steps and msg=='select_idlecmd':return passive(s)
        if len(acts)==1:return 0
        raise RuntimeError('Unresolved recovery witness: '+json.dumps(dict(steps=self.steps,message=msg,actions=acts)))
def goal(case,s,interruption):
    field=[x for x in own(s,4) if x['position']&5];monsters={x['code'] for x in field};spells={x['code'] for x in own(s,8) if x['position']&5}
    hand=collections.Counter(x['code'] for x in own(s,2));arm=case['arm']
    if case.get('target_profile')=='two_bosses':
        board={CRYSTAL,BARONNE}<=monsters and {PAR,RHAP}<=spells
        label='Research witness: Crystal Wing + Baronne + Parallellism + Rhapsodia; Regina hand retention is not required.'
    elif case.get('family') in ['dina_review','regina_review']:
        board=any(x['code']==REG and x['sequence']==2 for x in field) and any(x['code']==RUK and x['sequence']!=2 for x in field) and PAR in spells and hand[FORT]>0
        label='Existing Dina/Regina opening review; no normal-summon condition.'
    elif arm in ['ash_rukina','veiler_rukina']:
        board={STREL,RUK}<=monsters and any(x['code']==STREL and x['sequence']==2 for x in field) and any(x['code']==RUK and x['sequence']!=2 for x in field)
        label='Strelitzia in center plus side Rukina; requires a level-1 tuner in the opening hand.'
    elif arm=='ash_spirit':board=CRYSTAL in monsters;label='Crystal Wing after the Spirit search was negated.'
    elif arm=='veiler_strelitzia':board={STREL,BARONNE}<=monsters and {PAR,RHAP}<=spells and hand[REG]>0;label='Strelitzia plus Baronne and both continuous spells after the revival was negated.'
    else:board={CRYSTAL,BARONNE}<=monsters and {PAR,RHAP}<=spells and hand[REG]>0;label='User-provided full target; Parallellism Ash is answered by Crystal Wing when present.'
    disruption=(arm=='pass' or interruption.used and interruption.resolved is not None)
    if arm=='ash_parallelism':disruption=disruption and interruption.countered
    complete=s['turn_player']==1 and s['chain_size']==0 and not s['done']
    return dict(passed=bool(board and disruption and complete),board=bool(board),interruption_verified=bool(disruption),turn_completed=bool(complete),
        normal_summon_usage_scored=False,scope=label)
def run(case,policy=None,collect=False):
    env,obs,info=create(case);interruption=Interruption(case['arm']);teacher=Teacher(case,interruption);trace=[];examples=[];failure=None;normal=0
    if policy:policy.reset()
    try:
        for step in range(320):
            s=native.benchmark_snapshot();interruption.observe(s);check_observation(obs,info,s)
            if s['turn_player']==0:normal=max(normal,s['normal_summons'])
            if s['turn_player']==1 and s['chain_size']==0:break
            assert int(info['raw_num_options'][0])<=128
            if s['to_play']==0:
                index=policy.act(obs,info) if policy else teacher(s)
                if s['public_chain'] and s['public_chain'][-1]['player']==1 and s['public_chain'][-1]['code']==ASH and s['actions'][index]['code']==CRYSTAL and s['actions'][index]['act']=='Activate':interruption.countered=True
                if collect:examples.append(dict(obs={k:np.asarray(obs[k])[0].copy() for k in ['cards_','global_','actions_','h_actions_']},chain=features(info)[0].copy(),label=index))
            else:index=interruption.act(s)
            trace.append(dict(step=step,player=s['to_play'],snapshot=compact(s),selected=s['actions'][index],remaining_plan=list(teacher.steps) if not policy else None))
            obs,_,term,trunc,info=env.step(np.array([index],np.int32))
            if term[0] or trunc[0]:failure='terminated_or_truncated';break
        else:failure='decision_limit'
    except Exception as exc:failure=str(exc)
    final=compact(native.benchmark_snapshot());result=goal(case,final,interruption)
    result.update(case=case,failure=failure,trace=trace,final=final,normal_summons_diagnostic=normal,
        interruption_injected=interruption.used,interruption_countered=interruption.countered,interruption_resolution=interruption.resolved,
        lua_errors=native.get_lua_error_count(),model_received_plan=False if policy else None,remaining_plan=list(teacher.steps) if not policy else None)
    result['passed']=result['passed'] and failure is None and result['lua_errors']==0
    env.close();del env;gc.collect();assert native.get_env_resource_stats()['live_duels']==0
    return result,examples
if __name__=='__main__':
    out=P/'evaluation/rukina-recovery-20261108/probes';out.mkdir(parents=True,exist_ok=True)
    for i,arm in enumerate(ARMS):
        case=dict(case_id='probe-'+arm,arm=arm,hand=[RUK,VEILER,ASH,DROLL,G],seed=51200+i)
        result,_=run(case);write(out/(arm+'.json'),result)
        print(arm,json.dumps({k:result[k] for k in ['passed','failure','board','interruption_injected','interruption_countered','normal_summons_diagnostic','remaining_plan']},ensure_ascii=False),flush=True)
        if not result['passed']:raise SystemExit(1)
