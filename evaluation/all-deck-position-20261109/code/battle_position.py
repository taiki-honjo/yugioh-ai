"""Legal summon-position choices using public stats when no attack is planned.

An external decision helper, not retrained model weights. Later own Main1/Battle
keeps the model choice unless the caller supplies an explicit attack intent.
"""
import json
from pathlib import Path
import numpy as np
TRACK=42781164;STREL=42302563;REMIX=88170262;LOUD=41069676

def load_card_data():return json.loads(Path(__file__).with_name('position_cards.json').read_text())

class PositionDecision:
    def __init__(self,data):self.data=data;self.events=[]
    def reset(self):self.events=[]
    def choose(self,code,positions,raw,turn,own_turn,phase_after_battle,public_tuner=False,zone=None,attack_intent=None):
        meta=self.data.get(str(code));chosen=raw;atk=defense=None;reason='model-choice';source='public-card-data'
        allowed=[i for i,p in enumerate(positions) if p in [1,4]]
        if len(allowed)<2:reason='forced-legal-position'
        elif positions[raw] not in [1,4]:reason='face-down-choice-model-choice'
        elif attack_intent is True:
            chosen=positions.index(1);reason='explicit-attack-plan'
        elif attack_intent is None and turn>1 and own_turn and not phase_after_battle:
            reason='attack-plan-uncertain-model-choice'
        elif not meta:reason='unknown-card-model-choice'
        elif meta[3]:reason='position-dependent-effect-model-choice'
        elif len(meta)>4 and meta[4]:reason='unresolved-stat-effect-model-choice'
        else:
            atk,defense=int(meta[0]),int(meta[1])
            if code==STREL:
                atk=3000 if zone==3 else 0 if zone in [1,2,4,5,6,7] else -1
                source='projected-strelitzia-central-effect'
            if code in [REMIX,LOUD] and public_tuner:
                atk+=1500 if code==REMIX else 3300;source='projected-public-opponent-tuner-effect'
            if atk<0 or defense<0:reason='unresolved-variable-stats-model-choice'
            else:
                target=1 if atk>defense else 4
                chosen=positions.index(target);reason='higher-atk' if target==1 else 'higher-def' if defense>atk else 'equal-stats-def-to-avoid-battle-damage'
        self.events.append(dict(code=code,atk=atk,defense=defense,stats_source=source,turn=turn,own_turn=own_turn,
            phase_after_battle=phase_after_battle,attack_intent=attack_intent,raw_position=positions[raw],chosen_position=positions[chosen],
            legal_positions=list(positions),reason=reason,changed=chosen!=raw))
        return chosen

class PositionPolicy:
    def __init__(self,placement_policy,id_codes,data=None):
        self.base=placement_policy;self.id_codes=id_codes;self.positions=PositionDecision(data or load_card_data());self.attack_intent=None
    def set_attack_intent(self,value):
        assert value in [None,True,False];self.attack_intent=value
    def reset(self):self.base.reset();self.positions.reset();self.attack_intent=None
    def act(self,obs,info):
        raw=self.base.act(obs,info);actions=np.asarray(obs['actions_'])[0]
        if int(actions[raw,3])!=5:return raw
        n=int(info['num_options'][0]);code=self.id_codes.get(int(actions[raw,1])*256+int(actions[raw,2]),0)
        cards=np.asarray(obs['cards_'])[0];g=np.asarray(obs['global_'])[0]
        public_tuner=any((int(c[0])*256+int(c[1])) and int(c[4])==1 and
            (int(c[2])==5 or int(c[2])==3 and int(c[5]) in [1,4,5]) and bool(c[27]) for c in cards)
        zone=None
        if getattr(self.base,'memory',None) and self.base.memory.events:
            last=self.base.memory.events[-1]
            if last['incoming_code']==code:zone=last['chosen_place']
        return self.positions.choose(code,actions[:n,8].astype(int).tolist(),raw,int(g[4]),bool(g[7]),int(g[5]) in [8,9],public_tuner,zone,self.attack_intent)

class PositionTeacher:
    def __init__(self,teacher,data):self.base=teacher;self.positions=PositionDecision(data);self.data=data
    def __getattr__(self,name):return getattr(self.base,name)
    def __setattr__(self,name,value):
        if name in ['base','positions','data'] or 'base' not in self.__dict__:object.__setattr__(self,name,value)
        else:setattr(self.base,name,value)
    def __call__(self,s):
        raw=self.base(s)
        if s['message']!='select_position':return raw
        code=s['actions'][raw]['code'];owner=s['to_play']
        # Never consult hidden opponent hand/deck/face-down card identities.
        tuner=any(c['controller']!=owner and (c['location']==16 or c['location']==4 and c['position']&5)
            and bool(self.data.get(str(c['code']),[0,0,0,0])[2]&0x1000) for c in s['cards'])
        zone=None
        if getattr(self.base,'memory',None) and self.base.memory.events:
            last=self.base.memory.events[-1]
            if last['incoming_code']==code:zone=last['chosen_place']
        return self.positions.choose(code,[a['position'] for a in s['actions']],raw,s['turn_count'],s['turn_player']==owner,s['phase'] in [256,512],tuner,zone)
