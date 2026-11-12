"""A placement preference over legal actions and public/own-visible information only.

This is an external placement adapter, not newly trained neural-network weights.
No goals, teacher plans, hidden deck contents, or benchmark snapshots are inputs.
"""
import numpy as np
REG=56651978;STREL=42302563;PAR=64491754;RHAP=24092792
ELF={13597785,59581480,85976588,REG}
ACT={'None':0,'Set':1,'Repo':2,'SpSummon':3,'Summon':4,'MSet':5,'Attack':6,'DirectAttack':7,'Activate':8,'Cancel':9}
MSG={'select_idlecmd':1,'select_chain':2,'select_card':3,'select_tribute':4,'select_position':5,'select_effectyn':6,
     'select_yesno':7,'select_battlecmd':8,'select_unselect_card':9,'select_option':10,'select_place':11,'select_sum':12,'select_disfield':13}

class PlacementMemory:
    def __init__(self):self.reset()
    def reset(self):self.incoming=0;self.locked=False;self.events=[]
    def observe(self,msg,act,code,hand=False):
        if msg==11:self.incoming=0;self.locked=False;return
        if act in [3,4,5]:self.incoming=code;self.locked=True
        elif act==8:
            self.incoming=REG if code==REG and hand else 0
            self.locked=bool(self.incoming)
        elif msg in [3,9] and code and not self.locked:self.incoming=code
        elif msg==1 and act==9:self.incoming=0;self.locked=False
    def choose(self,places,raw,scores=None,strel_material=False,public_protection=False):
        """Only own monster placements; spell placement and zone disabling never enter here."""
        alternatives=[i for i,p in enumerate(places) if p in [1,2,4,5,6,7]]
        reason='model-choice';chosen=raw
        if places[raw]==3:
            if not alternatives:reason='only-legal-monster-zone'
            elif self.incoming==REG:reason='regina-center-trigger'
            elif self.incoming==STREL and not strel_material:reason='strelitzia-center-attack'
            elif public_protection:reason='public-protection-without-elf-recovery-reservation'
            elif not self.incoming:reason='unknown-arrival-defer-to-model'
            else:
                chosen=max(alternatives,key=lambda i:float(scores[i])) if scores is not None else alternatives[0]
                reason='reserve-center-for-elfnotes'
        self.events.append(dict(incoming_code=self.incoming,raw_place=places[raw],chosen_place=places[chosen],reason=reason,changed=chosen!=raw))
        return chosen

class PlacementPolicy:
    """Use the existing model for all actions and the next-best legal placement when appropriate."""
    def __init__(self,base_policy,id_codes):self.base=base_policy;self.id_codes=id_codes;self.memory=PlacementMemory()
    def reset(self):self.base.reset();self.memory.reset()
    def act(self,obs,info):
        import all_openings_common as common
        self.base.state,logits=self.base.infer(self.base.base,self.base.head,self.base.state,obs,common.features(info))
        scores=np.asarray(logits)[0];actions=np.asarray(obs['actions_'])[0];cards=np.asarray(obs['cards_'])[0]
        n=int(info['num_options'][0]);raw=int(scores[:n].argmax());msg=int(actions[raw,3]);idx=raw
        if msg==11:
            # Only face-up own spell effects and own known hand/grave are consulted.
            visible=[(self.id_codes.get(int(c[0])*256+int(c[1]),0),int(c[2]),int(c[4]),int(c[5])) for c in cards if int(c[0])*256+int(c[1])]
            protection=any(code in [PAR,RHAP] and loc==4 and owner==0 and pos in [1,4,5] for code,loc,owner,pos in visible)
            reserve=any(code in ELF and loc in [2,5] and owner==0 for code,loc,owner,pos in visible)
            meaningful_protection=protection and not reserve and self.memory.incoming in [50954680,84815190]
            idx=self.memory.choose(actions[:n,10].astype(int).tolist(),raw,scores,public_protection=meaningful_protection)
        row=actions[idx];code=self.id_codes.get(int(row[1])*256+int(row[2]),0);spec=int(row[0])
        hand=bool(spec and spec<=len(cards) and int(cards[spec-1,2])==2 and int(cards[spec-1,4])==0)
        self.memory.observe(msg,int(row[4]),code,hand)
        return idx

class PlacementTeacher:
    """Correct future teacher demonstrations; the teacher may inspect its own remaining plan."""
    def __init__(self,teacher):self.base=teacher;self.memory=PlacementMemory()
    def __getattr__(self,name):return getattr(self.base,name)
    def __setattr__(self,name,value):
        if name in ['base','memory'] or 'base' not in self.__dict__:object.__setattr__(self,name,value)
        else:setattr(self.base,name,value)
    def __call__(self,s):
        idx=self.base(s);a=s['actions'][idx];msg=s['message']
        if msg=='select_place':
            material=any(x.get('act')=='SpSummon' and x.get('code')==50954680 for x in self.base.steps)
            idx=self.memory.choose([x['place'] for x in s['actions']],idx,strel_material=material)
            a=s['actions'][idx]
        self.memory.observe(MSG.get(msg,0),ACT[a['act']],a['code'],a['spec'].startswith('h'))
        return idx
