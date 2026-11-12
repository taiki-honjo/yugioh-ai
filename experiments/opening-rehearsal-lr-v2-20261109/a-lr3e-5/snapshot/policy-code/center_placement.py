"""A placement preference over legal actions and public/own-visible information only.

This is an external placement adapter, not newly trained neural-network weights.
No goals, teacher plans, hidden deck contents, or benchmark snapshots are inputs.

v2 (2026-11-09, placement-remix-fix-20261109): Kewl Tune Remix tracking only.
v1 started collecting Remix picks on ANY Activate-labelled choice of Remix, including
the yes/no prompt for the optional Synchro Summon after Remix resolves. The Synchro
monster and its materials were then recorded as Remix picks and the third pick raised
ValueError (list.remove). v2 collects picks only after a real activation choice
(idle/chain/battle command), stops on confirmation prompts, when the public chain has
ended, and when a pick is not one of the two recorded cards; such a pick then follows
the unchanged generic rule and the reason is kept in `diagnostics`. All other
behavior is identical to v1.
"""
import numpy as np
REG=56651978;STREL=42302563;PAR=64491754;RHAP=24092792;REMIX=88170262
ELF={13597785,59581480,85976588,REG}
ACTIVATION_MESSAGES=(1,2,8)  # select_idlecmd, select_chain, select_battlecmd: a real activation choice
ACT={'None':0,'Set':1,'Repo':2,'SpSummon':3,'Summon':4,'MSet':5,'Attack':6,'DirectAttack':7,'Activate':8,'Cancel':9}
MSG={'select_idlecmd':1,'select_chain':2,'select_card':3,'select_tribute':4,'select_position':5,'select_effectyn':6,
     'select_yesno':7,'select_battlecmd':8,'select_unselect_card':9,'select_option':10,'select_place':11,'select_sum':12,'select_disfield':13}

class PlacementMemory:
    def __init__(self):self.reset()
    def reset(self):self.incoming=0;self.locked=False;self.events=[];self.remix_targets=None;self.diagnostics=[]
    def end_remix(self,reason,code=0):
        if self.remix_targets is not None:
            self.diagnostics.append(dict(event='remix-tracking-ended',reason=reason,code=code,picks=list(self.remix_targets)))
        self.remix_targets=None
    def observe(self,msg,act,code,hand=False,chain_length=None):
        """chain_length: public chain size at this choice; None when unknown (teacher demonstrations)."""
        if msg==11:self.incoming=0;self.locked=False;self.remix_targets=None;return
        remix_activation=act==8 and code==REMIX and msg in ACTIVATION_MESSAGES
        if chain_length==0 and not remix_activation:
            # The chain that held Remix is over (resolved or negated) without both picks.
            self.end_remix('chain-ended',code)
        if act in [3,4,5]:self.incoming=code;self.locked=True
        elif act==8 and code:
            self.incoming=REG if code==REG and hand else 0
            self.locked=bool(self.incoming)
            if remix_activation:self.remix_targets=[]
            else:self.end_remix('remix-confirmation-prompt' if code==REMIX else 'other-activation',code)
        elif msg in [3,9] and code and not self.locked:
            if self.remix_targets is not None:
                if len(self.remix_targets)<2:self.remix_targets.append(code);self.incoming=0
                elif code in self.remix_targets:
                    remaining=list(self.remix_targets);remaining.remove(code)
                    self.incoming=remaining[0];self.locked=True
                else:
                    self.end_remix('pick-outside-remix-picks',code)
                    self.incoming=code
            else:self.incoming=code
        elif msg==1 and act==9:self.incoming=0;self.locked=False
    def choose(self,places,raw,scores=None,strel_material=False,public_protection=False,level_target=False):
        """Only own monster placements; spell placement and zone disabling never enter here."""
        alternatives=[i for i,p in enumerate(places) if p in [1,2,4,5,6,7]]
        reason='model-choice';chosen=raw
        if places[raw]==3:
            if not alternatives:reason='only-legal-monster-zone'
            elif self.incoming==REG:reason='regina-center-trigger'
            elif level_target:reason='spirit-central-level-increase-target'
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
            spirit_available=any(code==12375297 and loc in [2,3,5] and owner==0 for code,loc,owner,pos in visible)
            baronne_available=any(code==84815190 and loc==7 and owner==0 for code,loc,owner,pos in visible)
            level_target=self.memory.incoming in ELF and spirit_available and baronne_available
            idx=self.memory.choose(actions[:n,10].astype(int).tolist(),raw,scores,public_protection=meaningful_protection,level_target=level_target)
        row=actions[idx];code=self.id_codes.get(int(row[1])*256+int(row[2]),0);spec=int(row[0])
        hand=bool(spec and spec<=len(cards) and int(cards[spec-1,2])==2 and int(cards[spec-1,4])==0)
        self.memory.observe(msg,int(row[4]),code,hand,int(np.asarray(info['public_chain_length'])[0]))
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
            level_target=self.memory.incoming in ELF and any(x.get('act')=='Activate' and x.get('code')==12375297 for x in self.base.steps)
            idx=self.memory.choose([x['place'] for x in s['actions']],idx,strel_material=material,level_target=level_target)
            a=s['actions'][idx]
        self.memory.observe(MSG.get(msg,0),ACT[a['act']],a['code'],a['spec'].startswith('h'))
        return idx

def install_teacher_preferences():
    """Install corrected demonstration choices in the isolated current runtime, once."""
    import all_opening_witness as witness
    if getattr(witness,'center_preferences_installed',False):return
    original_class=witness.v.Teacher;original_factory=witness.teacher
    witness.v.Teacher=lambda *a,**kw:PlacementTeacher(original_class(*a,**kw))
    def factory(case):
        if case['kind']=='elf':return witness.v.Teacher(case,witness.v.Interruption('pass'))
        return PlacementTeacher(original_factory(case))
    witness.teacher=factory;witness.center_preferences_installed=True
