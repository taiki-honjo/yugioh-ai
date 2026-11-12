"""Research-only current-recipe first-turn Cue and Rukina+Cue witnesses. No model training."""
import json,gc
import rukina_recovery_witness as v
a=v.a;c=v.choose
CUE=16387555;RECO=89392810;ROTARY=17209452;MIX=16509007;TRACK=42781164;REMIX=88170262;JUKE=14442329;SYNC=78058681
def tune_plan():
    return [a('Summon',CUE),a('Activate',CUE),c(RECO,4),a('Activate',RECO),c(ROTARY),a('SpSummon',TRACK),
        a('Activate',CUE),a('Activate',ROTARY),a('Activate',TRACK),c(SYNC),c(JUKE),
        a('Activate',JUKE),a('Activate',JUKE),c(RECO,16),c(MIX,4),a('Activate',MIX),c(RECO),a('SpSummon',REMIX),a('Set',SYNC)]
def elf_plan():
    return [a('SpSummon',v.RUK),a('Activate',v.RUK),c(v.REG),a('Activate',v.REG),c(v.RUK,16),c(v.SPIRIT,4),a('Activate',v.SPIRIT),a('SpSummon',v.BARONNE),c(v.DINA),
        a('SpSummon',v.DINA),a('Activate',v.DINA),c(v.PAR,8),a('Activate',v.PAR),c(v.DINA,16),c(v.FORT,4),a('Activate',v.FORT),c(v.RHAP,8),
        a('Activate',v.RHAP),c(v.REG,16),c(v.SPIRIT,4),a('SpSummon',v.STREL),a('Activate',v.STREL),c(v.SPIRIT,4),a('SpSummon',v.CRYSTAL)]
def reco_plan():
    return [a('Summon',RECO),a('Activate',RECO),c(ROTARY),a('SpSummon',TRACK),a('Activate',ROTARY),a('Activate',TRACK),c(JUKE),c(SYNC),
        a('Activate',JUKE),a('Activate',JUKE),c(TRACK,16),c(MIX,4),a('Activate',MIX),c(CUE),a('Summon',CUE),a('Activate',CUE),c(TRACK,4),
        a('SpSummon',REMIX),a('Activate',CUE),a('Set',SYNC)]
def juke_plan():
    return [a('Activate',JUKE),a('Summon',v.VEILER),a('Activate',JUKE),c(v.VEILER,16),c(RECO,4),a('Activate',RECO),c(ROTARY),a('SpSummon',TRACK),
        a('Activate',ROTARY),a('Activate',TRACK),c(MIX),c(SYNC),a('Summon',MIX),a('Activate',MIX),c(RECO),a('SpSummon',REMIX),a('Set',SYNC)]
class Teacher(v.Teacher):
    def __call__(self,s):
        if s['message']=='select_yesno' and self.steps and self.steps[0]['code'] in [JUKE,SYNC]:
            return next(x['index'] for x in s['actions'] if x['act']!='Cancel')
        if self.context in [TRACK,REMIX] and s['message'] in ['select_card','select_unselect_card','select_sum']:
            required=self.case['tune_materials'][str(self.context)]
            for code,loc in required:
                found=next((x for x in s['actions'] if x['code']==code and x['spec'].startswith(loc) and code not in self.materials),None)
                if found:self.materials.add(code);return found['index']
            finish=next((x for x in s['actions'] if x['finish'] or x['act']=='Cancel'),None)
            if finish:return finish['index']
        # Cue's two publicly revealed opposing top cards: choose the first legal option, not private fixture data.
        if s['message']=='select_card' and s['actions'] and all(x['spec'].startswith('o') for x in s['actions']):return 0
        if s['message']=='select_option':return next((x['index'] for x in s['actions'] if x['effect']==1152),0)
        return super().__call__(s)
