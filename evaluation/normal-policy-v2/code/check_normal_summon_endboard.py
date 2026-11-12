import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from endboard_witness import *
case=definition('dina',c.FILL,'pass',36108,'legal-tribute-summon-endboard')
case['plan']=list(case['plan'])+[c.action('Summon',c.FORT),c.choose(c.REG,16)]
d=run_reference(case);d.pop('examples')
assert d['normal_used']==1 and d['goal']['passed'] and d['goal']['normal_summons_scored'] is False
path=P/'evaluation/normal-policy-v2/normal-summon-endboard-control.json';write(path,d)
print('LEGAL_NORMAL_SUMMON_ENDBOARD_PASSED',d['goal'],flush=True)
