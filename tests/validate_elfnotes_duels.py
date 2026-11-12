import json
import os
from pathlib import Path
import sys
import time
r=Path('/home/hatk8/projects/yugioh-ai/experiments/elfnotes-20261106/runtime')
sys.path[:0]=[str(r/'agent'),str(r/'agent/ygoenv')]
os.chdir(r/'agent/scripts')
import numpy as np
import ygoenv
from ygoai.utils import init_ygopro
from ygoenv.ygopro.ygopro_ygoenv import get_lua_error_count
deck=init_ygopro('YGOPro-v1','english','/home/hatk8/projects/yugioh-ai/assets/deck/ElfnotesKewlTune.ydk','code_list.txt')
rows=[]; peak=0; own_search_prompts=0; opponent_disclosure_prompts=0; start=time.perf_counter()
for seed in [2011,2012,2013,2014]:
    env=ygoenv.make('YGOPro-v1',env_type='gymnasium',num_envs=8,num_threads=4,seed=seed,
                   deck1=deck,deck2=deck,player=-1,play_mode='self',verbose=False,
                   max_options=128,n_history_actions=32,async_reset=False,oppo_info=False,
                   greedy_reward=False,max_steps=4000,max_episode_steps=4001,timeout=10)
    obs,info=env.reset()
    # At the initial prompt no deck order or opponent hand has been disclosed.
    f=obs['cards_']
    private=(f[:,:,2]==1) | ((f[:,:,2]==2)&(f[:,:,4]==1))
    assert np.all(f[private,0:2]==0), 'Hidden card IDs exposed'
    assert np.all(f[private,7:]==0), 'Hidden card attributes exposed'
    own_hand=(f[:,:,2]==2)&(f[:,:,4]==0)
    assert np.all(np.any(f[own_hand,0:2]!=0,axis=1)), 'Own hand card IDs missing'
    rng=np.random.default_rng(seed+1)
    collected=np.zeros(8,dtype=bool)
    lengths=np.zeros(8,dtype=int)
    for step in range(4000):
        card_rows=obs['cards_']
        known=np.any(card_rows[:,:,0:2]!=0,axis=2)
        own_search_prompts+=int(np.any(known&(card_rows[:,:,2]==1)&(card_rows[:,:,4]==0)))
        opponent_disclosure_prompts+=int(np.any(known&(card_rows[:,:,2]==1)&(card_rows[:,:,4]==1)))
        peak=max(peak,int(np.max(info['raw_num_options'])))
        assert int(np.max(info['raw_num_options']))<=128, 'Action space clipped'
        actions=np.array([rng.integers(int(n)) for n in info['num_options']],dtype=np.int32)
        obs,reward,term,trunc,info=env.step(actions)
        lengths+=~collected
        assert get_lua_error_count()==0, 'Lua error'
        for i in np.flatnonzero((term|trunc)&~collected):
            rows.append(dict(seed=seed,slot=int(i),steps=int(lengths[i]),
                             reason=int(info['termination_reason'][i]),reward=float(reward[i]),
                             terminated=bool(term[i]),truncated=bool(trunc[i])))
            collected[i]=True
        if np.all(collected): break
    assert np.all(collected)
    env.close()
    print('Completed seed',seed,flush=True)
result=dict(duels=len(rows),natural=sum(e['reason']==0 for e in rows),
            step_budget=sum(e['reason']==1 for e in rows),timeouts=sum(e['reason']==2 for e in rows),
            lua_errors=get_lua_error_count(),peak_legal_options=peak,
            own_deck_search_prompts=own_search_prompts,
            opponent_deck_disclosure_prompts=opponent_disclosure_prompts,
            initial_hidden_information_checks_passed=True,seconds=time.perf_counter()-start,episodes=rows)
assert own_search_prompts>0 and opponent_disclosure_prompts>0
(r.parent/'duel-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
