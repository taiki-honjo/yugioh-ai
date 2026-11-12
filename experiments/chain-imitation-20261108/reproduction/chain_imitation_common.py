"""Shared fixed fixtures plus a learned public-chain action head."""
import importlib.util,sys,json,os,collections,hashlib,gc
from pathlib import Path
import numpy as np
P=Path('/home/hatk8/projects/yugioh-ai');B=P/'experiments/chain-imitation-20261108'
source=P/'experiments/combo-benchmark-20261107/evaluation-v1/runner.py'
helper=B/'combo.py'
if not helper.exists():
    text=source.read_text().replace("B=P/'experiments/combo-benchmark-20261107'","B=P/'experiments/chain-imitation-20261108'")
    helper.write_text(text)
spec=importlib.util.spec_from_file_location('fixed_combo',helper);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
CODE_IDS={int(line.split()[0]):i+1 for i,line in enumerate(c.VOCAB.read_text().splitlines())}
ID_CODES={v:k for k,v in CODE_IDS.items()}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def chain_features(info):
    length=np.asarray(info['public_chain_length'],dtype=np.int32)
    owner=np.asarray(info['public_chain_player'],dtype=np.int32)
    player=np.asarray(info['to_play'],dtype=np.int32)
    role=np.where(length>0,np.where(owner==player,1,2),0)
    return np.stack([length,role,np.asarray(info['public_chain_card_id'],dtype=np.int32),
                     np.asarray(info['public_chain_effect'],dtype=np.int32)],axis=-1)
def close(env):
    env.close();del env
    gc.collect()
def collect_snapshot_checks(obs,info,s):
    acts=np.asarray(obs['actions_'])[0];n=int(info['num_options'][0])
    assert n==len(s['actions']) and int(info['to_play'][0])==s['to_play']
    assert int(info['public_chain_length'][0])==len(s['public_chain'])
    if s['public_chain']:
        public=s['public_chain'][-1]
        assert int(info['public_chain_player'][0])==public['player']
        assert int(info['public_chain_card_id'][0])==CODE_IDS.get(public['code'],0)
        assert int(info['public_chain_effect'][0])==public['description'] & 15
    for a in s['actions']:
        cid=int(acts[a['index'],1])*256+int(acts[a['index'],2])
        assert ID_CODES.get(cid,0)==a['code'],(a,cid)
        if s['message'] in ['select_idlecmd','select_chain','select_effectyn','select_yesno','select_option']:
            expected=['None','Set','Repo','SpSummon','Summon','MSet','Attack','DirectAttack','Activate','Cancel'].index(a['act'])
            assert int(acts[a['index'],4])==expected,(a,acts[a['index']].tolist())
    # IDs in opponent private hand/deck must stay hidden, including with chain metadata enabled.
    cards=np.asarray(obs['cards_'])[0]
    hidden=(cards[:,4]==1)&np.isin(cards[:,2],[1,2])
    assert np.all(cards[hidden,:2]==0)
    return n

