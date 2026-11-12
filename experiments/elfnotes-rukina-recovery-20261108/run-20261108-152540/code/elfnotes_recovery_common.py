"""Pinned engine, model architecture and masked inference for opponent BC."""
import os,sys,json,hashlib
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).parent))
import rukina_recovery_witness as v
import numpy as np
import jax,jax.numpy as jnp,flax.linen as nn,flax.serialization
from dataclasses import asdict
from ygoai.rl.jax.agent import RNNAgent,ModelArgs
P=v.P;B=Path((P/'experiments/elfnotes-rukina-recovery-20261108/latest-run.txt').read_text().strip())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def features(info):
    length=np.asarray(info['public_chain_length'],np.int32)
    owner=np.asarray(info['public_chain_player'],np.int32);player=np.asarray(info['to_play'],np.int32)
    role=np.where(length>0,np.where(owner==player,1,2),0)
    return np.stack([length,role,np.asarray(info['public_chain_card_id'],np.int32),np.asarray(info['public_chain_effect'],np.int32)],axis=-1)
class ChainHead(nn.Module):
    @nn.compact
    def __call__(self,chain,actions):
        ids=actions[...,1].astype(jnp.int32)*256+actions[...,2].astype(jnp.int32)
        ae=nn.Embed(1000,8,name='action_card')(jnp.clip(ids,0,999))
        ce=nn.Embed(1000,8,name='chain_card')(jnp.clip(chain[:,2],0,999))
        role=jax.nn.one_hot(jnp.clip(chain[:,1],0,2),3)
        effect=jax.nn.one_hot(jnp.clip(chain[:,3],0,15),16)
        context=jnp.concatenate([ce,role,effect,jnp.minimum(chain[:,:1],16)/16.0],axis=-1)
        context=jnp.broadcast_to(context[:,None,:],(*actions.shape[:2],context.shape[-1]))
        act=jax.nn.one_hot(jnp.clip(actions[...,4].astype(jnp.int32),0,9),10)
        x=jnp.concatenate([ae,act,context],axis=-1)
        x=nn.tanh(nn.Dense(64,name='hidden')(x))
        delta=nn.Dense(1,kernel_init=nn.initializers.zeros_init(),bias_init=nn.initializers.zeros_init(),name='output')(x)[...,0]
        return delta*(chain[:,0:1]>0)
AGENT=RNNAgent(**asdict(ModelArgs()),embedding_shape=None);HEAD=ChainHead()
def init_head():return HEAD.init(jax.random.PRNGKey(1198),jnp.zeros((1,4),jnp.int32),jnp.zeros((1,128,12),jnp.uint8))
def load(path):return jax.device_put(flax.serialization.msgpack_restore(Path(path).read_bytes()))
class Policy:
    def __init__(self,base_path,head_path=None):
        self.base=load(base_path);self.head=load(head_path) if head_path else init_head()
        @jax.jit
        def infer(base,head,state,obs,chain):
            result=AGENT.apply(base,obs,state)
            return result[0],result[1]+HEAD.apply(head,chain,obs['actions_'])
        self.infer=infer;self.reset()
    def reset(self):self.state=jax.device_put(AGENT.init_rnn_state(1))
    def act(self,obs,info):
        self.state,logits=self.infer(self.base,self.head,self.state,obs,features(info))
        return int(np.asarray(logits)[0].argmax())
