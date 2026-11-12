import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from chain_imitation_common import *
import jax,jax.numpy as jnp,flax.linen as nn,flax.serialization
from dataclasses import asdict
from ygoai.rl.jax.agent import RNNAgent,ModelArgs
class ChainHead(nn.Module):
    @nn.compact
    def __call__(self,chain,actions):
        action_id=actions[...,1].astype(jnp.int32)*256+actions[...,2].astype(jnp.int32)
        ae=nn.Embed(1000,8,name='action_card')(jnp.clip(action_id,0,999))
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
AGENT=RNNAgent(**asdict(ModelArgs()),embedding_shape=None)
HEAD=ChainHead()
def original_path():
    return json.loads((P/'experiments/elfnotes-20261106/expansion-revised-20261106-205114/additional-training-result.json').read_text())['trained_checkpoint']
def load_base(path=None):return jax.device_put(flax.serialization.msgpack_restore(Path(path or original_path()).read_bytes()))
def init_head():return HEAD.init(jax.random.PRNGKey(8118),jnp.zeros((1,4),jnp.int32),jnp.zeros((1,128,12),jnp.uint8))
class Policy:
    def __init__(self,base_path=None,head_path=None,use_head=True):
        self.base=load_base(base_path);self.head=flax.serialization.msgpack_restore(Path(head_path).read_bytes()) if head_path else init_head()
        self.head=jax.device_put(self.head);self.use_head=bool(use_head and head_path is not None)
        enabled=self.use_head
        @jax.jit
        def infer(base,head,state,obs,chain):
            output=AGENT.apply(base,obs,state);logits=output[1]
            delta=HEAD.apply(head,chain,obs['actions_']) if enabled else jnp.zeros_like(logits)
            return output[0],logits+delta,logits,delta
        self.infer=infer;self.reset()
    def reset(self):self.state=jax.device_put(AGENT.init_rnn_state(1))
    def observe(self,obs,info):
        self.state,self.logits,self.base_logits,self.delta=self.infer(self.base,self.head,self.state,obs,chain_features(info))
    def act(self,obs,info):self.observe(obs,info);return int(np.asarray(self.logits)[0].argmax())
