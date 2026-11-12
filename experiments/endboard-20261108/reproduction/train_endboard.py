import sys,json,time,threading,subprocess,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from chain_policy import *
import optax
S=P/'experiments/endboard-20261108'
SOURCE_RECORD=json.loads((B/'training/result.json').read_text())
SOURCE_SELECTED=Path(SOURCE_RECORD['selected_checkpoint'])
SOURCE_BASE=SOURCE_SELECTED/'base.flax_model';SOURCE_HEAD=SOURCE_SELECTED/'chain-head.flax_model'
D=S/'dataset';OUT=S/'training';OUT.mkdir(exist_ok=False)
manifest=json.loads((D/'split-manifest.json').read_text());assert manifest['status']=='complete'
plan=manifest['training_plan'];start=time.perf_counter()
train={k:v for k,v in np.load(D/'train.npz').items()};validation={k:v for k,v in np.load(D/'validation.npz').items()}
base=load_base(SOURCE_BASE);head=jax.device_put(flax.serialization.msgpack_restore(SOURCE_HEAD.read_bytes()));initial_base_sha=sha(SOURCE_BASE)
dummy=jnp.array([[1,1,14,0],[1,2,14,0]],jnp.int32)
assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(head))
params=dict(base=base,head=head)
def time_major(data,indices):
    return {k:jnp.swapaxes(jnp.asarray(v[indices]),0,1) for k,v in data.items() if k!='lengths'}
def unroll(base,head,batch,use_head=True):
    batch_size=batch['labels'].shape[1];initial=AGENT.init_rnn_state(batch_size)
    def step(state,values):
        cards,glob,actions,history,chain=values
        obs=dict(cards_=cards,global_=glob,actions_=actions,h_actions_=history,mask_=None)
        output=AGENT.apply(base,obs,state)
        logits=output[1]+(HEAD.apply(head,chain,actions) if use_head else 0)
        return output[0],logits
    return jax.lax.scan(step,initial,(batch['cards_'],batch['global_'],batch['actions_'],batch['h_actions_'],batch['chain']))[1]
old_probs=jax.jit(lambda base,head,batch:jax.nn.softmax(unroll(base,head,batch,True),axis=-1))
for label,data in [('train',train),('validation',validation)]:
    probabilities=[]
    for i in range(0,len(data['labels']),4):
        indices=np.arange(i,min(i+4,len(data['labels'])))
        if len(indices)<4:indices=np.pad(indices,(0,4-len(indices)),mode='edge')
        probs=np.asarray(old_probs(base,head,time_major(data,indices)))
        probabilities.append(np.swapaxes(probs,0,1)[:min(4,len(data['labels'])-i)])
    data['old_probs']=np.concatenate(probabilities,axis=0)
    print('OLD_POLICY_DISTRIBUTIONS',label,len(data['labels']),flush=True)
tx_base=optax.adam(plan['base_learning_rate']);tx_head=optax.adam(plan['head_learning_rate'])
state_base=tx_base.init(base);state_head=tx_head.init(head)
def loss_fn(params,batch):
    logits=unroll(params['base'],params['head'],batch)
    logp=jax.nn.log_softmax(logits,axis=-1)
    ce=-jnp.take_along_axis(logp,batch['labels'][...,None],axis=-1)[...,0]
    valid=batch['actions_'][...,3]!=0
    safe_logp=jnp.where(valid,logp,0)
    old=batch['old_probs'];kl=jnp.sum(old*(jnp.log(jnp.maximum(old,1e-8))-safe_logp),axis=-1)
    weight=batch['weights'];denom=jnp.maximum(weight.sum(),1)
    ce_loss=(ce*weight).sum()/denom;kl_loss=(kl*weight).sum()/denom
    accuracy=((logits.argmax(-1)==batch['labels'])*weight).sum()/denom
    return ce_loss+plan['kl_weight']*kl_loss,(ce_loss,kl_loss,accuracy)
@jax.jit
def update(params,st_base,st_head,batch):
    (loss,metrics),grads=jax.value_and_grad(loss_fn,has_aux=True)(params,batch)
    norm=optax.global_norm(grads);scale=jnp.minimum(1.0,1.0/(norm+1e-8))
    grads=jax.tree.map(lambda x:x*scale,grads)
    ub,st_base=tx_base.update(grads['base'],st_base,params['base'])
    uh,st_head=tx_head.update(grads['head'],st_head,params['head'])
    params=dict(base=optax.apply_updates(params['base'],ub),head=optax.apply_updates(params['head'],uh))
    return params,st_base,st_head,loss,metrics,norm
evaluate=jax.jit(loss_fn)
def validation_metrics(params):
    metrics=[];weights=[]
    for i in range(0,len(validation['labels']),4):
        indices=np.arange(i,min(i+4,len(validation['labels'])))
        batch=time_major(validation,indices)
        _,m=evaluate(params,batch);metrics.append(np.asarray(m));weights.append(float(batch['weights'].sum()))
    return np.average(metrics,axis=0,weights=weights).tolist()
stop=threading.Event();gpu=[]
def monitor():
    with (OUT/'gpu.csv').open('w') as h:
        h.write('elapsed_seconds,memory_used_mib,utilization_percent\n')
        while not stop.is_set():
            d=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            if d.returncode==0:
                row=[time.perf_counter()-start,*map(float,d.stdout.strip().splitlines()[0].split(','))]
                gpu.append(row);h.write(','.join(map(str,row))+'\n');h.flush()
            stop.wait(2)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
rng=np.random.default_rng(plan['seed']);history=[];best=float('inf');best_step=None
try:
    print('DEVICES',jax.devices(),flush=True)
    initial_metrics=validation_metrics(params);print('INITIAL_VALIDATION',initial_metrics,flush=True)
    for step in range(1,plan['updates']+1):
        indices=rng.integers(0,len(train['labels']),size=plan['batch_size'])
        params,state_base,state_head,loss,metrics,norm=update(params,state_base,state_head,time_major(train,indices))
        if step%20==0:
            values=np.asarray([loss,*metrics,norm]);assert np.all(np.isfinite(values))
            print('UPDATE',step,'LOSS',float(loss),'CE_KL_ACC',np.asarray(metrics).tolist(),flush=True)
        if step%100==0:
            vm=validation_metrics(params);entry=dict(step=step,validation_ce=vm[0],validation_kl=vm[1],validation_accuracy=vm[2])
            history.append(entry);checkpoint=OUT/f'step-{step:04d}';checkpoint.mkdir()
            for key,name in [('base','base.flax_model'),('head','chain-head.flax_model')]:
                data=jax.device_get(params[key]);assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(data))
                (checkpoint/name).write_bytes(flax.serialization.to_bytes(data))
            if vm[0]<best:
                best=vm[0];best_step=step
                (OUT/'best-checkpoint.txt').write_text(str(checkpoint)+'\n')
            (OUT/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            print('VALIDATION',json.dumps(entry),'BEST_STEP',best_step,flush=True)
finally:stop.set();thread.join(timeout=12)
assert sha(SOURCE_BASE)==initial_base_sha
selected=Path((OUT/'best-checkpoint.txt').read_text().strip())
new=flax.serialization.msgpack_restore((selected/'base.flax_model').read_bytes());old=flax.serialization.msgpack_restore(SOURCE_BASE.read_bytes())
new_leaves=jax.tree.leaves(new);old_leaves=jax.tree.leaves(old)
changed=sum(int(np.count_nonzero(a!=b)) for a,b in zip(new_leaves,old_leaves));assert changed>0
record=dict(status='complete',source_model=str(SOURCE_BASE),source_model_sha256=initial_base_sha,source_head=str(SOURCE_HEAD),source_head_sha256=sha(SOURCE_HEAD),
            dataset_manifest_sha256=sha(D/'split-manifest.json'),train_sha256=sha(D/'train.npz'),validation_sha256=sha(D/'validation.npz'),
            holdout_used_for_training_or_selection=False,plan=plan,initial_validation=initial_metrics,
            selected_step=best_step,selected_checkpoint=str(selected),selected_validation_ce=best,
            changed_base_values=changed,all_weights_finite=True,source_head_preserved=True,
            elapsed_seconds=time.perf_counter()-start,gpu_devices=list(map(str,jax.devices())),
            peak_total_gpu_memory_mib=max(r[1] for r in gpu) if gpu else None,
            files={p.name:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in selected.iterdir()})
(OUT/'result.json').write_text(json.dumps(record,indent=2)+'\n');print('TRAINING_COMPLETE',json.dumps(record),flush=True)
