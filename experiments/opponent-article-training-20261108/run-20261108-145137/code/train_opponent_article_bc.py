"""Warm-start one opponent's existing RL weights; select only on validation CE."""
import argparse,json,sys,time,threading,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from opponent_article_bc_common import *
import optax
p=argparse.ArgumentParser();p.add_argument('--deck',required=True,choices=['ClownTroupe','Artmage']);args=p.parse_args()
manifest=json.loads((B/'manifest.json').read_text());assert manifest['status']=='complete'
assert manifest['grading_policy']=='normal-summon-agnostic-v2' and manifest['normal_summon_usage_scored'] is False
spec=manifest['decks'][args.deck];plan=manifest['training_plan'];D=B/'dataset'/args.deck;OUT=B/'training'/args.deck
OUT.mkdir(parents=True,exist_ok=False)
assert sha(spec['source_model'])==spec['source_model_sha256']
assert sha(v.R/'agent/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so')==manifest['runtime']['sha256']
data={split:{k:a.copy() for k,a in np.load(D/(split+'.npz')).items()} for split in ['train','validation']}
assert all(sha(D/(split+'.npz'))==spec[split]['sha256'] for split in data)
base=load(spec['source_model']);head=init_head();params=dict(base=base,head=head)
assert np.all(np.asarray(HEAD.apply(head,jnp.ones((1,4),jnp.int32),jnp.zeros((1,128,12),jnp.uint8)))==0)
def batch_of(d,indices):return {k:jnp.swapaxes(jnp.asarray(a[indices]),0,1) for k,a in d.items() if k!='lengths'}
def unroll(base,head,batch):
    initial=AGENT.init_rnn_state(batch['labels'].shape[1])
    def step(state,values):
        cards,glob,actions,history,chain=values
        obs=dict(cards_=cards,global_=glob,actions_=actions,h_actions_=history,mask_=None)
        result=AGENT.apply(base,obs,state)
        return result[0],result[1]+HEAD.apply(head,chain,actions)
    return jax.lax.scan(step,initial,(batch['cards_'],batch['global_'],batch['actions_'],batch['h_actions_'],batch['chain']))[1]
frozen_probs=jax.jit(lambda b,h,d:jax.nn.softmax(unroll(b,h,d),axis=-1))
start=time.perf_counter()
for split,d in data.items():
    probs=[]
    for i in range(0,len(d['labels']),2):
        idx=np.arange(i,min(i+2,len(d['labels'])))
        probabilities=np.swapaxes(np.asarray(frozen_probs(base,head,batch_of(d,idx))),0,1)
        probs.append(probabilities)
    d['old_probs']=np.concatenate(probs)
    print('FROZEN_DISTRIBUTIONS',args.deck,split,len(d['labels']),flush=True)
tx_b=optax.adam(plan['base_learning_rate']);tx_h=optax.adam(plan['head_learning_rate'])
st_b=tx_b.init(base);st_h=tx_h.init(head)
def loss_fn(params,batch):
    logits=unroll(params['base'],params['head'],batch);logp=jax.nn.log_softmax(logits,axis=-1)
    ce=-jnp.take_along_axis(logp,batch['labels'][...,None],axis=-1)[...,0]
    valid=batch['actions_'][...,3]!=0;safe=jnp.where(valid,logp,0)
    old=batch['old_probs'];kl=jnp.sum(old*(jnp.log(jnp.maximum(old,1e-8))-safe),axis=-1)
    w=batch['weights'];denom=jnp.maximum(w.sum(),1)
    metrics=((ce*w).sum()/denom,(kl*w).sum()/denom,((logits.argmax(-1)==batch['labels'])*w).sum()/denom)
    return metrics[0]+plan['kl_weight']*metrics[1],metrics
@jax.jit
def update(params,sb,sh,batch):
    (loss,metrics),grads=jax.value_and_grad(loss_fn,has_aux=True)(params,batch)
    norm=optax.global_norm(grads);scale=jnp.minimum(1.,1./(norm+1e-8))
    grads=jax.tree.map(lambda g:g*scale,grads)
    ub,sb=tx_b.update(grads['base'],sb,params['base']);uh,sh=tx_h.update(grads['head'],sh,params['head'])
    return dict(base=optax.apply_updates(params['base'],ub),head=optax.apply_updates(params['head'],uh)),sb,sh,metrics,norm
evaluate=jax.jit(loss_fn)
def metrics_on(params,split):
    d=data[split];values=[];weights=[]
    for i in range(0,len(d['labels']),2):
        batch=batch_of(d,np.arange(i,min(i+2,len(d['labels']))))
        _,metrics=evaluate(params,batch);values.append(np.asarray(metrics));weights.append(float(batch['weights'].sum()))
    return np.average(values,axis=0,weights=weights).tolist()
stop=threading.Event();gpu=[]
def monitor():
    with (OUT/'gpu.csv').open('w') as h:
        h.write('elapsed_seconds,memory_mib,utilization_percent\n')
        while not stop.is_set():
            r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            if r.returncode==0:
                sample=[time.perf_counter()-start,*map(float,r.stdout.strip().splitlines()[0].split(','))]
                gpu.append(sample);h.write(','.join(map(str,sample))+'\n');h.flush()
            stop.wait(2)
t=threading.Thread(target=monitor,daemon=True);t.start()
history=[];best=float('inf');selected=None;rng=np.random.default_rng(plan['seed'])
try:
    initial={split:metrics_on(params,split) for split in data}
    print('INITIAL',args.deck,initial,'DEVICES',jax.devices(),flush=True)
    for step in range(1,plan['updates']+1):
        idx=rng.integers(0,len(data['train']['labels']),size=plan['batch_size'])
        params,st_b,st_h,metrics,norm=update(params,st_b,st_h,batch_of(data['train'],idx))
        if step%20==0:
            values=np.asarray([*metrics,norm]);assert np.all(np.isfinite(values))
            print('UPDATE',args.deck,step,values.tolist(),flush=True)
            write(OUT/'progress.json',dict(step=step,total=plan['updates'],metrics=values.tolist(),elapsed_seconds=time.perf_counter()-start))
        if step%100==0:
            vm=metrics_on(params,'validation');entry=dict(step=step,validation_ce=vm[0],validation_kl=vm[1],validation_accuracy=vm[2])
            history.append(entry);checkpoint=OUT/f'step-{step:04d}';checkpoint.mkdir()
            for key,name in [('base','base.flax_model'),('head','chain-head.flax_model')]:
                value=jax.device_get(params[key]);assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(value))
                (checkpoint/name).write_bytes(flax.serialization.to_bytes(value))
            if vm[0]<best:best=vm[0];selected=checkpoint;write(OUT/'selected.json',dict(checkpoint=str(checkpoint),step=step,validation_ce=best))
            write(OUT/'history.json',history);print('VALIDATION',args.deck,entry,flush=True)
finally:stop.set();t.join(timeout=12)
assert sha(spec['source_model'])==spec['source_model_sha256']
new=flax.serialization.msgpack_restore((selected/'base.flax_model').read_bytes())
old=flax.serialization.msgpack_restore(Path(spec['source_model']).read_bytes())
changed=sum(int(np.count_nonzero(a!=b)) for a,b in zip(jax.tree.leaves(new),jax.tree.leaves(old)));assert changed>0
result=dict(status='complete',deck=args.deck,source_model=spec['source_model'],source_sha256=spec['source_model_sha256'],
    source_unchanged=True,selected_checkpoint=str(selected),selected_step=int(selected.name.rsplit('-',1)[1]),
    initial_metrics=initial,selected_validation_ce=best,changed_base_values=changed,all_weights_finite=True,
    plan=plan,optimizer_started_fresh=True,head_started_neutral=True,holdout_used_for_training_or_selection=False,
    normal_summon_usage_scored=False,runtime=manifest['runtime'],elapsed_seconds=time.perf_counter()-start,
    gpu_devices=list(map(str,jax.devices())),peak_total_gpu_memory_mib=max(x[1] for x in gpu) if gpu else None,
    files={p.name:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in selected.iterdir()})
write(OUT/'result.json',result);print('TRAINING_COMPLETE',args.deck,json.dumps(result),flush=True)
