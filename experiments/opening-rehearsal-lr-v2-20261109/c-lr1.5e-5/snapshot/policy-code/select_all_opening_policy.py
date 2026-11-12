"""Select only on held-out validation rollouts, then freeze the candidate before final holdout."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import flax.serialization
P=Path('/home/hatk8/projects/yugioh-ai');B=Path((P/'experiments/all-openings-bar-alternative-20261108/latest-run.txt').read_text().strip())
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
assert not (B/'evaluation/holdout/after').exists(),'Candidate must be frozen before final holdout.'
m=read(B/'manifest.json');D=B/'training/ElfnotesKewlTune';r=read(D/'result.json');assert r['status']=='complete'
preliminary=D/'preliminary-ce-result.json';assert not preliminary.exists()
history=read(D/'history.json');rows=[]
spec=m['decks']['ElfnotesKewlTune']
candidates=[(0,Path(spec['source_model']).parent,r['initial_metrics']['validation'][0],B/'evaluation/validation/step-0200/summary.json')]
candidates +=[(h['step'],D/f"step-{h['step']:04d}",h['validation_ce'],B/f"evaluation/validation/cpu-step-{h['step']:04d}/summary.json") for h in history]
for step,checkpoint,ce,path in candidates:
    score=read(path);assert score['total']==66 and score['lua_errors']==0 and not score['model_received_goals_or_plans']
    assert len(score['families'])==22 and all(f['total']==3 for f in score['families'].values())
    expected=str(checkpoint) if step else str(checkpoint)
    assert score['checkpoint']==expected,(score['checkpoint'],expected)
    rows.append(dict(step=step,checkpoint=str(checkpoint),validation_completed=score['passed'],validation_total=66,
        validation_ce=ce,hands_per_context=3,evaluation=str(path),evaluation_sha256=sha(path),
        legacy_scope_text_typo=score.get('scope','').startswith('Four')))
best=min(rows,key=lambda a:(-a['validation_completed'],a['validation_ce'],a['step']))
selected=Path(best['checkpoint'])
write(preliminary,r)
def leaves(x):
    if isinstance(x,dict):return [a for k in sorted(x) for a in leaves(x[k])]
    if isinstance(x,(list,tuple)):return [a for v in x for a in leaves(v)]
    return [np.asarray(x)]
def changed(a,b):
    aa=leaves(flax.serialization.msgpack_restore(a.read_bytes()));bb=leaves(flax.serialization.msgpack_restore(b.read_bytes()))
    assert len(aa)==len(bb) and all(x.shape==y.shape for x,y in zip(aa,bb))
    return sum(int(np.count_nonzero(x!=y)) for x,y in zip(aa,bb))
original=Path(m['original_source']['source_model']);assert sha(original)==m['original_source']['source_model_sha256']
r['training_changed_base_values_at_ce_candidate']=r['changed_base_values']
r.update(selected_checkpoint=str(selected),selected_step=best['step'],selected_validation_ce=best['validation_ce'],
    selected_validation_completed=best['validation_completed'],selected_validation_total=66,
    changed_base_values=changed(selected/'base.flax_model',Path(spec['source_model'])),
    changed_base_values_from_original_before_task=changed(selected/'base.flax_model',original),
    selection_metric='Maximum completed validation openings; tie-break minimum validation acceptable-action CE, then earlier checkpoint.',
    selection_finalized_before_holdout=True,holdout_used_for_training_or_selection=False,
    files={n:dict(path=str(selected/n),sha256=sha(selected/n),bytes=(selected/n).stat().st_size) for n in ['base.flax_model','chain-head.flax_model']})
r['plan']['checkpoint_selection']=r['selection_metric']
assert r['changed_base_values_from_original_before_task']>0
selection=dict(selected=best,candidates=rows,method=r['selection_metric'],finalized_unix=time.time(),holdout_used=False,
    weights={k:v['sha256'] for k,v in r['files'].items()})
write(D/'validation-selection.json',selection);write(D/'result.json',r)
m['training_plan']['checkpoint_selection']=r['selection_metric'];m['final_selection']=selection;write(B/'manifest.json',m)
print(json.dumps(dict(selected_step=r['selected_step'],validation_completed=best['validation_completed'],total=66,selected_checkpoint=str(selected))))
