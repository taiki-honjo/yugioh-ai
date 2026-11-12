import json,shutil,hashlib
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai');B=P/'experiments/chain-imitation-20261108';S=P/'experiments/endboard-20261108';W=Path(__file__).parent
code=S/'reproduction'
for name in ['chain_counter_cases.py','evaluate_endboard.py','reinspect_endboard.py','prepare_endboard_validation.py','run_ygo_memory_limited.py','run_endboard_validation.py']:
    shutil.copy2(W/name,code/name)
def adapt(source,replacements,dest):
    t=source.read_text()
    for a,b in replacements:
        assert t.count(a)==1,(a,t.count(a));t=t.replace(a,b)
    dest.write_text(t)
    return dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
reg=adapt(B/'reproduction/evaluate_chain_imitation.py',[
    ("OUT=B/'evaluation';OUT.mkdir(exist_ok=False)","S=P/'experiments/endboard-20261108';OUT=S/'short-regression';OUT.mkdir(exist_ok=False)"),
    ("train=json.loads((B/'training/result.json').read_text())","train=json.loads((S/'training/result.json').read_text())"),
    ("assert sha(B/'dataset/split-manifest.json')==train['dataset_manifest_sha256']","assert sha(S/'dataset/split-manifest.json')==train['dataset_manifest_sha256']")
],code/'evaluate_short_regression.py')
battle=adapt(B/'reproduction/eval_chain_battle.py',[
    ("R=B/'runtime'","R=B/'runtime';S=P/'experiments/endboard-20261108'"),
    ("training=json.loads((B/'training/result.json').read_text());selected=Path(training['selected_checkpoint'])", "training=json.loads((S/'training/result.json').read_text());selected=Path(training['selected_checkpoint']);before=Path(json.loads((B/'training/result.json').read_text())['selected_checkpoint'])"),
    ("candidate=original_path() if args.policy=='baseline' else str(selected/'base.flax_model')", "candidate=str((before if args.policy=='baseline' else selected)/'base.flax_model')"),
    ("out=B/'battle'/args.policy/args.deck", "out=S/'battle'/args.policy/args.deck"),
    ("scorer=B/'scoring-evaluator.py'", "scorer=S/'scoring-evaluator.py'"),
    ("head=jax.device_put(flax.serialization.msgpack_restore((selected/'chain-head.flax_model').read_bytes()))", "head_path=(before if args.policy=='baseline' else selected)/'chain-head.flax_model'\nhead=jax.device_put(flax.serialization.msgpack_restore(head_path.read_bytes()))"),
    ("enabled=np.float32(args.policy=='bc_with_chain' and params is candidate_params)","enabled=np.float32(params is candidate_params)"),
    ("base_seed=11037", "base_seed=41037"),
    ("episodes=64,seed=11037", "episodes=64,seed=41037"),
    ("head_sha256=sha(selected/'chain-head.flax_model') if args.policy=='bc_with_chain' else None", "head_sha256=sha(head_path)"),
    ("assert ctx.head_opponent_batches==0 and (args.policy!='bc_with_chain' or ctx.head_candidate_batches>0)", "assert ctx.head_opponent_batches==0 and ctx.head_candidate_batches>0")
],code/'evaluate_battle.py')
(S/'validation-driver-provenance.json').write_text(json.dumps(dict(short_regression=reg,battle=battle,
    battle_baseline='Previous BC model with its previous chain head; not the original RL-only model.',
    battle_seed=41037,episodes_per_case=64,opponents_fixed=True),indent=2)+'\n')
print('VALIDATION_DRIVERS_PREPARED')
