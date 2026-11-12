"""Adapt the frozen, already exercised trainer to the new source model and corpus."""
from pathlib import Path
import hashlib,json,shutil
P=Path('/home/hatk8/projects/yugioh-ai');B=P/'experiments/chain-imitation-20261108';S=P/'experiments/endboard-20261108'
source=B/'reproduction/train_chain_imitation.py';t=source.read_text()
def replace(a,b):
    global t
    assert t.count(a)==1,(a,t.count(a));t=t.replace(a,b)
replace("D=B/'dataset';OUT=B/'training';OUT.mkdir(exist_ok=False)","""S=P/'experiments/endboard-20261108'
SOURCE_RECORD=json.loads((B/'training/result.json').read_text())
SOURCE_SELECTED=Path(SOURCE_RECORD['selected_checkpoint'])
SOURCE_BASE=SOURCE_SELECTED/'base.flax_model';SOURCE_HEAD=SOURCE_SELECTED/'chain-head.flax_model'
D=S/'dataset';OUT=S/'training';OUT.mkdir(exist_ok=False)""")
replace("base=load_base();head=init_head();initial_base_sha=sha(original_path())","base=load_base(SOURCE_BASE);head=jax.device_put(flax.serialization.msgpack_restore(SOURCE_HEAD.read_bytes()));initial_base_sha=sha(SOURCE_BASE)")
replace("assert np.all(np.asarray(HEAD.apply(head,dummy,jnp.zeros((2,128,12),jnp.uint8)))==0)","assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(head))")
replace("assert sha(original_path())==initial_base_sha","assert sha(SOURCE_BASE)==initial_base_sha")
replace("Path(original_path()).read_bytes()","SOURCE_BASE.read_bytes()")
replace("source_model=original_path(),source_model_sha256=initial_base_sha,","source_model=str(SOURCE_BASE),source_model_sha256=initial_base_sha,source_head=str(SOURCE_HEAD),source_head_sha256=sha(SOURCE_HEAD),")
replace("zero_initialized_head_was_neutral=True","source_head_preserved=True")
replace("unroll(base,head,batch,False)","unroll(base,head,batch,True)")
code=S/'reproduction';code.mkdir(exist_ok=True)
for name in ['chain_policy.py','chain_imitation_common.py','inspect_full_turn.py','endboard_witness.py','build_endboard_dataset.py','run_endboard_inspection.py','prepare_endboard_training.py']:
    shutil.copy2(Path(__file__).parent/name,code/name)
path=code/'train_endboard.py';path.write_text(t)
(S/'train-driver-provenance.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    generated=str(path),generated_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    changes=['New corpus/output directory','Resume previous BC base and head','KL reference is the resumed policy including its learned head','Remove zero-head initialization check; preserve loaded head']),indent=2)+'\n')
print(path)
