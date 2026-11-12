"""Load the adopted paired model plus the adapters of the CURRENT production policy config.

The policy config is the one referenced by config/continuous-elfnotes.json
(`verified_policy_config`), so this entry and continuous_opening_policy.py follow the same
adopted adapter version. Model and adapter files are checked against recorded hashes, and
the adapter modules must come from the configured directory: if another version of a
module is already imported in this process, loading is refused (use a fresh process).
"""
import hashlib,json,sys
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai')
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def policy_config_path():
    return Path(json.loads((P/'config/continuous-elfnotes.json').read_text())['verified_policy_config'])
def _origin(module,root):
    path=Path(module.__file__).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise RuntimeError(f'{module.__name__} was already imported from {path}, not {root}; use a fresh process')
    return path
def create_policy():
    p=policy_config_path()
    cfg=json.loads(p.read_text());checkpoint=Path(cfg['selected_checkpoint']);adapter=cfg.get('placement_adapter',{})
    for n,k in [('base.flax_model','base_sha256'),('chain-head.flax_model','head_sha256')]:
        assert _sha(checkpoint/n)==cfg[k],('Pinned model file changed',n)
    root=adapter['runtime_code_directory'];sys.path.insert(0,root)
    import all_openings_common;_origin(all_openings_common,root)
    from all_openings_common import Policy,v
    policy=Policy(checkpoint/'base.flax_model',checkpoint/'chain-head.flax_model')
    if adapter.get('enabled'):
        import center_placement;path=_origin(center_placement,root)
        if adapter.get('center_placement_sha256') and _sha(path)!=adapter['center_placement_sha256']:
            raise RuntimeError('Placement adapter file differs from the policy config')
        policy=center_placement.PlacementPolicy(policy,v.v.engine.ID_CODES)
    if cfg.get('battle_position_adapter',{}).get('enabled'):
        import battle_position;_origin(battle_position,root)
        policy=battle_position.PositionPolicy(policy,v.v.engine.ID_CODES)
    policy.policy_config=str(p)
    return policy
