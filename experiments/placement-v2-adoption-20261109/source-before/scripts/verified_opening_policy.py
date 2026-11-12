"""Load the paired verified model plus its explicitly configured placement adapter."""
import hashlib,json,sys
from pathlib import Path
def create_policy():
    p=Path('/home/hatk8/projects/yugioh-ai/config/latest-verified-opening-policy.json')
    cfg=json.loads(p.read_text());checkpoint=Path(cfg['selected_checkpoint']);adapter=cfg.get('placement_adapter',{})
    for n,k in [('base.flax_model','base_sha256'),('chain-head.flax_model','head_sha256')]:
        assert hashlib.sha256((checkpoint/n).read_bytes()).hexdigest()==cfg[k]
    root=adapter['runtime_code_directory'];sys.path.insert(0,root)
    from all_openings_common import Policy,v
    policy=Policy(checkpoint/'base.flax_model',checkpoint/'chain-head.flax_model')
    if adapter.get('enabled'):
        from center_placement import PlacementPolicy
        policy=PlacementPolicy(policy,v.v.engine.ID_CODES)
    if cfg.get('battle_position_adapter',{}).get('enabled'):
        from battle_position import PositionPolicy
        policy=PositionPolicy(policy,v.v.engine.ID_CODES)
    return policy
