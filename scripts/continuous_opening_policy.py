"""Load the adopted paired policy and its adapters. Does not start any learning."""
from pathlib import Path
import continuous_learning as control

def create_policy(checkpoint=None):
    cfg=control.load_config(Path(__file__).resolve().parent.parent/'config/continuous-elfnotes.json')
    if cfg.get('backend')!='verified_opening':raise RuntimeError('Verified opening backend is not configured')
    control.runtime_check(cfg)
    return control.backend().create_policy(cfg,checkpoint)
