"""Current deck policies. act(obs, info), reset(), and explicit attack intent.

Uses the same pinned, masked native observation format as the verified rollouts.
Historical evaluate.py/eval-meta-opponents.py are intentionally not redirected.
"""
import hashlib,json,sys
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai')
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create_policy(deck,positions=True):
    if deck=='ElfnotesKewlTune':
        if not positions:raise ValueError('Use the frozen Elfnotes baseline for comparisons.')
        sys.path.insert(0,str(P/'scripts'))
        from verified_opening_policy import create_policy as elfnotes
        return elfnotes()
    cfg=json.loads((P/'config/latest-deck-position-policies.json').read_text());entry=cfg['decks'][deck]
    for path,digest in [(entry['base'],entry['base_sha256']),(entry['deck'],entry['deck_sha256']),
                        (cfg['vocabulary'],cfg['vocabulary_sha256']),
                        (Path(cfg['code_directory'])/'battle_position.py',cfg['adapter_sha256']),
                        (Path(cfg['code_directory'])/'position_cards.json',cfg['card_data_sha256']),
                        (Path(cfg['model_code_directory'])/'opponent_article_bc_common.py',cfg['model_code_sha256'])]:
        assert _sha(path)==digest,('Pinned policy file changed',str(path))
    if entry['head']:assert _sha(entry['head'])==entry['head_sha256']
    native=Path(cfg['runtime'])/'agent/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so'
    assert _sha(native)==cfg['native_sha256']
    sys.path.insert(0,cfg['model_code_directory'])
    from opponent_article_bc_common import Policy,v
    ids={i+1:int(line.split()[0]) for i,line in enumerate(Path(cfg['vocabulary']).read_text().splitlines())}
    assert ids==v.ID_CODES,'Loaded native vocabulary differs from registry'
    assert str(v.R)==cfg['runtime'],'Loaded native runtime differs from registry'
    policy=Policy(entry['base'],entry['head'])
    if positions and entry['positions_enabled']:
        # Load by file, so an Elfnotes module of the same name cannot leak in.
        import importlib.util
        spec=importlib.util.spec_from_file_location('all_deck_battle_position',Path(cfg['code_directory'])/'battle_position.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        policy=module.PositionPolicy(policy,ids)
    policy.deck_name=deck;policy.policy_metadata=dict(entry,weights_retrained=False,positions_enabled=positions)
    return policy

def create_teacher(teacher):
    """Wrap an existing legal teacher; no Elfnotes-only placement memory required."""
    import importlib.util
    cfg=json.loads((P/'config/latest-deck-position-policies.json').read_text())
    path=Path(cfg['code_directory'])/'battle_position.py';assert _sha(path)==cfg['adapter_sha256']
    spec=importlib.util.spec_from_file_location('all_deck_position_teacher',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.PositionTeacher(teacher,module.load_card_data())
