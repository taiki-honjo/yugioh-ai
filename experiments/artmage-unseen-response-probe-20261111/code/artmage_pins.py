"""Pinned paths and hashes for the isolated Artmage entry (no engine import; safe for registration and tests)."""
import hashlib
from pathlib import Path

P = Path('/home/hatk8/projects/yugioh-ai')
W = P / 'experiments/artmage-parallel-20261110'
MODEL_CODE = P / 'experiments/opponent-article-training-20261108/run-20261108-145203/code'
ADAPTER_DIR = P / 'evaluation/all-deck-position-20261109/code'
RUNTIME = P / 'evaluation/opponent-article-basics-20261108/runtime'
META_RUN = P / 'experiments/meta-opponents-20261107/run-20261107-224631'
NATIVE = RUNTIME / 'agent/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so'
DECK_KEY = 'Artmage'

PINS = {
    'base': (W / 'models/initial/base.flax_model', '12500c7724b2d723983a2963a960d75172a6df51fb55ea4f3acf102eeb34132e'),
    'head': (W / 'models/initial/chain-head.flax_model', '0bc7534cab102cd87586e5d22bd4ad0cd3b55d52d08e284146976506ae6193fc'),
    'deck_copy': (W / 'decks/Artmage.ydk', 'a93c4d30d0945f58f741f1e135fdcbc0b0c438aff575cf6d0d02ce9a82c381d6'),
    'engine_deck': (META_RUN / 'decks/Artmage.ydk', 'a93c4d30d0945f58f741f1e135fdcbc0b0c438aff575cf6d0d02ce9a82c381d6'),
    'vocabulary': (META_RUN / 'code_list.txt', 'aeb5be89d4a5591ccc5cce6af96def74ef47463a55265f75c11b92029e772e7b'),
    'meta_manifest': (META_RUN / 'manifest.json', 'ace4f6b8408011a55e9b61d3c4917580942d86960bb3f087afe75ce2774c824a'),
    'meta_latest_run': (P / 'experiments/meta-opponents-20261107/latest-run.txt', '32770bc4534c729e66f7addbc76770b78798ee4e618b40fd9baa45e83392e544'),
    'article_latest_run': (P / 'experiments/opponent-article-training-20261108/latest-run.txt', '39db5d660d4bde37dbd9b6fc0c662781da48b71b0e6c50d8357a7a42f73b0159'),
    'model_code': (MODEL_CODE / 'opponent_article_bc_common.py', 'f4c3903e3a4841b97f6ffb8840bb4cfa5513b5abfead9ff41fa7c785ceba6cce'),
    'witness_code': (MODEL_CODE / 'article_witness.py', '0e4f3a0e92c86aba7a95f96c1e5f93f6a0293eaea7f3ab8a63dee772455ae07a'),
    'adapter': (ADAPTER_DIR / 'battle_position.py', '4fe74c83a460ee3565c3ff276ce2d080b233a6152439ec92147ba90d60d46f90'),
    'adapter_cards': (ADAPTER_DIR / 'position_cards.json', '1baa04349bb31833ff01b05cb365ee0f9416de827ac4fbf52b835011a35f1fde'),
    'native': (NATIVE, '19a60f38a3d39b61a2ee082168d5a788a1022b2eb569633816596c8b3de21294'),
    'runtime_json': (P / 'evaluation/opponent-article-basics-20261108/runtime.json', 'ebdfc37ed9c7ebcae73b81281876dd6f22bd8d9afcbdfe8f78905dcc9c220d14'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pins():
    """Hash every pinned file; raise on any difference. Returns {name: sha256}."""
    out = {}
    for name, (path, digest) in PINS.items():
        got = sha(path)
        if got != digest:
            raise RuntimeError(f'Pinned file changed: {name} {path} {got} != {digest}')
        out[name] = got
    return out
