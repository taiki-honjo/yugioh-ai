#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

project=Path('/home/hatk8/projects/yugioh-ai')
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=Path)
args=parser.parse_args()
manifest=json.loads((project/'env/elfnotes/runtime.json').read_text())
for field,checksum in [('native_binary','native_sha256'),('vocabulary','vocabulary_sha256'),('deck','deck_sha256')]:
    path=Path(manifest[field])
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=manifest[checksum]: raise SystemExit(f'固定版から変更されています: {path}')
if args.checkpoint:
    associated=args.checkpoint.parent/'runtime.json'
    if not associated.is_file(): raise SystemExit('重みに対応するruntime.jsonがありません。カードIDの対応を確認してください。')
    prior=json.loads(associated.read_text())
    for field in ['vocabulary_sha256','deck_sha256']:
        if prior[field]!=manifest[field]: raise SystemExit(f'重みと実行環境の対応が異なります: {field}')
    compatible=set(manifest.get('compatible_checkpoint_native_sha256',[manifest['native_sha256']]))
    if prior['native_sha256'] not in compatible:
        raise SystemExit('重みに対応するエンジンとの互換性が確認されていません。')
print('エルフーンの固定環境・デッキ・カードID対応: OK')
