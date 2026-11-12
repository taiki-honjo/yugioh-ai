"""Independently read back every exported file, preferably from Windows."""
import argparse,hashlib,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
root=args.directory.resolve();mp=root/'SHA256-manifest.json'
manifest=json.loads(mp.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    assert len(manifest)>100
    for relative,record in manifest.items():
        path=(root/relative).resolve();assert path.is_relative_to(root)
        assert path.is_file() and path.stat().st_size==record['bytes'],relative
        assert sha(path)==record['sha256'],relative
verify()
p=root/'verification.json';checks=json.loads(p.read_text(encoding='utf-8'))
checks.update(package_file_hashes_verified=True,verified_files_excluding_manifest=len(manifest))
p.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest['verification.json']=dict(sha256=sha(p),bytes=p.stat().st_size)
mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
verify()
print(json.dumps(dict(status='passed',verified_files=len(manifest),output=str(root)),ensure_ascii=False))
