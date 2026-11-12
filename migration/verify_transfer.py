"""Check a transfer snapshot using only the standard library; never start the engine."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def local_path(name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name:
        raise ValueError(f'Unsafe manifest path: {name}')
    return ROOT.joinpath(*rel.parts)


def main():
    manifest = ROOT / 'migration/tracked-files.jsonl'
    if not manifest.exists():
        print('Missing migration/tracked-files.jsonl', file=sys.stderr)
        return 1
    errors, links = [], []
    checked = 0
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        path = local_path(row['path'])
        checked += 1
        if 'link' in row:
            if not path.is_symlink() or os.readlink(path) != row['link']:
                errors.append(f"Link differs: {row['path']}")
            if row['link'].startswith('/'):
                links.append({'path': row['path'], 'target': row['link'], 'target_exists': path.exists()})
        elif not path.is_file() or path.is_symlink():
            errors.append(f"Missing file: {row['path']}")
        elif path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            errors.append(f"Contents differ: {row['path']}")
    # Resolve historical absolute references in metadata into this checkout without rewriting originals.
    prefix = '/home/hatk8/projects/yugioh-ai/'
    hand_manifest = 'experiments/artmage-hand-branch-20261110/data/manifest.json'
    branch = json.loads(local_path(hand_manifest).read_text())
    base = json.loads(local_path(branch['existing']['manifest'].removeprefix(prefix)).read_text())
    refs = [item for items in base['groups'].values() for item in items]
    refs += [h for pair in branch['pairs'] for h in pair['hands']]
    for item in refs:
        name = item['path'].removeprefix(prefix)
        path = local_path(name)
        if not path.is_file() or digest(path) != item['sha256']:
            errors.append(f'Artmage rehearsal asset missing/changed: {name}')
        if item.get('record_sha256'):
            record = path.with_name('record.json')
            if not record.is_file() or digest(record) != item['record_sha256']:
                errors.append(f'Artmage demonstration record missing/changed: {name}')
    print(json.dumps({'checked_files_and_links': checked,
                      'artmage_rehearsal_sequences': sum(x.get('sequences', 1) for x in refs),
                      'absolute_links_requiring_compatible_layout': links,
                      'errors': errors, 'engine_started': False}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
