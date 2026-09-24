#!/usr/bin/env python3
"""Inventory a DLSS5@AMD full package for offline map-recovery assets.

Does not extract or execute package contents. The C32 requirements mirror
recover_head70_maps.py; C256 block files are reported as the next audit input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

STEMS_C32 = tuple(f'{"post70" if b == 70 else f"block{b}"}-{kind}'
                  for b in (1, 2, 3, 67, 68, 69, 70)
                  for kind in ('ffn', 'attention')) + ('post70-scales', 'post70-head')
STEMS_C256 = tuple(f'block{b}-{kind}'
                   for b in (*range(15, 22), *range(49, 56))
                   for kind in ('ffn', 'attention'))
COUNTS_C32 = {stem: 8736 if stem.endswith('-ffn') else 8225
              for stem in STEMS_C32 if stem not in ('post70-scales', 'post70-head')}
COUNTS_C32.update({'post70-scales': 64, 'post70-head': 96})


def is_asset_name(name: str, direct: bool) -> bool:
    normalized = name.replace('\\', '/')
    return (direct and '/' not in normalized.strip('/')) or '/native-game-tiled-assets/' in f'/{normalized}'


def inventory(path: Path, with_hashes: bool) -> dict:
    if path.is_file():
        if not zipfile.is_zipfile(path):
            raise ValueError('input file is not a ZIP')
        archive = zipfile.ZipFile(path)
        entries = [(i.filename.replace('\\', '/'), i.file_size, i)
                   for i in archive.infolist() if not i.is_dir()]
        direct = False
    elif path.is_dir():
        archive = None
        asset_dirs = [path] if path.name.lower() == 'native-game-tiled-assets' else [
            p for p in path.rglob('native-game-tiled-assets') if p.is_dir()]
        if len(asset_dirs) != 1:
            raise ValueError(f'expected one native-game-tiled-assets directory; found {len(asset_dirs)}')
        asset_dir = asset_dirs[0]
        entries = [(p.name, p.stat().st_size, p) for p in asset_dir.iterdir() if p.is_file()]
        direct = True
    else:
        raise FileNotFoundError(path)
    try:
        candidates: dict[str, list[tuple[str, int, object]]] = {}
        for name, size, source in entries:
            if is_asset_name(name, direct):
                candidates.setdefault(name.rsplit('/', 1)[-1], []).append((name, size, source))
        groups = {}
        for label, stems in (('c32_recovery', STEMS_C32), ('c256_followup', STEMS_C256)):
            files = {}
            for stem in stems:
                matches = [item for ext in ('.f16', '.f32')
                           for item in candidates.get(stem + ext, ())]
                result = {'status': 'missing' if not matches else 'duplicate' if len(matches) > 1 else 'present'}
                if len(matches) == 1:
                    name, size, source = matches[0]
                    result.update(path=name, bytes=size)
                    if label == 'c32_recovery':
                        expected = COUNTS_C32[stem] * (2 if name.endswith('.f16') else 4)
                        result['expected_bytes'] = expected
                        if size != expected:
                            result['status'] = 'wrong_size'
                    if with_hashes:
                        h = hashlib.sha256()
                        if archive:
                            stream = archive.open(source)
                        else:
                            stream = source.open('rb')
                        with stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                                h.update(chunk)
                        result['sha256'] = h.hexdigest()
                files[stem] = result
            groups[label] = {'ready': all(x['status'] == 'present' for x in files.values()),
                             'present': sum(x['status'] == 'present' for x in files.values()),
                             'required': len(stems), 'files': files}
        return {'source_name': path.name, 'source_kind': 'zip' if archive else 'directory',
                'asset_file_count': sum(len(v) for v in candidates.values()),
                'hashes_computed': with_hashes, **groups}
    finally:
        if archive:
            archive.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='full package ZIP or extracted native-game-tiled-assets directory')
    parser.add_argument('--hash', action='store_true', help='include SHA256 of found target files')
    parser.add_argument('--out', type=Path, help='save JSON report at this path')
    args = parser.parse_args()
    report = inventory(args.path, args.hash)
    encoded = json.dumps(report, indent=2) + '\n'
    if args.out:
        if args.out.exists():
            raise FileExistsError(f'refusing to overwrite {args.out}')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding='utf-8')
    print(encoded, end='')
    return 0 if report['c32_recovery']['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
