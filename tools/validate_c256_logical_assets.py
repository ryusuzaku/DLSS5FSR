#!/usr/bin/env python3
"""Compare predeclared C256 candidate maps with supplied logical assets.

The candidate maps are derived from C64/C128 and never fitted to these C256
files. This offline check reads a full package ZIP or its extracted asset
directory and reports only counts and hashes, not model coefficients. Exact
agreement validates this package's logical-coordinate interpretation; the
package's own provenance and original-kernel parity remain separate questions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = tuple(range(15, 22)) + tuple(range(49, 56))
DLL_SHA256 = 'E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E'
FFN_PARTS = (('W1', 1024*256), ('W2', 256*1024),
             ('W3', 256*256), ('ffn_skip', 256))
ATTENTION_PARTS = (('Q', 256*256), ('K', 256*256), ('V', 256*256),
                   ('projection', 256*256), ('bias', 8*64*64),
                   ('scales', 8), ('attention_skip', 256))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compare_parts(expected, supplied, parts):
    if expected.dtype != np.float32 or expected.ndim != 1 or len(expected) != sum(n for _, n in parts):
        raise ValueError('invalid candidate component layout')
    if supplied.shape != expected.shape or not np.isfinite(expected).all():
        raise ValueError('invalid candidate/supplied values')
    result = {}
    offset = 0
    for name, count in parts:
        a = expected[offset:offset+count]
        b = supplied[offset:offset+count]
        mismatch = np.flatnonzero(a != b)
        result[name] = dict(values=count, exact=int(count-len(mismatch)),
                            mismatches=int(len(mismatch)),
                            first_mismatch_index=(int(mismatch[0]) if len(mismatch) else None))
        offset += count
    return result


def read_logical(assets, stem, count):
    before = set(assets.provenance)
    values = assets.read(stem, count)
    names = set(assets.provenance) - before
    if len(names) != 1:
        raise ValueError(f'asset provenance ambiguous for {stem}')
    name = names.pop()
    if not name.endswith(('.f16', '.f32')):
        raise ValueError('unsupported logical asset dtype')
    return values, name


def asset_source(path):
    path = Path(path)
    if path.is_dir() and path.name.lower() != 'native-game-tiled-assets' and \
       not any(path.glob('block*-ffn.f16')) and not any(path.glob('block*-ffn.f32')):
        folders = [p for p in path.rglob('native-game-tiled-assets') if p.is_dir()]
        if len(folders) != 1:
            raise ValueError(f'expected one native-game-tiled-assets directory, found {len(folders)}')
        return folders[0]
    return path


def run(asset_path, out, dll_path, blocks=BLOCKS, raw_only=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    dll_hash = file_digest(dll_path)
    if dll_hash.upper() != DLL_SHA256:
        raise ValueError(f'DLL SHA256 differs from required model build: {dll_hash}')
    records = {item['name']: item for item in
               json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    if raw_only:
        ffn_maps = attention_maps = assets = None
    else:
        from check_c256_ffn_candidate import candidate_ffn_maps, decode_ffn
        from check_c256_attention_candidate import candidate_attention_maps, decode_attention
        from recover_head70_maps import Assets
        ffn_maps = candidate_ffn_maps()
        attention_maps = candidate_attention_maps()
        assets = Assets(asset_source(asset_path))
    report = dict(method=('packed C256 tensors versus matching original DLL' if raw_only else
                          'predeclared C256 candidate maps; no C256 fitting'),
                  source_name=(Path(asset_path).name if asset_path else None),
                  archive_sha256=(assets.provenance.get('archive_sha256') if assets else None),
                  dll_sha256=dll_hash, required_dll_sha256=DLL_SHA256,
                  requested_blocks=list(blocks), raw_only=raw_only, blocks={},
                  candidate_source_sha256=({
                      name: digest((ROOT/'tools'/name).read_bytes()) for name in
                      ('check_c256_ffn_candidate.py', 'check_c256_attention_candidate.py')}
                      if not raw_only else {}),
                  original_kernel_executed=False)
    try:
        with Path(dll_path).open('rb') as dll:
            for block in blocks:
                entry = {}
                report['blocks'][str(block)] = entry
                step = 'raw_tensor'
                try:
                    record = records[f'block{block}.layer0.layer']
                    raw_path = ROOT/'dlss5-analysis/tensors'/f"tensor_{record['index']:03d}.bin"
                    raw_bytes = raw_path.read_bytes()
                    raw = np.frombuffer(raw_bytes, np.uint8)
                    if raw.size != 689232 or record['payload_size'] != raw.size:
                        raise ValueError('ordinary C256 raw record has wrong size')
                    dll.seek(record['abs_file_offset'])
                    if dll.read(raw.size) != raw_bytes:
                        raise ValueError('packed tensor differs from supplied DLL bytes')
                    entry['raw_tensor_matches_dll'] = True
                    entry['raw_tensor_sha256'] = digest(raw_bytes)
                    entry['raw_tensor_bytes'] = raw.size
                    if raw_only:
                        entry['status'] = 'raw_exact'
                        continue
                    step = 'candidate_decode'
                    w1, w2, w3, ffn_skip = decode_ffn(raw, ffn_maps)
                    qkv, bias, scales, projection, attention_skip = decode_attention(raw, attention_maps)
                    candidates = {
                        'ffn': np.concatenate((w1.ravel(), w2.ravel(), w3.ravel(), ffn_skip)).astype(np.float32),
                        'attention': np.concatenate((qkv[0].ravel(), qkv[1].ravel(), qkv[2].ravel(),
                                                     projection.ravel(), bias.ravel(), scales,
                                                     attention_skip)).astype(np.float32),
                    }
                    for kind, parts in (('ffn', FFN_PARTS), ('attention', ATTENTION_PARTS)):
                        step = f'{kind}_asset'
                        stem = f'block{block}-{kind}'
                        expected = candidates[kind]
                        logical, name = read_logical(assets, stem, sum(n for _, n in parts))
                        # A half-precision asset cannot retain an f32 scale; compare
                        # against the candidate represented in that asset's dtype.
                        if name.endswith('.f16'):
                            expected = expected.astype('<f2').astype(np.float32)
                        checks = compare_parts(expected, logical, parts)
                        entry[kind] = dict(asset_name=name,
                                           asset_sha256=assets.provenance[name]['sha256'],
                                           values=int(expected.size),
                                           exact=sum(x['exact'] for x in checks.values()),
                                           components=checks)
                    entry['status'] = ('exact' if all(entry[k]['exact'] == entry[k]['values']
                                                      for k in ('ffn', 'attention')) else 'mismatch')
                except (KeyError, ValueError, FileNotFoundError) as error:
                    entry['status'] = 'input_error'
                    entry['error_type'] = type(error).__name__
                    entry['failed_step'] = step
            target = 'raw_exact' if raw_only else 'exact'
            report['status'] = (target if all(v['status'] == target for v in report['blocks'].values())
                                else 'failed')
            report['raw_bytes_verified'] = sum(v.get('raw_tensor_bytes', 0)
                                               for v in report['blocks'].values()
                                               if v.get('raw_tensor_matches_dll'))
            report['checked_values'] = sum(v.get(k, {}).get('values', 0)
                                           for v in report['blocks'].values() for k in ('ffn', 'attention'))
            report['exact_values'] = sum(v.get(k, {}).get('exact', 0)
                                         for v in report['blocks'].values() for k in ('ffn', 'attention'))
            report['asset_hashes'] = ({k: v for k, v in assets.provenance.items()
                                       if k != 'archive_sha256'} if assets else {})
    finally:
        if assets:
            assets.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] in ('exact', 'raw_exact') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path,
                        help='full package ZIP or extracted native-game-tiled-assets directory')
    parser.add_argument('--dll', type=Path, required=True,
                        help='matching original nvngx_dlssnr.dll; only its hash and tensor slices are read')
    parser.add_argument('--out', type=Path, required=True,
                        help='new JSON report path; refuses to overwrite')
    parser.add_argument('--blocks', type=int, nargs='+', choices=BLOCKS, default=BLOCKS,
                        help='subset for diagnosis; default checks all fourteen blocks')
    parser.add_argument('--raw-only', action='store_true',
                        help='verify extracted packed tensors against the DLL without logical assets')
    args = parser.parse_args()
    if len(set(args.blocks)) != len(args.blocks):
        parser.error('duplicate block number')
    if not args.raw_only and args.assets is None:
        parser.error('--assets is required unless --raw-only is set')
    return run(args.assets, args.out, args.dll, tuple(args.blocks), args.raw_only)


if __name__ == '__main__':
    raise SystemExit(main())
