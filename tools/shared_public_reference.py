"""Compute public FP16 comparison boundaries once per captured frame.

Only the extracted graph is cached between frames. Every call to prepare()
runs inference again and binds its tensor files to the prepared input hashes.
These are public ONNX controls, not original NVIDIA kernel outputs.
"""

from pathlib import Path
import argparse
import hashlib
import io
import json
import time

import numpy as np
import onnxruntime as ort

from cached_public_branch import digest, extract_cached
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256


SCHEMA = 'dlss5fsr-public-reference-v1'
BRANCH = 'all_boundaries.onnx'


def output_groups():
    from extract_peer_encoder64_inputs import NODES as c64
    from extract_peer_encoder128_inputs import NODES as c128
    from extract_peer_encoder256_inputs import NODES as c256
    from extract_peer_split512_encoder_inputs import NODES as encoder30
    from extract_peer_decoder39_inputs import NODES as vit39
    from extract_peer_split512_inputs import NODES as decoder47
    from extract_peer_decoder48_inputs import NODES as decoder55
    from extract_peer_decoder56_inputs import NODES as decoder61
    from extract_peer_decoder62_inputs import NODES as decoder65
    from extract_peer_decoder66_inputs import (
        BLOCK65, SKIP4, MERGE66, BLOCK66, BLOCK67, BLOCK68, BLOCK69)
    from extract_peer_coherent_head_inputs import SKIP, LATENT, FUSED, BODY, ENHANCED

    encoder22 = dict(c64)
    encoder22.update({k: v for k, v in c128.items() if k != 'block8_down'})
    encoder22.update({k: v for k, v in c256.items() if k != 'block14_down'})
    return dict(encoder22=encoder22, encoder30=encoder30, vit39=vit39,
                decoder47=decoder47, decoder55=decoder55, decoder61=decoder61,
                decoder65=decoder65,
                decoder69=dict(block65=BLOCK65, skip4=SKIP4, merge66=MERGE66,
                               block66=BLOCK66, block67=BLOCK67,
                               block68=BLOCK68, block69=BLOCK69),
                head_inputs=dict(skip=SKIP, latent=LATENT, fused=FUSED, body=BODY,
                                 enhanced=ENHANCED, final='output'))


def tensor_name(node):
    return hashlib.sha256(node.encode('utf-8')).hexdigest() + '.npy'


def input_identity(prepared_file, rgb):
    prepared_file = Path(prepared_file)
    raw = prepared_file.read_bytes()
    prepared = json.loads(raw)
    color_sha = hashlib.sha256(np.asarray(rgb, dtype='<f4').tobytes()).hexdigest()
    if (prepared['output_size'] != [256, 256] or rgb.shape != (256, 256, 3) or
            not np.isfinite(rgb).all() or prepared['color_linear_sha256'] != color_sha):
        raise ValueError('prepared input or manifest differs for public reference')
    return dict(source_model_sha256=MODEL_SHA256,
                source_capture_sha256=prepared['source_capture_sha256'],
                prepared_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                color_linear_sha256=color_sha)


def prepare(prepared_dir, output_dir):
    start = time.monotonic()
    prepared_dir, out = Path(prepared_dir), Path(output_dir)
    rgb = np.fromfile(prepared_dir / 'color_linear.f32', '<f4').reshape(256, 256, 3)
    identity = input_identity(prepared_dir / 'manifest.json', rgb)
    if digest(MODEL) != MODEL_SHA256:
        raise ValueError('pinned public model differs')
    out.mkdir(parents=True, exist_ok=True)
    nodes = list(dict.fromkeys(node for group in output_groups().values()
                              for node in group.values()))
    branch = out / BRANCH
    extract_start = time.monotonic()
    rebuilt = extract_cached(MODEL, branch, ['rgb'], nodes, MODEL_SHA256)
    extract_seconds = time.monotonic() - extract_start
    load_start = time.monotonic()
    session = ort.InferenceSession(str(branch), providers=['CPUExecutionProvider'])
    load_seconds = time.monotonic() - load_start
    run_start = time.monotonic()
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    run_seconds = time.monotonic() - run_start
    outputs = {}
    for node, array in zip(nodes, arrays):
        array = np.asarray(array, dtype='<f4')
        if not np.isfinite(array).all():
            raise ValueError(f'nonfinite public reference: {node}')
        path = out / tensor_name(node)
        temp = path.with_suffix('.npy.tmp')
        with temp.open('wb') as dest:
            np.save(dest, array, allow_pickle=False)
        temp.replace(path)
        outputs[node] = dict(sha256=digest(path), shape=list(array.shape), dtype='<f4')
    report = dict(schema=SCHEMA, **identity, branch_sha256=digest(branch),
                  outputs=outputs, single_inference_call=True,
                  runtime_version=ort.__version__, provider='CPUExecutionProvider',
                  original_kernel_executed=False, graph_rebuilt=rebuilt,
                  extract_seconds=extract_seconds, session_load_seconds=load_seconds,
                  inference_seconds=run_seconds, total_seconds=time.monotonic() - start)
    # Publish last. A partial write leaves mismatching hashes and is rejected.
    temp = out / 'manifest.json.tmp'
    temp.write_text(json.dumps(report, indent=2) + '\n')
    temp.replace(out / 'manifest.json')
    return report


def run_reference(prepared_file, rgb, nodes, branch_file, shared_dir=None):
    """Return ordered arrays, actual graph path, and additional report ancestry."""
    identity = input_identity(prepared_file, rgb)
    if shared_dir is None:
        extract_cached(MODEL, branch_file, ['rgb'], list(nodes.values()), MODEL_SHA256)
        session = ort.InferenceSession(str(branch_file), providers=['CPUExecutionProvider'])
        arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
        return arrays, Path(branch_file), {}

    shared_dir = Path(shared_dir)
    raw_manifest = (shared_dir / 'manifest.json').read_bytes()
    manifest = json.loads(raw_manifest)
    if (manifest.get('schema') != SCHEMA or
            any(manifest.get(key) != value for key, value in identity.items()) or
            manifest.get('single_inference_call') is not True):
        raise ValueError('shared public reference belongs to a different input or model')
    branch = shared_dir / BRANCH
    if digest(branch) != manifest['branch_sha256']:
        raise ValueError('shared public reference graph hash differs')
    arrays = []
    for node in nodes.values():
        entry = manifest['outputs'].get(node)
        if not entry:
            raise ValueError(f'shared public reference is missing output {node}')
        raw = (shared_dir / tensor_name(node)).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f'shared public reference tensor hash differs: {node}')
        array = np.load(io.BytesIO(raw), allow_pickle=False)
        if (entry['dtype'] != '<f4' or array.dtype != np.dtype('<f4') or
                list(array.shape) != entry['shape'] or not np.isfinite(array).all()):
            raise ValueError(f'shared public reference tensor shape/type differs: {node}')
        arrays.append(array)
    return arrays, branch, dict(shared_public_reference_manifest_sha256=
                               hashlib.sha256(raw_manifest).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.prepared_dir, args.output_root)
    print(f'{len(report["outputs"])} public outputs in {report["total_seconds"]:.2f}s')
