"""Public ONNX producers for the resident candidate's external inputs.

Extracts only the early graph needed by the resident server: the block4
downsample (resident encoder5 input), the block4 skip and the preblock0 skip.
The same conversions as the offline stages are applied: FP8 rounding and the
multihead peer-to-native map for boundary4, FP8 rounding and the C32
peer-to-native map for skip4, and the C32 map for the preblock0 skip. The
session is kept between frames; every call runs inference again. These are
public-graph producers, not original NVIDIA kernels.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import onnxruntime as ort

from audit_peer_native_c32_basis import peer_to_native, peer_to_native_multihead
from cached_public_branch import digest, extract_cached
from check_c256_ffn_candidate import F
from extract_peer_encoder64_inputs import NODES as C64_NODES
from extract_peer_decoder66_inputs import SKIP4
from extract_peer_coherent_head_inputs import SKIP
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from shared_public_reference import input_identity

NODES = dict(block4_down=C64_NODES['block4_down'], skip4=SKIP4, skip0=SKIP)
SHAPES = dict(block4_down=(1, 64, 64, 64), skip4=(1, 128, 128, 32), skip0=(1, 256, 256, 32))


class PublicInputs:
    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if digest(MODEL) != MODEL_SHA256:
            raise ValueError('pinned public model differs')
        self.branch = cache_dir / 'resident_inputs.onnx'
        extract_cached(MODEL, self.branch, ['rgb'], list(NODES.values()), MODEL_SHA256)
        self.branch_sha256 = digest(self.branch)
        self.session = ort.InferenceSession(str(self.branch), providers=['CPUExecutionProvider'])
        names = [o.name for o in self.session.get_outputs()]
        if names != list(NODES.values()):
            raise ValueError('extracted branch outputs differ from the requested nodes')
        self.p64 = peer_to_native_multihead(np.arange(64))
        self.p32 = peer_to_native(np.arange(32))

    def run(self, prepared_dir):
        prepared_dir = Path(prepared_dir)
        start = time.monotonic()
        rgb = np.fromfile(prepared_dir / 'color_linear.f32', '<f4').reshape(256, 256, 3)
        identity = input_identity(prepared_dir / 'manifest.json', rgb)
        if digest(self.branch) != self.branch_sha256:
            raise ValueError('resident input branch changed while loaded')
        arrays = self.session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
        public = {}
        for (name, _), array in zip(NODES.items(), arrays):
            array = np.asarray(array, '<f4')
            if array.shape != SHAPES[name] or not np.isfinite(array).all():
                raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
            public[name] = array[0]
        boundary = np.empty_like(public['block4_down'])
        boundary[..., self.p64] = public['block4_down']
        skip4 = np.empty_like(public['skip4'])
        skip4[..., self.p32] = F(public['skip4'])
        skip0 = np.empty_like(public['skip0'])
        skip0[..., self.p32] = public['skip0']
        inputs = dict(c64_input=F(boundary), skip4=skip4, skip0=skip0, color=rgb)
        info = dict(identity, branch_sha256=self.branch_sha256, seconds=time.monotonic()-start,
                    input_sha256={k: hashlib.sha256(np.asarray(v, '<f4').tobytes()).hexdigest()
                                  for k, v in inputs.items()})
        return inputs, info


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prepared_dir', type=Path)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    producer = PublicInputs(a.cache_dir)
    inputs, info = producer.run(a.prepared_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in inputs.items():
        np.asarray(array, '<f4').tofile(a.output_dir / f'{name}.f32')
    (a.output_dir / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))
