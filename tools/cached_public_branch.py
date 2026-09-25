"""Reuse a pinned public ONNX branch across captured frames.

The candidate runner writes each stage into the same work directory on every
preview update. Extracted graph structure depends on the pinned source model
and output nodes, not on the frame, so keep it only while those inputs and the
branch file hash match its sidecar manifest.
"""

from pathlib import Path
import hashlib
import json

import onnx


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_cached(model, branch, input_nodes, output_nodes, model_sha256):
    model = Path(model)
    branch = Path(branch)
    manifest = branch.with_suffix(branch.suffix + '.cache.json')
    expected = dict(source_model_sha256=model_sha256,
                    input_nodes=list(input_nodes), output_nodes=list(output_nodes))
    if branch.is_file() and manifest.is_file():
        try:
            saved = json.loads(manifest.read_text())
            if (all(saved.get(key) == value for key, value in expected.items()) and
                    saved.get('branch_sha256') == digest(branch)):
                return False
        except (OSError, ValueError, TypeError):
            pass
    if digest(model) != model_sha256:
        raise ValueError('pinned public model differs before branch extraction')
    branch.parent.mkdir(parents=True, exist_ok=True)
    temp = branch.with_name(branch.stem + '.extracting.onnx')
    try:
        onnx.utils.extract_model(str(model), str(temp),
                                 expected['input_nodes'], expected['output_nodes'])
        branch_sha = digest(temp)
        temp.replace(branch)
        manifest_temp = manifest.with_name(manifest.name + '.tmp')
        manifest_temp.write_text(json.dumps(dict(**expected, branch_sha256=branch_sha),
                                            indent=2) + '\n')
        manifest_temp.replace(manifest)
    finally:
        temp.unlink(missing_ok=True)
    return True
