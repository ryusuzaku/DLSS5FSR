#!/usr/bin/env python3
"""Extract text source from a downloaded upstream ZIP without running it."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXTENSIONS={'.py','.h','.hpp','.hlsl','.hlsli','.hip','.cpp','.cu','.md','.sh','.ps1','.inc','.txt'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def extract(archive,out):
    archive=Path(archive);out=Path(out)
    if out.exists():raise FileExistsError(f'refusing to overwrite {out}')
    prefix='dlss5-on-amd-9070xt-porting-0.28.1/'
    selected=[]
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name=info.filename
            if info.is_dir() or not name.startswith(prefix):continue
            rel=PurePosixPath(name[len(prefix):])
            if not rel.parts or '..' in rel.parts or rel.is_absolute():raise ValueError('unsafe ZIP path')
            if rel.suffix.lower() not in EXTENSIONS or info.file_size>5_000_000:continue
            selected.append((info,rel))
        if not selected:raise ValueError('no source files in archive')
        out.mkdir(parents=True)
        for info,rel in selected:
            dest=out.joinpath(*rel.parts)
            if not dest.resolve().is_relative_to(out.resolve()):raise ValueError('ZIP path escaped output')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(z.read(info))
    report=dict(archive=str(archive.resolve()),archive_sha256=digest(archive),
                source_files=len(selected),source_bytes=sum(info.file_size for info,_ in selected),
                filter='text source <=5 MB/file; no binaries, captures, weights, logs or executable runs')
    (out/'SOURCE_PROVENANCE.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'dlss5-on-amd-9070xt-porting-0.28.1.zip')
    p.add_argument('--out',type=Path,default=ROOT/'ref/dlss5-port-0.28.1')
    a=p.parse_args();print(json.dumps(extract(a.archive,a.out),indent=2))


if __name__=='__main__':main()
