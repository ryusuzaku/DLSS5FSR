#!/usr/bin/env python3
"""List substantive text changes between vendored upstream snapshots."""
from pathlib import Path
import argparse
import collections

ROOT=Path(__file__).resolve().parents[1]


def normalized(path):
    return path.read_bytes().replace(b'\r\n',b'\n')


def audit(old,new):
    old=Path(old);new=Path(new)
    changed=[];added=[];same=[]
    for path in new.rglob('*'):
        if not path.is_file() or path.name=='SOURCE_PROVENANCE.json':continue
        rel=path.relative_to(new);prior=old/rel
        if not prior.is_file():added.append(str(rel))
        elif normalized(prior)==normalized(path):same.append(str(rel))
        else:changed.append(str(rel))
    return changed,added,same


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,default=ROOT/'ref/dlss5-port')
    p.add_argument('--new',type=Path,default=ROOT/'ref/dlss5-port-0.28.1')
    a=p.parse_args();changed,added,same=audit(a.old,a.new)
    print(f'changed={len(changed)} added={len(added)} same={len(same)}')
    for label,items in (('CHANGED',changed),('ADDED',added)):
        print(label)
        for item in items:print(item)


if __name__=='__main__':main()
