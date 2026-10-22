#!/usr/bin/env python3
"""Import the353 frozen arrays from the existing transfer bundle, then verify them."""
import argparse,hashlib,json,os,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def expected():
    rows={}
    for line in (ROOT/'ASSET_SHA256SUMS').read_text().splitlines():
        digest,name=line.split(maxsplit=1);p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.parts[0]!='assets':raise ValueError('Unsafe asset path')
        if name in rows:raise ValueError('Duplicate asset '+name)
        rows[name]=digest
    if len(rows)!=353:raise ValueError('Require194 source and159 dry arrays')
    return rows

def verify():
    rows=expected()
    missing=[name for name in rows if not (ROOT/name).is_file()]
    if missing:raise FileNotFoundError(str(len(missing))+' frozen arrays missing; run prepare_inputs.py --bundle-root /path/to/8_jetson_export_20261022 or --archive /path/to/Jetson_export_20261022.tar.gz; first missing '+missing[0])
    for name,digest in rows.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen array hash mismatch '+name)
    return dict(status='PASS',arrays=353,asset_manifest_sha256=sha(ROOT/'ASSET_SHA256SUMS'))

def write_file(name,digest,stream):
    dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if not dest.is_file() or sha(dest)!=digest:raise ValueError('Preserve conflicting existing asset '+str(dest))
        return
    temp=dest.with_name(dest.name+'.'+str(os.getpid())+'.partial')
    try:
        with temp.open('xb') as out:shutil.copyfileobj(stream,out,4<<20)
        if sha(temp)!=digest:raise ValueError('Asset hash mismatch '+name)
        temp.replace(dest)
    finally:
        if temp.exists():temp.unlink()

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--bundle-root',type=Path);g.add_argument('--archive',type=Path);g.add_argument('--check',action='store_true');p.add_argument('--link',action='store_true',help='Use verified symlinks to an existing unpacked bundle, which must remain in place');a=p.parse_args()
    if a.link and not a.bundle_root:p.error('--link requires --bundle-root')
    rows=expected();source=None
    if a.bundle_root:
        source=str(a.bundle_root.resolve())
        for name,digest in rows.items():
            src=a.bundle_root/name
            if not src.is_file() or sha(src)!=digest:raise ValueError('Missing or mismatching bundle asset '+str(src))
            if a.link:
                dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists() or dest.is_symlink():
                    if not dest.is_file() or sha(dest)!=digest:raise ValueError('Preserve conflicting existing asset '+str(dest))
                else:dest.symlink_to(src.resolve())
            else:
                with src.open('rb') as stream:write_file(name,digest,stream)
    elif a.archive:
        info=json.loads((ROOT/'SOURCE_PACKAGE.json').read_text());digest=sha(a.archive)
        if digest!=info['transfer_archive_sha256']:raise ValueError('Transfer archive SHA256 mismatch; require '+info['transfer_archive_sha256'])
        source=dict(archive=str(a.archive.resolve()),sha256=digest);seen=set()
        with tarfile.open(a.archive,'r|gz') as tar:
            for member in tar:
                path=Path(member.name)
                if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive member')
                if not path.parts or path.parts[0]!=info['archive_prefix']:continue
                name=str(path.relative_to(info['archive_prefix']))
                if name not in rows:continue
                if not member.isfile() or name in seen:raise ValueError('Invalid/duplicate archive asset '+name)
                stream=tar.extractfile(member)
                with stream:write_file(name,rows[name],stream)
                seen.add(name)
        if seen!=set(rows):raise ValueError('Archive missing required assets')
    result=verify()
    if source is not None:
        result.update(source=source,linked=bool(a.link));temp=ROOT/'ASSETS_READY.json.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(ROOT/'ASSETS_READY.json')
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
