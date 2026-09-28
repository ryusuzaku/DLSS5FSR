"""Connect resident23-47 to block48 prefix and C25648-55 using declared skip pairs."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile

import numpy as np
from check_split512_resident import ROOT, SHIFTS, digest, list_file, run as check_fixed
from check_resident_decoder47 import PRIOR_NAMES, NEW_NAMES
from check_decoder49_candidate import run as run_block
from check_upsample48_prefix import bits, multiply, H, F, e4m3fn
from gpu_test_runner import close_worker
from audit_native_vit_logical_map import logical_map

NAMES = (*PRIOR_NAMES, *NEW_NAMES, 'low48', 'merge48', 'decoder55')


def verify_files(folder, manifest):
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(folder / name) != sha:
            raise ValueError(f'changed fixture: {folder}/{name}')


def run(source, output, captured_candidate=None, synthetic_skip=False, head_fixture=None,
        skips=None, skips_provenance=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('use separate source and output directories')
    prior = json.loads((source / 'report.json').read_text())
    width,height,channels = prior['extent']
    sequence = ['a','b','zero','b','a']
    if (width not in (8,32) or height != 8 or channels != 512 or
            prior['sequence'] != sequence or not prior['metrics']['resident_decoder']):
        raise ValueError('requires validated resident23-47 input')
    if bool(captured_candidate) == bool(synthetic_skip):
        raise ValueError('declare exactly one captured candidate or synthetic skip')
    lines = (source / 'fixed_control/blocks.txt').read_text(encoding='mbcs').splitlines()
    fixtures = Path(lines[0].split('\t')[1]).parent
    head = Path(head_fixture).resolve() if head_fixture else fixtures / f'block30_head_{width//2}x4'
    output.mkdir(parents=True,exist_ok=True)
    base = check_fixed(fixtures,output/'fixed_control',width=width,head_fixture=head,repeats=1)
    if base['sources'] != prior['c512_sources']: raise ValueError('encoder ancestry changed')
    decoder_base = check_fixed(source/'references/a/decoder',output/'decoder_control',first=40,width=width,repeats=1)
    if decoder_base['sources'] != prior['decoder_sources']['a']: raise ValueError('decoder ancestry changed')
    if len(prior['exact_outputs']) != 5: raise ValueError('incomplete saved outputs')
    for name, record in zip(sequence,prior['exact_outputs']):
        frame=source/'references'/name
        if record['frame'] != name or digest(frame/'input.f32') != record['input_sha256']:
            raise ValueError('saved input changed')
        for tensor in (*PRIOR_NAMES,*NEW_NAMES):
            if digest(frame/f'{tensor}.f32') != record['outputs'][tensor]: raise ValueError('saved output changed')
    vit_folders=[Path(v) for v in (source/'vit_blocks.txt').read_text(encoding='mbcs').splitlines()]
    if len(vit_folders)!=8: raise ValueError('wrong ViT block count')
    previous=source/'references/a/bridge.f32'
    for block,folder in zip(range(31,39),vit_folders):
        m=json.loads((folder/'manifest.json').read_text());verify_files(folder,m)
        if m['block']!=block or (folder/'input.f32').read_bytes()!=previous.read_bytes():
            raise ValueError('ViT block/handoff differs')
        for part in ('expansion','contraction','qkv','projection'):
            raw_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{m[part+'_tensor']:03d}.bin"
            if digest(raw_path)!=m[part+'_sha256']:raise ValueError('ViT raw tensor changed')
        previous=folder/'projection_device.f32'
    if previous.read_bytes()!=(source/'references/a/vit.f32').read_bytes():raise ValueError('ViT38 differs')
    entry=source/'references/a/entry39'
    m=json.loads((entry/'manifest.json').read_text());verify_files(entry,m)
    gather=logical_map(width*height//4).astype('<i4')
    if ((source/'bridge_map.i32').read_bytes()!=gather.tobytes() or
            (entry/'inverse.i32').read_bytes()!=np.argsort(gather).astype('<i4').tobytes()):
        raise ValueError('forward/inverse candidate map differs')
    if (digest(ROOT/'dlss5-analysis/tensors'/f"tensor_{m['tensor']:03d}.bin")!=m['tensor_sha256'] or
            digest(entry/'inverse.i32')!=prior['inverse_map_sha256']):raise ValueError('decoder39 ancestry differs')
    if captured_candidate:
        capture=Path(captured_candidate).resolve()
        if width!=8:raise ValueError('captured skip fixture covers width8 only')
        enc=json.loads((capture/'encoder22/report.json').read_text())
        dec=json.loads((capture/'decoder47/report.json').read_text())
        skip_path=capture/'encoder22/block22/output/output_device.f32'
        if (enc['source_capture_sha256']!=dec['source_capture_sha256'] or
                enc['color_linear_sha256']!=dec['color_linear_sha256'] or
                enc['c256_blocks'][-1]['output_device_sha256']!=digest(skip_path) or
                dec['final_device_sha256']!=digest(source/'references/a/decoder47.f32') or
                (source/'references/a/input.f32').read_bytes()!=
                (capture/'encoder30/block23-8x8-s0/input.f32').read_bytes()):
            raise ValueError('captured encoder22 skip and resident A do not share verified ancestry')
        skip_a=np.fromfile(skip_path,'<f4').reshape(16,16,256)
        skip_provenance=dict(kind='same-capture AMD encoder22 output for A',sha256=digest(skip_path),
                             capture_sha256=enc['source_capture_sha256'],report_sha256=digest(capture/'encoder22/report.json'))
    else:
        skip_a=F(np.random.default_rng(4801).normal(0,.03125,(16,2*width,256)).astype(np.float32))
        skip_provenance=dict(kind='synthetic FP8 Gaussian skip; seed4801, sigma0.03125')
    if skips is None:
        skips=dict(a=skip_a,b=np.roll(skip_a,2,axis=1).copy(),zero=np.zeros_like(skip_a))
        paired='B shifts skip22 by two columns; zero has zero skip. B/zero are synthetic pairs, not full upstream encoder executions.'
    else:
        # Same-frame encoder22 outputs; a captured A must still equal the capture.
        if set(skips)!={'a','b','zero'} or not skips_provenance:raise ValueError('declared skips need a/b/zero and provenance')
        skips={k:np.asarray(v,np.float32).reshape(16,2*width,256) for k,v in skips.items()}
        if captured_candidate and skips['a'].tobytes()!=skip_a.tobytes():raise ValueError('declared skip A differs from capture')
        skip_provenance=dict(skip_provenance,declared=skips_provenance)
        paired='A/B/zero skips are same-frame encoder22 outputs of their declared C256 inputs'
    records={r['name']:r for r in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    raw_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{records['block48.layer0.layer']['index']:03d}.bin"
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=820784:raise ValueError('wrong block48 tensor size')
    rows=bits(256*512,[3]+list(range(6,13)));cols=bits(256*512,[1,0,4,5,2]+list(range(13,17)))
    if np.unique(rows*512+cols).size!=256*512:raise ValueError('projection map collides')
    weights=np.empty((256,512),np.float32);weights[rows,cols]=e4m3fn(raw[0x58000:0x78000])
    c=np.arange(256);order=(c//16)*16+(c%8)*2+(c%16//8)
    scale=np.empty(256,np.float32);scale[order]=raw[0x78200:0x78400].view('<f2').astype(np.float32)
    references={}
    try:
        for name in ('a','b','zero'):
            frame=output/'references'/name;prefix=frame/'prefix48';prefix.mkdir(parents=True,exist_ok=True)
            for tensor in ('input',*PRIOR_NAMES,*NEW_NAMES):
                shutil.copyfile(source/f'references/{name}/{tensor}.f32',frame/f'{tensor}.f32')
            skip=skips[name];skip.astype('<f4').tofile(frame/'skip22.f32')
            x=np.fromfile(frame/'decoder47.f32','<f4').reshape(8,width,512)
            low=multiply(x,weights);merged=F(H(np.repeat(np.repeat(low,2,0),2,1)+skip*scale))
            for tensor,array in dict(input=x,weights=weights,scale=scale,skip=skip,low=low,merged=merged).items():
                if not np.isfinite(array).all():raise ValueError('nonfinite prefix reference')
                np.asarray(array,'<f4').tofile(prefix/f'{tensor}.f32')
            result=subprocess.run([str(ROOT/'build/upsample48_prefix_test.exe'),str(prefix),str(width),'8'],
                                  capture_output=True,text=True,check=True,timeout=120)
            (prefix/'device.log').write_text(result.stdout+result.stderr)
            if (prefix/'merged_device.f32').read_bytes()!=(prefix/'merged.f32').read_bytes():
                raise AssertionError('prefix scalar/standalone differs')
            shutil.copyfile(prefix/'low.f32',frame/'low48.f32');shutil.copyfile(prefix/'merged_device.f32',frame/'merge48.f32')
            previous=prefix/'merged_device.f32';blocks=[]
            for block,shift in zip(range(48,56),SHIFTS):
                folder=frame/f'block{block}'
                previous=run_block(block,previous,width=2*width,height=16,output_root=folder)
                blocks.append((folder,shift))
            shutil.copyfile(previous,frame/'decoder55.f32')
            references[name]=dict(skip_sha256=digest(frame/'skip22.f32'),
                manifests=[dict(block=b,sha256=digest(p/'manifest.json')) for b,(p,_) in zip(range(48,56),blocks)])
            if name=='a':list_file(output/'c256_blocks.txt',blocks)
    finally:close_worker()
    if len({digest(output/f'references/{v}/decoder55.f32') for v in ('a','b','zero')})!=3:
        raise AssertionError('control pairs did not yield three different outputs')
    (output/'frames.txt').write_bytes(''.join(str(output/'references'/v)+'\n' for v in sequence).encode('mbcs'))
    for i in range(5):(output/f'frame_{i}').mkdir(exist_ok=True)
    command=[str(ROOT/'build/split512_frames_test.exe'),str(output/'fixed_control/blocks.txt'),str(width),'8',
             str(output/'frames.txt'),str(output),str(head),str(source/'bridge_map.i32'),str(source/'vit_blocks.txt'),
             str(entry),str(output/'decoder_control/blocks.txt'),str(output/'references/a/prefix48'),str(output/'c256_blocks.txt')]
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    (output/'device.log').write_text(result.stdout+result.stderr)
    if result.returncode or 'FAIL' in result.stdout:raise RuntimeError(f'resident C256 failed: {output}/device.log\n{result.stdout[-2500:]}{result.stderr}')
    metrics=json.loads((output/'frames_metrics.json').read_text());n=width*8*512;hn=width*2*1024
    if (metrics['frames']!=5 or metrics['invalid_views_rejected']!=16 or
            metrics['vit_stage_comparisons']!=80 or metrics['decoder39_stage_comparisons']!=5 or
            metrics['decoder_stage_comparisons']!=112 or
            metrics['prefix48_stage_comparisons']!=4 or metrics['c256_stage_comparisons']!=120 or
            not all(metrics[k] for k in ('resident_c256','resident_decoder','resident_vit','input_preserved','weights_loaded_once')) or
            metrics['frame_upload_bytes']!=5*3*n*4 or metrics['compute_d2d_bytes']!=5*(13*n+n//2+4*hn+hn//2)*4 or
            any(metrics[k] for k in ('post_setup_allocations','compute_h2d_bytes','compute_d2h_bytes')) or
            result.stdout.count('PASS')!=386):raise AssertionError('C256 residency/diagnostic contract differs')
    exact=[]
    for i,name in enumerate(sequence):
        hashes={}
        for tensor in NAMES:
            p=output/f'frame_{i}/{tensor}.f32'
            if p.read_bytes()!=(output/f'references/{name}/{tensor}.f32').read_bytes():raise AssertionError(f'endpoint differs: {i}/{tensor}')
            hashes[tensor]=digest(p)
        exact.append(dict(frame=name,outputs=hashes,input_sha256=digest(output/f'references/{name}/input.f32'),
                          skip22_sha256=digest(output/f'references/{name}/skip22.f32')))
    controls=[]
    with tempfile.TemporaryDirectory(prefix='c256 controls ',dir=output) as temporary:
        tmp=Path(temporary);bad=tmp/'bad_prefix';shutil.copytree(output/'references/a/prefix48',bad)
        v=np.fromfile(bad/'skip.f32','<f4');v[0]=123456;v.tofile(bad/'skip.f32')
        cmd=command.copy();cmd[5]=str(tmp);cmd[11]=str(bad)
        failed=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        (output/'rejected_skip.log').write_text(failed.stdout+failed.stderr)
        if failed.returncode!=1 or 'bad_prefix: skip: FAIL' not in failed.stdout:raise AssertionError('altered skip accepted')
        controls.append('altered skip22 expectation rejected')
        bad=tmp/'bad_block49';shutil.copytree(output/'references/a/block49',bad)
        v=np.fromfile(bad/'spatial/input.f32','<f4');v[0]=123456;v.tofile(bad/'spatial/input.f32')
        lines=(output/'c256_blocks.txt').read_text(encoding='mbcs').splitlines();lines[1]='3\t'+str(bad)
        (tmp/'blocks.txt').write_bytes(('\n'.join(lines)+'\n').encode('mbcs'))
        cmd[11]=command[11];cmd[12]=str(tmp/'blocks.txt')
        failed=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        (output/'rejected_handoff.log').write_text(failed.stdout+failed.stderr)
        if (failed.returncode!=1 or 'bad_block49/spatial: input: FAIL' not in failed.stdout or
                (tmp/'frames_metrics.json').exists() or list(tmp.glob('frame_*'))):raise AssertionError('altered C256 handoff accepted')
        controls.append('altered block49 input rejected before publishing outputs')
    report=dict(extent=[width,8,512],output_extent=[width*2,16,256],sequence=sequence,metrics=metrics,exact_outputs=exact,
        source_report_sha256=digest(source/'report.json'),skip_provenance=skip_provenance,
        paired_controls=paired,
        prefix_tensor_sha256=digest(raw_path),c256_references=references,negative_controls=controls,
        original_kernel_executed=False,production_wiring=False,original_c256_maps_validated=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(extent=report['extent'],exact_arrays=65,metrics=metrics,negative_controls=controls),indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('decoder47',type=Path)
    p.add_argument('--output-root',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--captured-candidate',type=Path);group.add_argument('--synthetic-skip',action='store_true')
    p.add_argument('--head-fixture',type=Path);a=p.parse_args()
    run(a.decoder47,a.output_root,a.captured_candidate,a.synthetic_skip,a.head_fixture)
