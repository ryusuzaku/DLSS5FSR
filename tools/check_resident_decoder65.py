"""Extend a prepared resident decoder61 replay through decoder62-65 and encoder8 skip.

The saved command supplies fixture paths only; this tool invokes the repository's
known executable without a shell. Original NVIDIA parity is not implied.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile
import numpy as np
from check_resident_decoder61 import NAMES as OLD_NAMES
from check_split512_resident import ROOT, digest, list_file
from check_block56_candidate import decode
from check_block62_candidate import run as run_block
from check_upsample48_prefix import bits, H, F, e4m3fn
from check_upsample62_peer_image import project_sequential_f32
from gpu_test_runner import close_worker

SHIFTS=(0,3,1,2)
NAMES=(*OLD_NAMES,'low62','merge62','decoder65')


def run(source, output, captured_candidate=None, synthetic_skip=False, skips=None, skips_provenance=None):
    source,output=(Path(p).resolve() for p in (source,output))
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('use separate input/output directories')
    prior=json.loads((source/'report.json').read_text())
    prepared=json.loads((source/'native_command.json').read_text())
    width,height,channels=prior['extent'];sequence=['a','b','zero','b','a']
    if (width not in (8,32) or height!=8 or channels!=512 or prior['sequence']!=sequence or
            not prior['metrics']['resident_c128'] or len(prior['exact_outputs'])!=5):
        raise ValueError('requires a validated resident decoder61 report')
    if (not isinstance(prepared,list) or len(prepared)!=15 or not all(isinstance(v,str) for v in prepared) or
            Path(prepared[0]).resolve()!=ROOT/'build/split512_frames_test.exe' or prepared[2:4]!=[str(width),'8'] or
            Path(prepared[4]).resolve()!=source/'frames.txt' or Path(prepared[5]).resolve()!=source or
            Path(prepared[13]).resolve()!=source/'references/a/prefix56' or Path(prepared[14]).resolve()!=source/'c128_blocks.txt'):
        raise ValueError('unsupported prepared decoder61 command')
    if bool(captured_candidate)==bool(synthetic_skip):raise ValueError('declare captured or synthetic skip')
    for name,record in zip(sequence,prior['exact_outputs']):
        frame=source/'references'/name
        if record['frame']!=name:raise ValueError('wrong saved sequence')
        for key in ('input','skip22','skip14'):
            if digest(frame/f'{key}.f32')!=record[f'{key}_sha256']:raise ValueError('saved input/skip changed')
        for key in OLD_NAMES:
            if digest(frame/f'{key}.f32')!=record['outputs'][key]:raise ValueError('saved endpoint changed')
    records={r['name']:r for r in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    prefix=source/'references/a/prefix56'
    raw_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{records['block56.layer0.layer']['index']:03d}.bin"
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=230176 or digest(raw_path)!=prior['prefix_tensor_sha256']:raise ValueError('block56 raw source changed')
    rows=bits(128*256,[3]+list(range(6,12)));cols=bits(128*256,[1,0,4,5,2]+list(range(12,15)))
    weights=np.empty((128,256),np.float32);weights[rows,cols]=e4m3fn(raw[0x18000:0x20000])
    c=np.arange(128);order=(c//16)*16+(c%8)*2+(c%16//8)
    scale=np.empty(128,np.float32);scale[order]=raw[0x20100:0x20200].view('<f2').astype(np.float32)
    if ((prefix/'weights.f32').read_bytes()!=weights.astype('<f4').tobytes() or
            (prefix/'scale.f32').read_bytes()!=scale.astype('<f4').tobytes() or
            (prefix/'input.f32').read_bytes()!=(source/'references/a/decoder55.f32').read_bytes() or
            (prefix/'skip.f32').read_bytes()!=(source/'references/a/skip14.f32').read_bytes()):
        raise ValueError('block56 weights or input handoff changed')
    manifests=prior['c128_references']['a']['manifests']
    if len(manifests)!=6:raise ValueError('incomplete C128 references')
    expected_lines=[];previous=prefix/'merged_device.f32'
    for block,record,shift in zip(range(56,62),manifests,(0,2,0,3,1,2)):
        folder=source/f'references/a/block{block}';m=json.loads((folder/'manifest.json').read_text())
        raw_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{records[f'block{block}.layer0.layer']['index']:03d}.bin"
        if (digest(folder/'manifest.json')!=record['sha256'] or m['tensor_sha256']!=digest(raw_path) or
                m['input_device_sha256']!=digest(previous) or (folder/'spatial/input.f32').read_bytes()!=previous.read_bytes()):
            raise ValueError('C128 source or handoff changed')
        values=decode(np.fromfile(raw_path,np.uint8))
        names=('ffn/w1','ffn/w2','ffn/w3','ffn/skip','attention/qkv_weights','attention/bias','attention/scales','attention/projection_weights','attention/attention_skip')
        for key,array in zip(names,values):
            if (folder/f'{key}.f32').read_bytes()!=np.asarray(array,'<f4').tobytes():raise ValueError(f'C128 weight changed: {block}/{key}')
        expected_lines.append(f'{shift}\t{folder}');previous=folder/'output/output_device.f32'
    if ((source/'c128_blocks.txt').read_text(encoding='mbcs').splitlines()!=expected_lines or
            previous.read_bytes()!=(source/'references/a/decoder61.f32').read_bytes()):raise ValueError('C128 chain changed')
    output.mkdir(parents=True,exist_ok=True)
    if captured_candidate:
        capture=Path(captured_candidate).resolve()
        if width!=8:raise ValueError('captured fixture requires width8')
        enc=json.loads((capture/'encoder22/report.json').read_text());dec=json.loads((capture/'decoder61/report.json').read_text())
        skip_path=capture/'encoder22/block8/output/output_device.f32'
        if (enc['source_capture_sha256']!=dec['source_capture_sha256'] or enc['color_linear_sha256']!=dec['color_linear_sha256'] or
                prior['skip_provenance'].get('capture_sha256')!=enc['source_capture_sha256'] or
                enc['blocks'][-1]['output_device_sha256']!=digest(skip_path) or
                dec['final_device_sha256']!=digest(source/'references/a/decoder61.f32')):raise ValueError('encoder8 does not match captured decoder61')
        skip_a=np.fromfile(skip_path,'<f4').reshape(64,64,64)
        provenance=dict(kind='same-capture AMD encoder8 output for A',sha256=digest(skip_path),capture_sha256=enc['source_capture_sha256'])
    else:
        skip_a=F(np.random.default_rng(6201).normal(0,.03125,(64,width*8,64)).astype(np.float32))
        provenance=dict(kind='synthetic FP8 Gaussian skip; seed6201, sigma0.03125')
    paired='B rolls skip8 eight columns; zero has zero skip; synthetic pairs, not full upstream executions'
    if skips is None:
        skips=dict(a=skip_a,b=np.roll(skip_a,8,axis=1).copy(),zero=np.zeros_like(skip_a))
    else:
        # Same-frame encoder8 outputs; a captured A must still equal the capture.
        if set(skips)!={'a','b','zero'} or not skips_provenance:raise ValueError('declared skips need a/b/zero and provenance')
        skips={k:np.asarray(v,np.float32).reshape(64,width*8,64) for k,v in skips.items()}
        if captured_candidate and skips['a'].tobytes()!=skip_a.tobytes():raise ValueError('declared skip A differs from capture')
        provenance=dict(provenance,declared=skips_provenance)
        paired='A/B/zero skips are same-frame encoder8 outputs of their declared C64 inputs'
    raw_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{records['block62.layer0.layer']['index']:03d}.bin"
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=70048:raise ValueError('wrong block62 tensor extent')
    rows=bits(64*128,[3]+list(range(6,11)));cols=bits(64*128,[1,0,4,5,2]+list(range(11,13)))
    if np.unique(rows*128+cols).size!=64*128:raise ValueError('projection map collides')
    weights=np.empty((64,128),np.float32);weights[rows,cols]=e4m3fn(raw[0x7000:0x9000])
    c=np.arange(64);order=(c//16)*16+(c%8)*2+(c%16//8)
    scale=np.empty(64,np.float32);scale[order]=raw[0x9080:0x9100].view('<f2').astype(np.float32)
    refs={}
    try:
        for name in ('a','b','zero'):
            frame=output/'references'/name;prefix=frame/'prefix62';prefix.mkdir(parents=True,exist_ok=True)
            for tensor in ('input','skip22','skip14',*OLD_NAMES):shutil.copyfile(source/f'references/{name}/{tensor}.f32',frame/f'{tensor}.f32')
            skip=skips[name];skip.astype('<f4').tofile(frame/'skip8.f32')
            x=np.fromfile(frame/'decoder61.f32','<f4').reshape(32,width*4,128)
            low=project_sequential_f32(x,weights);merged=F(H(np.repeat(np.repeat(low,2,0),2,1)+skip*scale))
            for key,array in dict(input=x,weights=weights,scale=scale,skip=skip,low=low,merged=merged).items():
                if not np.isfinite(array).all():raise ValueError('nonfinite prefix fixture')
                np.asarray(array,'<f4').tofile(prefix/f'{key}.f32')
            result=subprocess.run([str(ROOT/'build/upsample62_prefix_test.exe'),str(prefix),str(width*4),'32'],capture_output=True,text=True,check=True,timeout=120)
            (prefix/'device.log').write_text(result.stdout+result.stderr)
            if (prefix/'merged_device.f32').read_bytes()!=(prefix/'merged.f32').read_bytes():raise AssertionError('prefix reference differs')
            shutil.copyfile(prefix/'low.f32',frame/'low62.f32');shutil.copyfile(prefix/'merged_device.f32',frame/'merge62.f32')
            previous=prefix/'merged_device.f32';blocks=[]
            for block,shift in zip(range(62,66),SHIFTS):
                folder=frame/f'block{block}'
                previous=run_block(case='from_candidate_encoder8_fp8' if captured_candidate and name=='a' else 'seeded',
                    block=block,previous=previous,width=width*8,height=64,output_root=folder)
                blocks.append((folder,shift))
            shutil.copyfile(previous,frame/'decoder65.f32')
            refs[name]=dict(skip_sha256=digest(frame/'skip8.f32'),manifests=[dict(block=b,sha256=digest(p/'manifest.json')) for b,(p,_) in zip(range(62,66),blocks)])
            if name=='a':list_file(output/'c64_blocks.txt',blocks)
    finally:close_worker()
    if len({digest(output/f'references/{v}/decoder65.f32') for v in ('a','b','zero')})!=3:raise AssertionError('control outputs not distinct')
    (output/'frames.txt').write_bytes(''.join(str(output/'references'/v)+'\n' for v in sequence).encode('mbcs'))
    for i in range(5):(output/f'frame_{i}').mkdir(exist_ok=True)
    command=[str(ROOT/'build/split512_frames_test.exe'),*prepared[1:]]
    command[4]=str(output/'frames.txt');command[5]=str(output)
    command.extend([str(output/'references/a/prefix62'),str(output/'c64_blocks.txt')])
    # Record the exact prepared invocation for later component regression/replay.
    (output/'native_command.json').write_text(json.dumps(command,indent=2)+'\n')
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    (output/'device.log').write_text(result.stdout+result.stderr)
    if result.returncode or 'FAIL' in result.stdout:raise RuntimeError(f'resident C64 failed: {output}/device.log\n{result.stdout[-2500:]}{result.stderr}')
    m=json.loads((output/'frames_metrics.json').read_text());n=width*8*512;hn=width*2*1024
    expected=dict(frames=5,invalid_views_rejected=28,vit_stage_comparisons=80,decoder39_stage_comparisons=5,decoder_stage_comparisons=112,
        prefix48_stage_comparisons=4,c256_stage_comparisons=120,prefix56_stage_comparisons=4,c128_stage_comparisons=90,prefix62_stage_comparisons=4,c64_stage_comparisons=60,
        post_setup_allocations=0,compute_h2d_bytes=0,compute_d2h_bytes=0,frame_upload_bytes=5*15*n*4,compute_d2d_bytes=5*(52*n+n//2+4*hn+hn//2)*4)
    if (any(m[k]!=v for k,v in expected.items()) or result.stdout.count('PASS')!=574 or
            not all(m[k] for k in ('resident_c64','resident_c128','resident_c256','resident_decoder','resident_vit','input_preserved','weights_loaded_once'))):
        raise AssertionError('C64 residency/diagnostic contract differs')
    exact=[]
    for i,name in enumerate(sequence):
        hashes={}
        for tensor in NAMES:
            p=output/f'frame_{i}/{tensor}.f32'
            if p.read_bytes()!=(output/f'references/{name}/{tensor}.f32').read_bytes():raise AssertionError(f'endpoint differs: {i}/{tensor}')
            hashes[tensor]=digest(p)
        exact.append(dict(frame=name,outputs=hashes,input_sha256=digest(output/f'references/{name}/input.f32'),
            skip22_sha256=digest(output/f'references/{name}/skip22.f32'),skip14_sha256=digest(output/f'references/{name}/skip14.f32'),skip8_sha256=digest(output/f'references/{name}/skip8.f32')))
    controls=[]
    with tempfile.TemporaryDirectory(prefix='c64 controls ',dir=output) as temporary:
        tmp=Path(temporary);bad=tmp/'bad_prefix';shutil.copytree(output/'references/a/prefix62',bad)
        v=np.fromfile(bad/'skip.f32','<f4');v[0]=123456;v.tofile(bad/'skip.f32')
        cmd=command.copy();cmd[5]=str(tmp);cmd[15]=str(bad)
        failure=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        (output/'rejected_skip.log').write_text(failure.stdout+failure.stderr)
        if failure.returncode!=1 or 'bad_prefix: skip: FAIL' not in failure.stdout:raise AssertionError('altered skip8 accepted')
        controls.append('altered skip8 expectation rejected')
        bad=tmp/'bad_block63';shutil.copytree(output/'references/a/block63',bad)
        v=np.fromfile(bad/'spatial/input.f32','<f4');v[0]=123456;v.tofile(bad/'spatial/input.f32')
        lines=(output/'c64_blocks.txt').read_text(encoding='mbcs').splitlines();lines[1]='3\t'+str(bad)
        (tmp/'blocks.txt').write_bytes(('\n'.join(lines)+'\n').encode('mbcs'))
        cmd[15]=command[15];cmd[16]=str(tmp/'blocks.txt')
        failure=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        (output/'rejected_handoff.log').write_text(failure.stdout+failure.stderr)
        if (failure.returncode!=1 or 'bad_block63/spatial: input: FAIL' not in failure.stdout or
                (tmp/'frames_metrics.json').exists() or list(tmp.glob('frame_*'))):raise AssertionError('altered C64 handoff accepted')
        controls.append('altered block63 input rejected before publishing outputs')
    report=dict(extent=[width,8,512],output_extent=[width*8,64,64],sequence=sequence,metrics=m,exact_outputs=exact,
        source_report_sha256=digest(source/'report.json'),source_command_sha256=digest(source/'native_command.json'),
        skip_provenance=provenance,paired_controls=paired,
        prefix_tensor_sha256=digest(raw_path),c64_references=refs,negative_controls=controls,original_kernel_executed=False,production_wiring=False,
        c64_map_status='measured FFN/matrix/bias rules; candidate attention residual order')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(extent=report['extent'],exact_arrays=95,metrics=m,negative_controls=controls),indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('decoder61',type=Path)
    p.add_argument('--output-root',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--captured-candidate',type=Path);g.add_argument('--synthetic-skip',action='store_true')
    a=p.parse_args();run(a.decoder61,a.output_root,a.captured_candidate,a.synthetic_skip)
