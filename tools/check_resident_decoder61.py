"""Connect resident decoder56-61 with an explicitly supplied encoder14 skip."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile

import numpy as np
from check_resident_decoder55 import NAMES as OLD_NAMES, verify_files
from check_split512_resident import ROOT, digest, list_file, run as check_fixed
from check_decoder49_candidate import decode_ffn, candidate_ffn_maps, decode_attention, candidate_attention_maps
from check_block56_candidate import run as run_block
from check_upsample48_prefix import bits, multiply, H, F, e4m3fn
from audit_native_vit_logical_map import logical_map
from gpu_test_runner import close_worker

SHIFTS = (0, 2, 0, 3, 1, 2)
NAMES = (*OLD_NAMES, 'low56', 'merge56', 'decoder61')


def run(source, upstream, output, captured_candidate=None, synthetic_skip=False, head_fixture=None):
    source, upstream, output = (Path(p).resolve() for p in (source, upstream, output))
    for p in (source,upstream):
        if p==output or p in output.parents or output in p.parents:raise ValueError('use separate input/output directories')
    prior=json.loads((source/'report.json').read_text()); ancestor=json.loads((upstream/'report.json').read_text())
    width,height,channels=prior['extent']; sequence=['a','b','zero','b','a']
    if (width not in (8,32) or height!=8 or channels!=512 or prior['sequence']!=sequence or
            not prior['metrics']['resident_c256'] or prior['source_report_sha256']!=digest(upstream/'report.json')):
        raise ValueError('requires matching validated resident decoder55/decoder47 reports')
    if bool(captured_candidate)==bool(synthetic_skip):raise ValueError('declare exactly one captured candidate or synthetic skip')
    output.mkdir(parents=True,exist_ok=True)
    lines=(upstream/'fixed_control/blocks.txt').read_text(encoding='mbcs').splitlines()
    fixtures=Path(lines[0].split('\t')[1]).parent
    head=Path(head_fixture).resolve() if head_fixture else fixtures/f'block30_head_{width//2}x4'
    base=check_fixed(fixtures,output/'fixed_control',width=width,head_fixture=head,repeats=1)
    decoder=check_fixed(upstream/'references/a/decoder',output/'decoder_control',first=40,width=width,repeats=1)
    if base['sources']!=ancestor['c512_sources'] or decoder['sources']!=ancestor['decoder_sources']['a']:
        raise ValueError('C512 ancestry changed')
    if len(prior['exact_outputs'])!=5:raise ValueError('incomplete decoder55 result')
    for name,record in zip(sequence,prior['exact_outputs']):
        frame=source/'references'/name
        if record['frame']!=name or digest(frame/'input.f32')!=record['input_sha256'] or digest(frame/'skip22.f32')!=record['skip22_sha256']:
            raise ValueError('saved input/skip pair changed')
        for tensor in OLD_NAMES:
            if digest(frame/f'{tensor}.f32')!=record['outputs'][tensor]:raise ValueError(f'saved endpoint changed: {name}/{tensor}')
    vit_folders=[Path(v) for v in (upstream/'vit_blocks.txt').read_text(encoding='mbcs').splitlines()]
    if len(vit_folders)!=8:raise ValueError('wrong ViT block count')
    previous=source/'references/a/bridge.f32'
    for block,folder in zip(range(31,39),vit_folders):
        m=json.loads((folder/'manifest.json').read_text());verify_files(folder,m)
        if m['block']!=block or (folder/'input.f32').read_bytes()!=previous.read_bytes():raise ValueError('ViT handoff changed')
        for part in ('expansion','contraction','qkv','projection'):
            raw_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{m[part+'_tensor']:03d}.bin"
            if digest(raw_path)!=m[part+'_sha256']:raise ValueError('ViT raw source changed')
        previous=folder/'projection_device.f32'
    if previous.read_bytes()!=(source/'references/a/vit.f32').read_bytes():raise ValueError('ViT38 changed')
    entry=upstream/'references/a/entry39';m=json.loads((entry/'manifest.json').read_text());verify_files(entry,m)
    if digest(ROOT/'dlss5-analysis/tensors'/f"tensor_{m['tensor']:03d}.bin")!=m['tensor_sha256']:raise ValueError('decoder39 raw source changed')
    gather=logical_map(width*2).astype('<i4')
    if ((upstream/'bridge_map.i32').read_bytes()!=gather.tobytes() or
            (entry/'inverse.i32').read_bytes()!=np.argsort(gather).astype('<i4').tobytes()):raise ValueError('candidate bridge maps changed')
    records={r['name']:r for r in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    prefix48=source/'references/a/prefix48'
    prefix48_raw=ROOT/'dlss5-analysis/tensors'/f"tensor_{records['block48.layer0.layer']['index']:03d}.bin"
    raw48=np.fromfile(prefix48_raw,np.uint8)
    if raw48.size!=820784 or digest(prefix48_raw)!=prior['prefix_tensor_sha256']:raise ValueError('prefix48 raw source changed')
    r48=bits(256*512,[3]+list(range(6,13)));c48=bits(256*512,[1,0,4,5,2]+list(range(13,17)))
    w48=np.empty((256,512),np.float32);w48[r48,c48]=e4m3fn(raw48[0x58000:0x78000])
    ch48=np.arange(256);order48=(ch48//16)*16+(ch48%8)*2+(ch48%16//8)
    s48=np.empty(256,np.float32);s48[order48]=raw48[0x78200:0x78400].view('<f2').astype(np.float32)
    if ((prefix48/'weights.f32').read_bytes()!=w48.astype('<f4').tobytes() or
            (prefix48/'scale.f32').read_bytes()!=s48.astype('<f4').tobytes() or
            (prefix48/'input.f32').read_bytes()!=(source/'references/a/decoder47.f32').read_bytes() or
            (prefix48/'skip.f32').read_bytes()!=(source/'references/a/skip22.f32').read_bytes() or
            (prefix48/'merged_device.f32').read_bytes()!=(source/'references/a/merge48.f32').read_bytes()):
        raise ValueError('prefix48 weights or input/output handoffs changed')
    # Decode the actual C256 source weights again before trusting saved resident fixtures.
    c256_blocks=[]
    previous=source/'references/a/prefix48/merged_device.f32'
    for block,record in zip(range(48,56),prior['c256_references']['a']['manifests']):
        folder=source/f'references/a/block{block}';m=json.loads((folder/'manifest.json').read_text())
        raw_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{records[f'block{block}.layer0.layer']['index']:03d}.bin"
        if (digest(folder/'manifest.json')!=record['sha256'] or m['tensor_sha256']!=digest(raw_path) or
                m['input_device_sha256']!=digest(previous) or (folder/'spatial/input.f32').read_bytes()!=previous.read_bytes()):
            raise ValueError('C256 source or handoff changed')
        raw=np.fromfile(raw_path,np.uint8)
        values=(*decode_ffn(raw,candidate_ffn_maps()),*decode_attention(raw,candidate_attention_maps()))
        names=('ffn/w1','ffn/w2','ffn/w3','ffn/skip','attention/qkv_weights','attention/bias','attention/scales','attention/projection_weights','attention/attention_skip')
        for key,array in zip(names,values):
            if (folder/f'{key}.f32').read_bytes()!=np.asarray(array,'<f4').tobytes():raise ValueError(f'C256 weight changed: {block}/{key}')
        previous=folder/'output/output_device.f32';c256_blocks.append((folder,m['shift']))
    if len(c256_blocks)!=8 or previous.read_bytes()!=(source/'references/a/decoder55.f32').read_bytes():raise ValueError('C256 chain incomplete')
    list_file(output/'c256_blocks.txt',c256_blocks)
    if captured_candidate:
        capture=Path(captured_candidate).resolve()
        if width!=8:raise ValueError('captured fixture requires width8')
        enc=json.loads((capture/'encoder22/report.json').read_text());dec=json.loads((capture/'decoder55/report.json').read_text())
        skip_path=capture/'encoder22/block14/output/output_device.f32'
        if (enc['source_capture_sha256']!=dec['source_capture_sha256'] or enc['color_linear_sha256']!=dec['color_linear_sha256'] or
                prior['skip_provenance'].get('capture_sha256')!=enc['source_capture_sha256'] or
                enc['c128_blocks'][-1]['output_device_sha256']!=digest(skip_path) or
                dec['final_device_sha256']!=digest(source/'references/a/decoder55.f32')):raise ValueError('encoder14 does not match captured decoder55')
        skip_a=np.fromfile(skip_path,'<f4').reshape(32,32,128)
        provenance=dict(kind='same-capture AMD encoder14 output for A',sha256=digest(skip_path),capture_sha256=enc['source_capture_sha256'])
    else:
        skip_a=F(np.random.default_rng(5601).normal(0,.03125,(32,width*4,128)).astype(np.float32))
        provenance=dict(kind='synthetic FP8 Gaussian skip; seed5601, sigma0.03125')
    skips=dict(a=skip_a,b=np.roll(skip_a,4,axis=1).copy(),zero=np.zeros_like(skip_a))
    raw_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{records['block56.layer0.layer']['index']:03d}.bin"
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=230176:raise ValueError('wrong block56 tensor extent')
    rows=bits(128*256,[3]+list(range(6,12)));cols=bits(128*256,[1,0,4,5,2]+list(range(12,15)))
    if np.unique(rows*256+cols).size!=128*256:raise ValueError('projection map collides')
    weights=np.empty((128,256),np.float32);weights[rows,cols]=e4m3fn(raw[0x18000:0x20000])
    c=np.arange(128);order=(c//16)*16+(c%8)*2+(c%16//8)
    scale=np.empty(128,np.float32);scale[order]=raw[0x20100:0x20200].view('<f2').astype(np.float32)
    refs={}
    try:
        for name in ('a','b','zero'):
            frame=output/'references'/name;prefix=frame/'prefix56';prefix.mkdir(parents=True,exist_ok=True)
            for tensor in ('input','skip22',*OLD_NAMES):shutil.copyfile(source/f'references/{name}/{tensor}.f32',frame/f'{tensor}.f32')
            skip=skips[name];skip.astype('<f4').tofile(frame/'skip14.f32')
            x=np.fromfile(frame/'decoder55.f32','<f4').reshape(16,width*2,256)
            low=multiply(x,weights);merged=F(H(np.repeat(np.repeat(low,2,0),2,1)+skip*scale))
            for key,array in dict(input=x,weights=weights,scale=scale,skip=skip,low=low,merged=merged).items():
                if not np.isfinite(array).all():raise ValueError('nonfinite prefix fixture')
                np.asarray(array,'<f4').tofile(prefix/f'{key}.f32')
            result=subprocess.run([str(ROOT/'build/upsample56_prefix_test.exe'),str(prefix),str(width*2),'16'],capture_output=True,text=True,check=True,timeout=120)
            (prefix/'device.log').write_text(result.stdout+result.stderr)
            if (prefix/'merged_device.f32').read_bytes()!=(prefix/'merged.f32').read_bytes():raise AssertionError('prefix reference differs')
            shutil.copyfile(prefix/'low.f32',frame/'low56.f32');shutil.copyfile(prefix/'merged_device.f32',frame/'merge56.f32')
            previous=prefix/'merged_device.f32';blocks=[]
            for block,shift in zip(range(56,62),SHIFTS):
                folder=frame/f'block{block}'
                previous=run_block(case='image_candidate_encoder14' if captured_candidate and name=='a' else 'seeded',
                    block=block,previous=previous,width=width*4,height=32,output_root=folder)
                blocks.append((folder,shift))
            shutil.copyfile(previous,frame/'decoder61.f32')
            refs[name]=dict(skip_sha256=digest(frame/'skip14.f32'),manifests=[dict(block=b,sha256=digest(p/'manifest.json')) for b,(p,_) in zip(range(56,62),blocks)])
            if name=='a':list_file(output/'c128_blocks.txt',blocks)
    finally:close_worker()
    if len({digest(output/f'references/{v}/decoder61.f32') for v in ('a','b','zero')})!=3:raise AssertionError('control outputs not distinct')
    (output/'frames.txt').write_bytes(''.join(str(output/'references'/v)+'\n' for v in sequence).encode('mbcs'))
    for i in range(5):(output/f'frame_{i}').mkdir(exist_ok=True)
    command=[str(ROOT/'build/split512_frames_test.exe'),str(output/'fixed_control/blocks.txt'),str(width),'8',str(output/'frames.txt'),str(output),str(head),
        str(upstream/'bridge_map.i32'),str(upstream/'vit_blocks.txt'),str(entry),str(output/'decoder_control/blocks.txt'),
        str(source/'references/a/prefix48'),str(output/'c256_blocks.txt'),str(output/'references/a/prefix56'),str(output/'c128_blocks.txt')]
    # Record the exact prepared invocation for later component regression/replay.
    (output/'native_command.json').write_text(json.dumps(command,indent=2)+'\n')
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    (output/'device.log').write_text(result.stdout+result.stderr)
    if result.returncode or 'FAIL' in result.stdout:raise RuntimeError(f'resident C128 failed: {output}/device.log\n{result.stdout[-2500:]}{result.stderr}')
    m=json.loads((output/'frames_metrics.json').read_text());n=width*8*512;hn=width*2*1024
    expected=dict(frames=5,invalid_views_rejected=22,vit_stage_comparisons=80,decoder39_stage_comparisons=5,decoder_stage_comparisons=112,
        prefix48_stage_comparisons=4,c256_stage_comparisons=120,prefix56_stage_comparisons=4,c128_stage_comparisons=90,
        post_setup_allocations=0,compute_h2d_bytes=0,compute_d2h_bytes=0,frame_upload_bytes=5*7*n*4,compute_d2d_bytes=5*(26*n+n//2+4*hn+hn//2)*4)
    if (any(m[k]!=v for k,v in expected.items()) or result.stdout.count('PASS')!=495 or
            not all(m[k] for k in ('resident_c128','resident_c256','resident_decoder','resident_vit','input_preserved','weights_loaded_once'))):
        raise AssertionError('C128 residency/diagnostic contract differs')
    exact=[]
    for i,name in enumerate(sequence):
        hashes={}
        for tensor in NAMES:
            p=output/f'frame_{i}/{tensor}.f32'
            if p.read_bytes()!=(output/f'references/{name}/{tensor}.f32').read_bytes():raise AssertionError(f'endpoint differs: {i}/{tensor}')
            hashes[tensor]=digest(p)
        exact.append(dict(frame=name,outputs=hashes,input_sha256=digest(output/f'references/{name}/input.f32'),
            skip22_sha256=digest(output/f'references/{name}/skip22.f32'),skip14_sha256=digest(output/f'references/{name}/skip14.f32')))
    controls=[]
    with tempfile.TemporaryDirectory(prefix='c128 controls ',dir=output) as temporary:
        tmp=Path(temporary);bad=tmp/'bad_prefix';shutil.copytree(output/'references/a/prefix56',bad)
        v=np.fromfile(bad/'skip.f32','<f4');v[0]=123456;v.tofile(bad/'skip.f32')
        cmd=command.copy();cmd[5]=str(tmp);cmd[13]=str(bad)
        failure=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        (output/'rejected_skip.log').write_text(failure.stdout+failure.stderr)
        if failure.returncode!=1 or 'bad_prefix: skip: FAIL' not in failure.stdout:raise AssertionError('altered skip14 accepted')
        controls.append('altered skip14 expectation rejected')
        bad=tmp/'bad_block57';shutil.copytree(output/'references/a/block57',bad)
        v=np.fromfile(bad/'spatial/input.f32','<f4');v[0]=123456;v.tofile(bad/'spatial/input.f32')
        lines=(output/'c128_blocks.txt').read_text(encoding='mbcs').splitlines();lines[1]='2\t'+str(bad)
        (tmp/'blocks.txt').write_bytes(('\n'.join(lines)+'\n').encode('mbcs'))
        cmd[13]=command[13];cmd[14]=str(tmp/'blocks.txt')
        failure=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        (output/'rejected_handoff.log').write_text(failure.stdout+failure.stderr)
        if (failure.returncode!=1 or 'bad_block57/spatial: input: FAIL' not in failure.stdout or
                (tmp/'frames_metrics.json').exists() or list(tmp.glob('frame_*'))):raise AssertionError('altered C128 handoff accepted')
        controls.append('altered block57 input rejected before publishing outputs')
    report=dict(extent=[width,8,512],output_extent=[width*4,32,128],sequence=sequence,metrics=m,exact_outputs=exact,
        source_report_sha256=digest(source/'report.json'),upstream_report_sha256=digest(upstream/'report.json'),
        skip_provenance=provenance,paired_controls='B rolls skip14 four columns; zero has zero skip; synthetic pairs, not full upstream executions',
        prefix_tensor_sha256=digest(raw_path),c128_references=refs,negative_controls=controls,original_kernel_executed=False,production_wiring=False,
        c128_map_status='measured FFN/matrix/bias rules; candidate attention residual order')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(extent=report['extent'],exact_arrays=80,metrics=m,negative_controls=controls),indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('decoder55',type=Path)
    p.add_argument('--decoder47-source',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--captured-candidate',type=Path);g.add_argument('--synthetic-skip',action='store_true')
    p.add_argument('--head-fixture',type=Path);a=p.parse_args()
    run(a.decoder55,a.decoder47_source,a.output_root,a.captured_candidate,a.synthetic_skip,a.head_fixture)
