"""Connect prepared resident decoder65 through C32 decoder66-69 and head70.

Public block4/preblock0 skips remain external. This validates candidate
arithmetic and GPU residency, not original NVIDIA parity or game integration.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile
import numpy as np
from check_resident_decoder65 import NAMES as OLD_NAMES
from check_split512_resident import ROOT, digest, list_file
from check_block62_candidate import decode as decode64
from check_block66_peer_candidate import decode, run as run_block, save
from check_upsample48_prefix import bits, H, F, e4m3fn
from audit_peer_native_c32_basis import peer_to_native, run as audit_basis
from head70_normalized_reference import trace, packed
from head70_reference import merge, windowise, dewindowise, finish
from head70_weights import extract, repack, PAD_AT
from gpu_test_runner import close_worker, run as run_gpu_test

SHIFTS=(0,3,1,2)
NAMES=(*OLD_NAMES,'low66','merge66','decoder69','merged70','peer70','body70','native70','rgb_native','rgb_public')
STAGES=('expanded','hidden','ffn','qkv','qknorm','scores','exp','den','prob','context','body','output')


def device(command,log,passes):
    result=run_gpu_test([str(v) for v in command],cwd=ROOT,capture_output=True,check=False)
    log.write_text(result.stdout+(result.stderr or ''))
    if result.returncode or 'FAIL' in result.stdout or result.stdout.count('PASS')!=passes:
        raise RuntimeError(f'device reference failed: {log}\n{result.stdout[-2000:]}')


def project(x,weights):
    # Preserve the prefix kernel's sequential non-FMA sum and half boundaries.
    acc=np.zeros((*x.shape[:-1],32),np.float32)
    for base in (0,32):
        part=np.zeros_like(acc)
        for channel in range(base,base+32):
            part=np.add(part,np.multiply(x[...,channel,None],weights[:,channel],dtype=np.float32),dtype=np.float32)
        acc=H(np.add(acc,part,dtype=np.float32))
    return acc


def validate_source(source):
    prior=json.loads((source/'report.json').read_text());prepared=json.loads((source/'native_command.json').read_text())
    width,height,channels=prior['extent'];sequence=['a','b','zero','b','a']
    if (width not in (8,32) or height!=8 or channels!=512 or prior['sequence']!=sequence or
            not prior['metrics']['resident_c64'] or len(prior['exact_outputs'])!=5):
        raise ValueError('requires validated resident decoder65')
    if (not isinstance(prepared,list) or len(prepared)!=17 or not all(isinstance(v,str) for v in prepared) or
            Path(prepared[0]).resolve()!=ROOT/'build/split512_frames_test.exe' or prepared[2:4]!=[str(width),'8'] or
            Path(prepared[4]).resolve()!=source/'frames.txt' or Path(prepared[5]).resolve()!=source or
            Path(prepared[15]).resolve()!=source/'references/a/prefix62' or Path(prepared[16]).resolve()!=source/'c64_blocks.txt'):
        raise ValueError('unsupported prepared decoder65 invocation')
    for name,record in zip(sequence,prior['exact_outputs']):
        frame=source/'references'/name
        if record['frame']!=name:raise ValueError('sequence differs')
        for key in ('input','skip22','skip14','skip8'):
            if digest(frame/f'{key}.f32')!=record[f'{key}_sha256']:raise ValueError('saved input/skip changed')
        for key in OLD_NAMES:
            if digest(frame/f'{key}.f32')!=record['outputs'][key]:raise ValueError('saved endpoint changed')
    records={v['name']:v['index'] for v in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    prefix=source/'references/a/prefix62'
    raw_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{records['block62.layer0.layer']:03d}.bin"
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=70048 or digest(raw_path)!=prior['prefix_tensor_sha256']:raise ValueError('block62 source changed')
    rows=bits(8192,[3]+list(range(6,11)));cols=bits(8192,[1,0,4,5,2]+list(range(11,13)))
    weights=np.empty((64,128),np.float32);weights[rows,cols]=e4m3fn(raw[0x7000:0x9000])
    c=np.arange(64);order=(c//16)*16+(c%8)*2+(c%16//8)
    scale=np.empty(64,np.float32);scale[order]=raw[0x9080:0x9100].view('<f2').astype(np.float32)
    if ((prefix/'weights.f32').read_bytes()!=weights.astype('<f4').tobytes() or
            (prefix/'scale.f32').read_bytes()!=scale.astype('<f4').tobytes() or
            (prefix/'input.f32').read_bytes()!=(source/'references/a/decoder61.f32').read_bytes() or
            (prefix/'skip.f32').read_bytes()!=(source/'references/a/skip8.f32').read_bytes()):
        raise ValueError('block62 weights or handoff changed')
    manifests=prior['c64_references']['a']['manifests']
    if len(manifests)!=4:raise ValueError('incomplete C64 references')
    lines=[];previous=prefix/'merged_device.f32'
    for block,record,shift in zip(range(62,66),manifests,SHIFTS):
        folder=source/f'references/a/block{block}';m=json.loads((folder/'manifest.json').read_text())
        raw_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{records[f'block{block}.layer0.layer']:03d}.bin"
        if (digest(folder/'manifest.json')!=record['sha256'] or m['tensor_sha256']!=digest(raw_path) or
                m['input_device_sha256']!=digest(previous) or (folder/'spatial/input.f32').read_bytes()!=previous.read_bytes()):
            raise ValueError('C64 source or handoff changed')
        keys=('ffn/w1','ffn/w2','ffn/w3','ffn/skip','attention/qkv_weights','attention/bias','attention/scales','attention/projection_weights','attention/attention_skip')
        for key,array in zip(keys,decode64(np.fromfile(raw_path,np.uint8))):
            if (folder/f'{key}.f32').read_bytes()!=np.asarray(array,'<f4').tobytes():raise ValueError('C64 weight changed')
        lines.append(f'{shift}\t{folder}');previous=folder/'output/output_device.f32'
    if ((source/'c64_blocks.txt').read_text(encoding='mbcs').splitlines()!=lines or
            previous.read_bytes()!=(source/'references/a/decoder65.f32').read_bytes()):raise ValueError('C64 chain changed')
    return prior,prepared,width


def head_reference(frame,main,skip,color,head_weights,sm,ss,coeff):
    head=frame/'head70';body=head/'body';body.mkdir(parents=True,exist_ok=True)
    height,width=color.shape[:2];mapping=peer_to_native(np.arange(32))
    merged=windowise(merge(main,skip,sm,ss));peer=merged[...,mapping]
    save(head,dict(main=main,skip=skip,color=color,sm=sm,ss=ss,coeff=coeff,merged=merged,peer=peer))
    packed_weights=packed(head_weights);packed_weights.tofile(body/'weights.f32')
    for key in STAGES:(body/f'{key}.f32').write_bytes(b'')
    # Bounded scalar/standalone chunks, assembled in window order for resident checks.
    chunk=head/'chunk'
    for start in range(0,len(peer),32):
        x=peer[start:start+32];stages=trace(x,head_weights);stages['output']=F(stages['body'])
        save(chunk,dict(input=x,weights=packed_weights,**stages))
        device([ROOT/'build/c32_peer_body_test.exe',chunk,len(x)],chunk/'device.log',12)
        if (chunk/'body_device.f32').read_bytes()!=(chunk/'body.f32').read_bytes():raise AssertionError('head raw body differs')
        for key in STAGES:
            with (body/f'{key}.f32').open('ab') as f:f.write(np.asarray(stages[key],'<f4').tobytes())
    raw_body=np.fromfile(body/'body.f32','<f4').reshape(-1,64,32)
    native=np.empty_like(raw_body);native[...,mapping]=raw_body
    features=dewindowise(native,height,width)
    native_rgb=finish(features,color,coeff,.03125);public_rgb=finish(features,color,coeff,1.)
    save(head,dict(native=native,rgb_native=native_rgb,rgb_public=public_rgb,body=raw_body,weights=packed_weights))
    device([ROOT/'build/head70_peer_frame_chain.exe',head,width,height],head/'device.log',4)
    for key,path in dict(merged70=head/'merged.f32',peer70=head/'peer.f32',body70=body/'body.f32',
                         native70=head/'native.f32',rgb_native=head/'rgb_native.f32',rgb_public=head/'rgb_public.f32').items():
        shutil.copyfile(path,frame/f'{key}.f32')


def run(source,output,captured_candidate=None,synthetic_skip=False):
    source,output=(Path(v).resolve() for v in (source,output))
    if source==output or source in output.parents or output in source.parents:raise ValueError('use separate directories')
    if bool(captured_candidate)==bool(synthetic_skip):raise ValueError('declare captured or synthetic skips')
    prior,prepared,width=validate_source(source);audit_basis(verbose=False)
    mapping=peer_to_native(np.arange(32));sequence=['a','b','zero','b','a']
    if captured_candidate:
        if width!=8:raise ValueError('captured source requires width8')
        capture=Path(captured_candidate).resolve();dec=json.loads((capture/'decoder69/report.json').read_text())
        hi=capture/'head_inputs';head_info=json.loads((hi/'manifest.json').read_text())
        skip4_path=capture/'decoder69/public_boundary/skip4_peer.f32'
        if (dec['source_capture_sha256']!=prior['skip_provenance'].get('capture_sha256') or
                dec['source_block65_device_sha256']!=digest(source/'references/a/decoder65.f32') or
                dec['public_boundary_sha256']['skip4']!=digest(skip4_path) or
                head_info['decoder69_report_sha256']!=digest(capture/'decoder69/report.json') or
                head_info['source_capture_sha256']!=dec['source_capture_sha256'] or
                head_info['color_linear_sha256']!=dec['color_linear_sha256'] or not head_info['same_inference_call']):
            raise ValueError('public skip/head sources do not match captured decoder65')
        for key in ('skip_peer','color_linear'):
            if digest(hi/f'{key}.f32')!=head_info[f'{key}_sha256']:raise ValueError('head input changed')
        peer4=F(np.fromfile(skip4_path,'<f4').reshape(128,128,32));skip4=np.empty_like(peer4);skip4[...,mapping]=peer4
        peer0=np.fromfile(hi/'skip_peer.f32','<f4').reshape(256,256,32);skip0=np.empty_like(peer0);skip0[...,mapping]=peer0
        color=np.fromfile(hi/'color_linear.f32','<f4').reshape(256,256,3)
        provenance=dict(kind='same-capture public ONNX block4 FP8 skip, preblock0 skip and prepared RGB',
                        capture_sha256=dec['source_capture_sha256'],public_skip4_sha256=digest(skip4_path),
                        public_skip0_sha256=digest(hi/'skip_peer.f32'),color_sha256=digest(hi/'color_linear.f32'))
    else:
        rng=np.random.default_rng(6670)
        skip4=F(rng.normal(0,.03125,(128,width*16,32)).astype(np.float32))
        skip0=H(rng.normal(0,.03125,(256,width*32,32)).astype(np.float32))
        color=rng.uniform(0,1,(256,width*32,3)).astype(np.float32)
        provenance=dict(kind='synthetic seed6670; FP8 block4 and FP16 preblock0 Gaussian sigma0.03125; uniform RGB')
    inputs=dict(a=(skip4,skip0,color),b=(np.roll(skip4,16,axis=1),np.roll(skip0,32,axis=1),np.roll(color,32,axis=1)),
                zero=(np.zeros_like(skip4),np.zeros_like(skip0),np.zeros_like(color)))
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_144.bin';raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=22784:raise ValueError('wrong block66 record')
    matrix=np.empty((32,64),np.float32);matrix[bits(2048,[3,6,7,8,9]),bits(2048,[1,0,4,5,2,10])]=e4m3fn(raw[0x2000:0x2800])
    order=np.array([0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31])
    c=np.arange(32);other=(c//16)*16+(c%8)*2+(c%16//8)
    weights=np.empty_like(matrix);weights[order]=matrix[other]
    scale=np.empty(32,np.float32);scale[order]=raw[0x2860:0x28a0].view('<f2').astype(np.float32)
    raw_head=(ROOT/'dlss5-analysis/tensors/tensor_150.bin').read_bytes();stage=(ROOT/'dlss5-analysis/tensors/tensor_001.bin').read_bytes()
    ordinary,sm,ss,coeff,pad=extract(raw_head,stage[PAD_AT:PAD_AT+16])
    if repack(ordinary,sm,ss,coeff,pad)!=raw_head:raise AssertionError('head extraction roundtrip differs')
    head_weights=decode(np.frombuffer(ordinary,np.uint8));coeff=coeff[[0,2,4]]
    output.mkdir(parents=True,exist_ok=True);refs={}
    try:
        for name in ('a','b','zero'):
            frame=output/'references'/name;prefix=frame/'prefix66';prefix.mkdir(parents=True,exist_ok=True)
            for key in ('input','skip22','skip14','skip8',*OLD_NAMES):shutil.copyfile(source/f'references/{name}/{key}.f32',frame/f'{key}.f32')
            skip4,skip0,color=inputs[name];save(frame,dict(skip4=skip4,skip0=skip0,color=color))
            x=np.fromfile(frame/'decoder65.f32','<f4').reshape(64,width*8,64)
            low=project(x,weights);merged=H(np.repeat(np.repeat(low,2,0),2,1)+skip4*scale)
            save(prefix,dict(input=x,weights=weights,scale=scale,skip=skip4,low=low,merged=merged))
            device([ROOT/'build/upsample66_prefix_test.exe',prefix,width*8,64],prefix/'device.log',2)
            if (prefix/'merged_device.f32').read_bytes()!=(prefix/'merged.f32').read_bytes():raise AssertionError('prefix reference differs')
            shutil.copyfile(prefix/'low.f32',frame/'low66.f32');shutil.copyfile(prefix/'merged_device.f32',frame/'merge66.f32')
            previous=prefix/'merged_device.f32';blocks=[]
            for block,shift in zip(range(66,70),SHIFTS):
                folder=frame/f'block{block}'
                previous=run_block(block=block,case='from_candidate_encoder8_fp8' if captured_candidate and name=='a' else 'seeded',
                                   previous=previous,width=width*16,height=128,output_root=folder)
                blocks.append((folder,shift))
            shutil.copyfile(previous,frame/'decoder69.f32')
            head_reference(frame,np.fromfile(previous,'<f4').reshape(128,width*16,32),skip0,color,head_weights,sm,ss,coeff)
            refs[name]=dict(manifests=[dict(block=b,sha256=digest(p/'manifest.json')) for b,(p,_) in zip(range(66,70),blocks)])
            if name=='a':list_file(output/'c32_blocks.txt',blocks)
            print(f'{name}: independent decoder/head references complete',flush=True)
    finally:close_worker()
    for key in ('decoder69','rgb_native'):
        if len({digest(output/f'references/{v}/{key}.f32') for v in ('a','b','zero')})!=3:raise AssertionError('control outputs not distinct')
    (output/'frames.txt').write_bytes(''.join(str(output/'references'/v)+'\n' for v in sequence).encode('mbcs'))
    for i in range(5):(output/f'frame_{i}').mkdir(exist_ok=True)
    command=[str(ROOT/'build/split512_frames_test.exe'),*prepared[1:]];command[4]=str(output/'frames.txt');command[5]=str(output)
    command.extend([str(output/'references/a/prefix66'),str(output/'c32_blocks.txt'),str(output/'references/a/head70')])
    (output/'native_command.json').write_text(json.dumps(command,indent=2)+'\n')
    result=subprocess.run(command,capture_output=True,text=True,timeout=240)
    (output/'device.log').write_text(result.stdout+result.stderr)
    if result.returncode or 'FAIL' in result.stdout:raise RuntimeError(f'resident head failed: {output}/device.log\n{result.stdout[-3000:]}{result.stderr}')
    m=json.loads((output/'frames_metrics.json').read_text());n=width*8*512;hn=n//2
    expected=dict(frames=5,invalid_views_rejected=40,vit_stage_comparisons=80,decoder39_stage_comparisons=5,decoder_stage_comparisons=112,
                  prefix48_stage_comparisons=4,c256_stage_comparisons=120,prefix56_stage_comparisons=4,c128_stage_comparisons=90,
                  prefix62_stage_comparisons=4,c64_stage_comparisons=60,prefix66_stage_comparisons=4,c32_stage_comparisons=60,head70_stage_comparisons=20,
                  post_setup_allocations=0,compute_h2d_bytes=0,compute_d2h_bytes=0,frame_upload_bytes=5*101*n*4,
                  compute_d2d_bytes=5*(372*n+n//2+4*hn+hn//2)*4)
    if (any(m[k]!=v for k,v in expected.items()) or result.stdout.count('PASS')!=703 or
            not all(m[k] for k in ('resident_head70','resident_c32','resident_c64','resident_c128','resident_c256','resident_decoder','resident_vit','input_preserved','weights_loaded_once'))):
        raise AssertionError('resident diagnostic/transfer contract differs')
    exact=[]
    for i,name in enumerate(sequence):
        hashes={}
        for key in NAMES:
            path=output/f'frame_{i}/{key}.f32'
            if path.read_bytes()!=(output/f'references/{name}/{key}.f32').read_bytes():raise AssertionError(f'endpoint differs: {i}/{key}')
            hashes[key]=digest(path)
        record=dict(frame=name,outputs=hashes)
        for key in ('input','skip22','skip14','skip8','skip4','skip0','color'):record[f'{key}_sha256']=digest(output/f'references/{name}/{key}.f32')
        exact.append(record)
    controls=[]
    with tempfile.TemporaryDirectory(prefix='head70 controls ',dir=output) as temporary:
        tmp=Path(temporary)
        for label,folder,relative,index,expected_text in (
            ('skip4','prefix66','skip.f32',17,'bad_skip4: skip: FAIL'),
            ('block67','block67','spatial/input.f32',18,'bad_block67/spatial: input: FAIL'),
            ('skip0','head70','skip.f32',19,'bad_skip0: skip: FAIL'),
            ('color','head70','color.f32',19,'bad_color: color: FAIL')):
            bad=tmp/f'bad_{label}';bad.mkdir();original=output/'references/a'/folder
            # Hard-link immutable expectations; replace only the file being corrupted.
            shutil.copytree(original,bad,dirs_exist_ok=True,copy_function=lambda a,b:Path(b).hardlink_to(a))
            p=bad/relative;values=np.fromfile(p,'<f4');values[0]=123456;p.unlink();values.tofile(p)
            cmd=command.copy();cmd[5]=str(tmp);cmd[index]=str(bad)
            if label=='block67':
                lines=(output/'c32_blocks.txt').read_text(encoding='mbcs').splitlines();lines[1]='3\t'+str(bad)
                (tmp/'blocks.txt').write_bytes(('\n'.join(lines)+'\n').encode('mbcs'));cmd[18]=str(tmp/'blocks.txt')
            failure=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
            (output/f'rejected_{label}.log').write_text(failure.stdout+failure.stderr)
            if (failure.returncode!=1 or expected_text not in failure.stdout or (tmp/'frames_metrics.json').exists() or list(tmp.glob('frame_*'))):
                raise AssertionError(f'altered {label} accepted')
            controls.append(f'altered {label} expectation rejected before publishing outputs')
    report=dict(extent=[width,8,512],output_extent=[width*32,256,3],sequence=sequence,metrics=m,exact_outputs=exact,
                source_report_sha256=digest(source/'report.json'),source_command_sha256=digest(source/'native_command.json'),skip_provenance=provenance,
                paired_controls='B rolls skip4 by16 and skip0/RGB by32 columns; zero zeros all; synthetic pairs, not upstream inference',
                prefix_tensor_sha256=digest(raw_path),head_tensor_sha256=digest(ROOT/'dlss5-analysis/tensors/tensor_150.bin'),
                c32_references=refs,negative_controls=controls,original_kernel_executed=False,production_wiring=False,
                c32_map_status='public QMMA candidate in audited transition basis; original C32 body maps unverified')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(extent=report['extent'],exact_arrays=140,metrics=m,negative_controls=controls),indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('decoder65',type=Path);p.add_argument('--output-root',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--captured-candidate',type=Path);g.add_argument('--synthetic-skip',action='store_true')
    a=p.parse_args();run(a.decoder65,a.output_root,a.captured_candidate,a.synthetic_skip)
