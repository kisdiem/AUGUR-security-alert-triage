"""Frozen FP32 text encoding on guarded physical GPU1, with durable prefix resume.
No training, no changed original vector, no access to test labels or metrics.
"""
import argparse,hashlib,json,os,pathlib,signal,sys,time
sys.dont_write_bytecode=True
STOP=False
def interrupted(signum,frame):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
def fsync_directory(p):
    fd=os.open(p,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def durable_json(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.write_tmp')
    with tmp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path);fsync_directory(path.parent)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='<RESEARCH_ROOT>/reproductions/security_completion_20261026_v1');ap.add_argument('--cpu',action='store_true');ap.add_argument('--probe-only',action='store_true');a=ap.parse_args()
    root=pathlib.Path(a.root).resolve()
    from gpu_resource_guard import pin_environment,choose_device,check_running,ReserveViolation
    pin_environment(root)
    import numpy as np,torch
    from transformers import AutoModel,AutoTokenizer
    from rotation_features import RotationFeatures,SCENES,read,load,sha
    r=RotationFeatures(root,pool_only=True);bank=r.bank
    for scene in SCENES:
        assert read(root/'cache/scenes'/scene/'features_complete.json')['complete']
        cache=load(root/'cache/scenes'/scene/'rows.pkl.gz')
        for sid,parts in cache['rows'].items():
            for part in (cache['native_rows'][sid],parts['norm'],parts['cf']):
                if part:
                    for signature in part['signatures']:bank.index(signature)
        del cache
    bank.register_texts();texts_sha=sha(bank.out/'extra_texts.json');total=len(bank.extra)
    old=0 if bank.extra_vectors is None else len(bank.extra_vectors);target=bank.out/'extra_vectors.npy'
    prior=read(bank.out/'encoding_complete.json') if (bank.out/'encoding_complete.json').exists() else None
    if old:
        assert prior and prior['vectors_sha256']==sha(target),'Existing sealed vector hash mismatch'
        prefix_sha=hashlib.sha256(json.dumps(bank.extra[:old],ensure_ascii=False,indent=2).encode()).hexdigest()
        assert prior['texts_sha256']==prefix_sha,'Existing sealed text prefix mismatch'
    if old==total and not a.probe_only:
        assert prior['texts_sha256']==texts_sha
        print(json.dumps({'already_complete':True,'rows':old,'vectors_sha256':sha(target)}),flush=True);return 0
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    tokenizer=AutoTokenizer.from_pretrained(r.dep.Q/'model',local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(r.dep.Q/'model',local_files_only=True,trust_remote_code=False,use_safetensors=True).eval().requires_grad_(False)
    device=choose_device(root,'cpu' if a.cpu else 'cuda:0',2048)
    if device!='cpu':torch.cuda.set_per_process_memory_fraction(.08,0)
    def forward(texts,where):
        tokens=tokenizer(texts,padding=True,truncation=True,max_length=256,return_tensors='pt')
        tokens={k:v.to(where) for k,v in tokens.items()}
        with torch.inference_mode():
            h=model(**tokens).last_hidden_state;mask=tokens['attention_mask'][...,None]
            vectors=((h*mask).sum(1)/mask.sum(1).clamp_min(1)).float().cpu().numpy()
        assert np.isfinite(vectors).all();return vectors
    if a.probe_only:
        probe=bank.extra[old:old+32] or bank.extra[:32];begin=time.monotonic();cpu=forward(probe,'cpu');cpu_seconds=time.monotonic()-begin
        answer={'passed':True,'rows':len(probe),'requested_physical_gpu':1,'chosen_device':device,'GPU0_used':False,'cpu_seconds':cpu_seconds,'encoder_helper_sha256':sha(__file__),'test_labels_read':False}
        if device!='cpu':
            check_running(root);model=model.to(device);begin=time.monotonic();gpu=forward(probe,device);torch.cuda.synchronize();seconds=time.monotonic()-begin;check_running(root)
            answer.update(physical_gpu=1,gpu_name=torch.cuda.get_device_name(0),gpu_seconds=seconds,max_abs_difference=float(np.max(np.abs(cpu-gpu))),fp32_cpu_agreement=bool(np.allclose(cpu,gpu,rtol=2e-5,atol=2e-6)),peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
            assert answer['fp32_cpu_agreement'];model=model.cpu();torch.cuda.empty_cache()
        durable_json(root/'audits/encoder_GPU1_guarded_probe.json',answer);print(json.dumps(answer),flush=True);return 0
    tmp=target.with_name(target.name+'.encoder_tmp');checkpoint=bank.out/'encoding_checkpoint_v2.json'
    assert root in tmp.resolve().parents
    definition={'version':2,'texts_sha256':texts_sha,'rows_total':total,'sealed_prefix_rows':old,'sealed_prefix_vectors_sha256':sha(target) if old else None,'dim':384,'dtype':'float32','max_length':256,'pool':'attention-mask mean including special tokens; noL2'}
    if checkpoint.exists():
        state=read(checkpoint);assert state['definition']==definition,'Checkpoint text/vector definition changed'
        assert tmp.is_file(),'Committed prefix file missing';vectors=np.lib.format.open_memmap(tmp,mode='r+')
        assert vectors.shape==(total,384) and vectors.dtype==np.float32
        i=state['next_row'];assert old<=i<=total
        if old:assert np.array_equal(vectors[:old],bank.extra_vectors)
        end=old
        for chunk in state['chunks']:
            assert chunk['start']==end and chunk['end']<=i
            assert hashlib.sha256(vectors[chunk['start']:chunk['end']].tobytes()).hexdigest()==chunk['sha256'],'Committed prefix checksum mismatch'
            end=chunk['end']
        assert end==i;chunks=state['chunks'];segments=state.get('device_segments',[]);fallbacks=state.get('fallbacks',[])
    else:
        if tmp.exists():
            archive=bank.out/'unsealed_revisions';archive.mkdir(exist_ok=True);tmp.replace(archive/(tmp.name+'.'+str(time.time_ns())))
        vectors=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(total,384))
        if old:vectors[:old]=bank.extra_vectors
        i=old;chunks=[];segments=[];fallbacks=[]
    resumed_at=i;committed=i;begin=time.monotonic();last=begin;batch_size=32
    if device!='cpu':
        try:check_running(root);model=model.to(device)
        except (ReserveViolation,torch.cuda.OutOfMemoryError) as e:
            model=model.cpu();torch.cuda.empty_cache();device='cpu';fallbacks.append({'row':i,'reason':type(e).__name__+': '+str(e)})
    segment={'start':i,'device':device,'physical_gpu':1 if device!='cpu' else None,'batch_size':batch_size,'pid':os.getpid()}
    def seal(status):
        nonlocal committed,last
        vectors.flush();fd=os.open(tmp,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if i>committed:chunks.append({'start':committed,'end':i,'sha256':hashlib.sha256(vectors[committed:i].tobytes()).hexdigest()})
        committed=i;last=time.monotonic()
        durable_json(checkpoint,{'definition':definition,'next_row':i,'chunks':chunks,'device_segments':segments+[dict(segment,end=i)],'fallbacks':fallbacks,'status':status,'pid':os.getpid(),'time_unix':time.time()})
        progress={'rows':i,'total':total,'new_rows_this_run':i-resumed_at,'seconds':time.monotonic()-begin,'device':device,'physical_gpu':1 if device!='cpu' else None,'status':status,'checkpoint_sha256':sha(checkpoint)}
        durable_json(bank.out/'encoding_progress.json',progress);print(json.dumps(progress),flush=True)
    seal('running')
    while i<total and not STOP and not (root/'PAUSED_BY_USER').exists():
        if device!='cpu':
            try:check_running(root)
            except ReserveViolation as e:
                segments.append(dict(segment,end=i));model=model.cpu();torch.cuda.empty_cache();device='cpu';batch_size=32
                fallbacks.append({'row':i,'reason':str(e)});segment={'start':i,'device':'cpu','physical_gpu':None,'batch_size':32,'pid':os.getpid()}
                if (root/'PAUSED_BY_USER').exists():break
        try:
            batch=bank.extra[i:i+batch_size];v=forward(batch,device);vectors[i:i+len(batch)]=v;i+=len(batch);del v
        except torch.cuda.OutOfMemoryError:
            segments.append(dict(segment,end=i));fallbacks.append({'row':i,'reason':'GPU1 OOM; reduce batch or CPU'})
            torch.cuda.empty_cache()
            if batch_size==32:batch_size=16
            else:model=model.cpu();torch.cuda.empty_cache();device='cpu';batch_size=32
            segment={'start':i,'device':device,'physical_gpu':1 if device!='cpu' else None,'batch_size':batch_size,'pid':os.getpid()};continue
        if i-committed>=1024 or time.monotonic()-last>=15 or i==total:seal('running')
    if i<total:
        seal('paused');model=model.cpu()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        return 75
    seal('complete_prefix');assert np.isfinite(vectors).all()
    assert sha(bank.out/'extra_texts.json')==texts_sha,'Concurrent text registry writer detected'
    if old:assert np.array_equal(vectors[:old],bank.extra_vectors)
    peak=torch.cuda.max_memory_allocated(0) if torch.cuda.is_initialized() else 0
    del vectors;bank.extra_vectors=None;tmp.replace(target);fsync_directory(target.parent)
    receipt=dict(complete=True,rows=total,new_texts=total-old,preserved_prefix_rows=old,texts_sha256=texts_sha,vectors_sha256=sha(target),encoder_helper_sha256=sha(__file__),resource_policy_sha256=sha(root/'manifests/resource_policy.json'),seconds=time.monotonic()-begin,dtype='float32',max_length=256,pool=definition['pool'],source_policy_sha256=sha(root/'manifests/features_policy.json'),device_segments=segments+[dict(segment,end=i)],fallbacks=fallbacks,physical_compute_gpu=1,GPU0_used=False,GPU1_memory_fraction=.08,GPU1_peak_allocated_bytes=peak,training=False,one_shared_vector_per_text=True,TF32=False,resumed_at=resumed_at,checkpoint_sha256=sha(checkpoint))
    durable_json(bank.out/'encoding_complete.json',receipt);print(json.dumps(receipt),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
