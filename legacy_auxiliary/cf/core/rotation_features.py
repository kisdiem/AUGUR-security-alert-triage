"""New-version B400/D240 feature reconstruction. Never trains a model.

Old projects are read-only inputs. Each native/Norm fold has its own original-
train IDF. New caches and matrices are written only beneath --root.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse,bisect,collections,copy,gzip,hashlib,importlib.util,json,math,os,pathlib,pickle,resource,sqlite3,time
import numpy as np
from history_ledger import CommonLedger,MemoryStore
from strict_cf_norm import transform_event,normalization_spec

SCENES=('fox','harrison','russellmitchell','santos','shaw','wardbeck','wheeler','wilson')
DEFAULT_ROOT='<RESEARCH_ROOT>/reproductions/security_completion_20261026_v1'
DEFAULT_BASE='<RESEARCH_ROOT>/reproductions'
DEFAULT_OLD='<RESEARCH_ROOT>/reproductions/ait_state_cf_systematic_20261024'
DEFAULT_DATA='<RESEARCH_ROOT>/datasets/additional_logs/ait_ads/extracted'
FEATURE_VERSION='rotation_features_v1'

def read(p): return json.loads(pathlib.Path(p).read_text('utf-8'))
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def object_sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def array_sha(x):
    x=np.ascontiguousarray(x)
    return object_sha({'shape':list(x.shape),'dtype':str(x.dtype),'bytes':hashlib.sha256(x.tobytes()).hexdigest()})
def save(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),'utf-8');t.replace(p)
def dump(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with gzip.open(t,'wb',compresslevel=1) as f:pickle.dump(x,f,pickle.HIGHEST_PROTOCOL)
    t.replace(p)
def load(p):
    with gzip.open(p,'rb') as f:return pickle.load(f)
def npz(p,**arrays):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with t.open('wb') as f:np.savez_compressed(f,**arrays)
    t.replace(p)
def log(root,stage,**kw):
    d={'time':time.strftime('%Y-%m-%d %H:%M:%S %z'),'stage':stage,**kw}
    save(root/'manifests/features_progress.json',d);print(json.dumps(d),flush=True)
def folds():
    return [{'id':f'fold_{i+1:02d}','train':[s for s in SCENES if s not in (scene,SCENES[(i+1)%8])],
             'validation':[SCENES[(i+1)%8]],'test':[scene]} for i,scene in enumerate(SCENES)]
def graph_summary(event,before,dep):
    row=dict(event,state_before=before);g=dep.graph(row)
    texts=[' '.join(b).replace('_',' ') for b in g['node_bags']] or ['no observed prior entity state']
    return texts,dep.state_summary(g)

class Dependencies:
    def __init__(self,base):
        self.base=pathlib.Path(base);self.P=self.base/'nllog_crossscene_422_20261023';self.Q=self.base/'nllog_raw_structured_20261024';self.S=self.base/'ait_ads_session_state_20261023';self.M=self.base/'nllog_cf_multifactor_20261024'
        sys.path[:0]=[str(self.P/'deps'),str(self.P),str(self.Q),str(self.S)]
        from alerts import parse,adapted
        from adapters import AITAdapter
        from engine import Environment
        from features_control import raw_normalize
        from representation import normalize,fit_idf,weight,state_summary
        from graph_features import graph
        spec=importlib.util.spec_from_file_location('rotation_structured',self.Q/'prepare_raw.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        spec=importlib.util.spec_from_file_location('rotation_dns',self.base/'nllog_dns_expert_20261024/features_dns.py');dns=importlib.util.module_from_spec(spec);spec.loader.exec_module(dns)
        for name,value in locals().copy().items():
            if name in ('parse','adapted','AITAdapter','Environment','raw_normalize','normalize','fit_idf','weight','state_summary','graph'):setattr(self,name,value)
        self.structured=mod.structured;self.DNSAggregate=dns.Aggregate
    def raw_text(self,source,record):
        t=self.raw_normalize(record['raw'],record['host'])
        return t or self.raw_normalize(self.structured(source),record['host'])

class VectorBank:
    """Read-only old text vectors plus a new, independently saved CPU cache."""
    def __init__(self,dep,old_root,out,threads=2):
        self.dep=dep;self.out=out;self.threads=threads;out.mkdir(parents=True,exist_ok=True)
        self.lookup={};self.refs=[];self.arrays=[];self.base_paths=[];self.extra=[]
        pairs=[(dep.Q/'raw/texts.json',dep.Q/'raw/vectors.npy'),(dep.M/'raw_texts.json',dep.M/'raw_vectors.npy'),
               (dep.P/'cache/texts.json',dep.P/'embeddings.npy'),(old_root/'data/history/new_texts.json',old_root/'data/history/new_vectors.npy')]
        self.q_to_bank=None
        for ti,(tp,vp) in enumerate(pairs):
            texts=read(tp);a=np.load(vp,mmap_mode='r',allow_pickle=False);assert a.shape==(len(texts),384)
            ai=len(self.arrays);self.arrays.append(a);mapped=[]
            for i,t in enumerate(texts):
                if t not in self.lookup:self.lookup[t]=len(self.refs);self.refs.append((ai,i))
                mapped.append(self.lookup[t])
            if ti==0:self.q_to_bank=np.asarray(mapped,np.int64)
            self.base_paths.extend([tp,vp]);del texts,mapped
        self.base_count=len(self.refs)
        if (out/'extra_texts.json').exists():
            self.extra=read(out/'extra_texts.json')
            for i,t in enumerate(self.extra):
                assert t not in self.lookup;self.lookup[t]=len(self.refs);self.refs.append((-1,i))
        self.extra_vectors=np.load(out/'extra_vectors.npy',mmap_mode='r') if (out/'extra_vectors.npy').exists() else None
    def index(self,text):
        if not text:return -1
        if text not in self.lookup:
            i=len(self.extra);self.extra.append(text);self.lookup[text]=len(self.refs);self.refs.append((-1,i))
        return self.lookup[text]
    def get(self,index):
        a,i=self.refs[int(index)]
        if a>=0:return self.arrays[a][i]
        assert self.extra_vectors is not None and i<len(self.extra_vectors), 'Run encode-cache before pooling'
        return self.extra_vectors[i]
    def register_texts(self):save(self.out/'extra_texts.json',self.extra)
    def encode(self):
        self.register_texts();old=0 if self.extra_vectors is None else len(self.extra_vectors)
        assert old<=len(self.extra)
        if old==len(self.extra):return {'new_texts':0,'total':old,'seconds':0.}
        import torch
        from transformers import AutoTokenizer,AutoModel
        torch.set_num_threads(self.threads)
        tokenizer=AutoTokenizer.from_pretrained(self.dep.Q/'model',local_files_only=True,trust_remote_code=False)
        model=AutoModel.from_pretrained(self.dep.Q/'model',local_files_only=True,trust_remote_code=False,use_safetensors=True).eval().requires_grad_(False)
        target=self.out/'extra_vectors.npy';tmp=target.with_name(target.name+'.tmp')
        vectors=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(self.extra),384))
        if old:vectors[:old]=self.extra_vectors
        begin=time.monotonic()
        with torch.inference_mode():
            for i in range(old,len(self.extra),32):
                batch=self.extra[i:i+32];tokens=tokenizer(batch,padding=True,truncation=True,max_length=256,return_tensors='pt')
                h=model(**tokens).last_hidden_state;mask=tokens['attention_mask'][...,None]
                vectors[i:i+len(batch)]=((h*mask).sum(1)/mask.sum(1).clamp_min(1)).numpy()
                if (i-old)%1024==0:save(self.out/'encoding_progress.json',{'rows':i+len(batch),'total':len(self.extra),'seconds':time.monotonic()-begin})
        assert np.isfinite(vectors).all();vectors.flush();del vectors
        self.extra_vectors=None;tmp.replace(target);self.extra_vectors=np.load(target,mmap_mode='r')
        receipt={'new_texts':len(self.extra)-old,'total':len(self.extra),'seconds':time.monotonic()-begin,'device':'cpu','max_length':256,'pool':'attention-mask mean including special tokens; no L2','texts_sha256':sha(self.out/'extra_texts.json'),'vectors_sha256':sha(target)}
        save(self.out/'encoding_complete.json',receipt);del model,tokenizer;return receipt

class SessionIndex:
    def __init__(self,dep,scenes):
        self.meta={};self.intervals={};self.starts={};self.by_scene={}
        for scene in scenes:
            values=read(dep.S/'cache'/scene/'complete.json')['sessions'];self.by_scene[scene]=values
            for m in values:
                assert m['id'] not in self.meta;self.meta[m['id']]=dict(m,scene=scene)
                self.intervals.setdefault((scene,m['host_hash']),[]).append((m['start'],m['end'],m['id']))
        for key,v in self.intervals.items():v.sort();self.starts[key]=[x[0] for x in v]
    def locate(self,scene,host,t):
        key=(scene,hashlib.sha256((scene+host).encode()).hexdigest());vv=self.intervals[key]
        i=bisect.bisect_right(self.starts[key],t)-1
        assert i>=0 and t<=vv[i][1];return vv[i][2]

def _newrow(native=None):
    return {'signatures':collections.Counter(),'raw_pairs':collections.Counter(),'state_pairs':collections.Counter(),
            'numeric_sum':np.zeros(224,np.float64),'numeric_max':np.zeros(224,np.float32),'n':0,
            'ledger_raw_pairs':collections.Counter(),'ledger_state_pairs':collections.Counter(),
            'ledger_numeric_sum':np.zeros(224,np.float64),'ledger_numeric_max':np.zeros(224,np.float32)}

class RotationFeatures:
    def __init__(self,root=DEFAULT_ROOT,base=DEFAULT_BASE,old_root=DEFAULT_OLD,data=DEFAULT_DATA,threads=2,pool_only=False):
        self.root=pathlib.Path(root).resolve();self.old=pathlib.Path(old_root).resolve();self.data=pathlib.Path(data)
        assert self.root!=self.old and self.old not in self.root.parents,'New root must be isolated from old experiment'
        self.root.mkdir(parents=True,exist_ok=True);self.dep=Dependencies(base);self.pool_only=pool_only
        self.index=SessionIndex(self.dep,() if pool_only else SCENES);self.native={};self.raw={}
        if not pool_only:
            with (self.dep.P/'cache/sessions.pkl').open('rb') as f:old_roles=pickle.load(f)
            self.native={r['id']:r for rows in old_roles.values() for r in rows};assert len(self.native)==sum(map(len,old_roles.values()))
            with (self.dep.Q/'raw/sessions.pkl').open('rb') as f:self.raw=pickle.load(f)
            assert set(self.native)==set(self.raw)
        self.bank=VectorBank(self.dep,self.old,self.root/'cache/embeddings',threads)
        self.code_hashes={p.name:sha(p) for p in (pathlib.Path(__file__),pathlib.Path(__file__).with_name('history_ledger.py'),pathlib.Path(__file__).with_name('strict_cf_norm.py'))}
        lock=self.root/'manifests/features_policy.json'
        if pool_only:
            self.policy=read(lock)
            assert self.policy['code']==self.code_hashes,'Code changed after scene preparation'
            assert self.policy['normalization']==normalization_spec()
            # Global text/vector banks are the sole cross-scene data exception.
            # No mixed-role sessions/raw pool pickle or other-scene metadata is
            # opened in this mode; only requested scene shards are opened below.
            for path in self.bank.base_paths:
                assert sha(path)==self.policy['source_hashes'][str(path)]
            return
        source_files=[self.dep.P/'cache/sessions.pkl',self.dep.Q/'raw/sessions.pkl',*self.bank.base_paths,
            self.dep.S/'alerts.sqlite',self.dep.S/'alerts.py',self.dep.S/'adapters.py',self.dep.S/'engine.py',
            self.dep.P/'features.py',self.dep.P/'graph_features.py',self.dep.P/'representation.py',
            self.dep.Q/'features_control.py',self.dep.Q/'prepare_raw.py',
            self.dep.base/'nllog_dns_expert_20261024/features_dns.py']
        source_files.extend(self.dep.S/'cache'/s/'complete.json' for s in SCENES)
        source_files.extend(p for p in sorted((self.dep.Q/'model').rglob('*')) if p.is_file())
        self.policy={'version':FEATURE_VERSION,'code':self.code_hashes,'normalization':normalization_spec(),'history':{'entities':12,'recent':6,'seconds':43200},'source_base':str(self.dep.base),'old_root':str(self.old),
                     'source_hashes':{str(p):sha(p) for p in dict.fromkeys(source_files)}}
        if lock.exists():assert read(lock)==self.policy,'Changed policy/code: use a new version root'
        else:save(lock,self.policy)

    def _source_db(self,scene,folder,max_events=0):
        target=folder/'sources.sqlite';done=folder/'sources_complete.json'
        if done.exists():
            receipt=read(done);assert receipt['policy_sha256']==object_sha(self.policy) and target.exists()
            assert receipt['max_events']==max_events;return receipt
        if target.exists():raise RuntimeError('Unsealed source cache exists; inspect and use a fresh scene version')
        tmp=folder/'sources.sqlite.tmp';tmp.unlink(missing_ok=True)
        db=sqlite3.connect(tmp);db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE events(id TEXT PRIMARY KEY,t REAL,host TEXT,sid TEXT,record TEXT,norm_record TEXT,raw_id INT,norm_raw_id INT,cf_record TEXT,cf_raw_id INT,cf_audit TEXT)')
        agg={};qual={};candidates=collections.defaultdict(list);counts=collections.Counter();hashes={};begin=time.monotonic()
        source_db=sqlite3.connect('file:'+str(self.dep.S/'alerts.sqlite')+'?mode=ro',uri=True)
        expected={p:(h,n) for p,h,n in source_db.execute('SELECT file,sha,rows FROM sources')};source_db.close()
        partial=False
        for family in ('aminer','wazuh'):
            p=self.data/(scene+'_'+family+'.json');h=hashlib.sha256();file_rows=0
            with p.open('rb') as f:
                for ordinal,line in enumerate(f,1):
                    h.update(line);file_rows+=1;source=json.loads(line);record=self.dep.parse(source,family,scene,ordinal)
                    sid=self.index.locate(scene,record['host'],record['t'])
                    norm=transform_event(source,record,'norm');cf=transform_event(source,record,'cf')
                    assert norm['audit']['eligible'],norm['audit']
                    a=cf['audit'];q=qual.setdefault(sid,{'events_total':0,'candidate_events':0,'changed_events':0,'reasons':collections.Counter()});q['events_total']+=1
                    if a['candidate']:
                        q['candidate_events']+=1
                        if a['eligible'] and a['changed']:
                            adapter=self.dep.AITAdapter('AIT_ADS:'+scene)
                            old=self.dep.adapted(record,adapter);new=self.dep.adapted(cf['record'],self.dep.AITAdapter('AIT_ADS:'+scene))
                            fields=('event_id','timestamp_ns','actor','target','focus','entity_observations','action','parent','account','session','host','arguments','return_value','outcome','audit_group')
                            if any(old.get(k)!=new.get(k) for k in fields):
                                a=dict(a,eligible=False,reasons=a['reasons']+['UNSUPPORTED_STATE_TRANSITION']);cf={'source':source,'record':record,'audit':a}
                            else:
                                q['changed_events']+=1;candidates[sid].append({'source':source,'record':record,'family':family,'ordinal':ordinal})
                        if not a['eligible']:q['reasons'].update(a['reasons'])
                    g=agg.setdefault(sid,{k:self.dep.DNSAggregate() for k in ('original','norm','cf')})
                    g['original'].add(record,source);g['norm'].add(norm['record'],norm['source']);g['cf'].add(cf['record'],cf['source'])
                    rv=self.bank.index(self.dep.raw_text(source,record));nv=self.bank.index(self.dep.raw_text(norm['source'],norm['record']));cv=self.bank.index(self.dep.raw_text(cf['source'],cf['record']))
                    changed_norm=norm['record']!=record;changed_cf=a.get('eligible') and a.get('changed')
                    db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?)',(record['id'],record['t'],record['host'],sid,json.dumps(record,separators=(',',':')),json.dumps(norm['record'],separators=(',',':')) if changed_norm else None,rv,nv,json.dumps(cf['record'],separators=(',',':')) if changed_cf else None,cv,json.dumps(a,separators=(',',':')) if a['candidate'] else None))
                    counts['events']+=1;counts[family]+=1
                    if counts['events']%25000==0:db.commit();log(self.root,'source_index',scene=scene,events=counts['events'],seconds=time.monotonic()-begin)
                    if max_events and file_rows>=max_events:partial=True;break
            hashes[str(p)]={'sha256':h.hexdigest(),'rows':file_rows,'partial':bool(max_events)}
            if not max_events:
                assert str(p) in expected and (h.hexdigest(),file_rows)==expected[str(p)],'Original source drift'
        db.execute('CREATE INDEX event_order ON events(t,id)');db.execute('CREATE INDEX session_events ON events(sid,t,id)');db.commit();db.close();tmp.replace(target)
        summary={};eligible=[]
        for sid,q in qual.items():
            reasons=sorted(q['reasons'])
            if not q['candidate_events']:reasons.append('NO_TARGET_CANDIDATE_IN_PARENT')
            if not q['changed_events'] and not reasons:reasons.append('NO_ACTUAL_INPUT_CHANGE')
            good=q['changed_events']>0 and not q['reasons'] and self.index.meta[sid]['y'] in (0,1)
            if self.index.meta[sid]['y'] not in (0,1):reasons.append('PENDING_NOT_SUPERVISED_PARENT')
            summary[sid]={**q,'reasons':reasons,'eligible':good,'label':self.index.meta[sid]['y']}
            if good:eligible.append(sid)
        dns={sid:{k:v.vector() for k,v in value.items()} for sid,value in agg.items()}
        dump(folder/'dns_and_qualification.pkl.gz',{'dns':dns,'qualification':summary})
        with gzip.open(folder/'candidate_parents.jsonl.gz','wt',encoding='utf-8') as f:
            for sid in sorted(eligible):
                m=self.index.meta[sid];events=candidates[sid]
                f.write(json.dumps({'parent_id':sid,'scene':scene,'label':m['y'],'host':events[0]['record']['host'],'start':m['start'],'end':m['end'],'events':events,'events_scope':'eligible change events only; complete original parent remains in sources.sqlite','static_only':True},ensure_ascii=False)+'\n')
        self.bank.register_texts()
        receipt={'scene':scene,'policy_sha256':object_sha(self.policy),'counts':dict(counts),'source_hashes':hashes,'max_events':max_events,'eligible_parents':len(eligible),'eligible_by_label':dict(collections.Counter(str(self.index.meta[s]['y']) for s in eligible)),'all_parent_qualification':'dns_and_qualification.pkl.gz','candidate_parents':'candidate_parents.jsonl.gz','seconds':time.monotonic()-begin,'complete':not max_events}
        save(done,receipt);return receipt

    def prepare_scene(self,scene,max_events=0):
        folder=self.root/'cache/scenes'/scene;folder.mkdir(parents=True,exist_ok=True)
        source_receipt=self._source_db(scene,folder,max_events);done=folder/'features_complete.json'
        if done.exists():
            d=read(done);assert d['policy_sha256']==object_sha(self.policy) and d['cache_sha256']==sha(folder/'rows.pkl.gz');return d
        d=load(folder/'dns_and_qualification.pkl.gz');qual=d['qualification'];dns=d['dns'];eligible={s for s,q in qual.items() if q['eligible']}
        active={sid for sid,r in self.native.items() if r['scene']==scene};rows={sid:{'norm':_newrow(),'ledger':_newrow(),'cf':None} for sid in active}
        for sid in eligible & active:
            old=self.native[sid];c=_newrow();c['signatures']=collections.Counter(old['signatures']);c['state_pairs']=collections.Counter({(s,self.bank.index(t)):n for (s,t),n in old['state_text'].items()})
            c['raw_pairs']=collections.Counter({(s,int(self.bank.q_to_bank[i]) if i>=0 else -1):n for (s,i),n in self.raw[sid]['pairs'].items()})
            c['numeric_mean']=old['numeric_mean'];c['numeric_max']=old['numeric_max'];c['n']=old['alerts'];rows[sid]['cf']=c
        env=self.dep.Environment(MemoryStore());normenv=self.dep.Environment(MemoryStore());adapter=self.dep.AITAdapter('AIT_ADS:'+scene);normadapter=self.dep.AITAdapter('AIT_ADS:'+scene)
        ledger=CommonLedger(self.dep.Environment);counts=collections.Counter();orig_sigs=collections.defaultdict(collections.Counter);events={k:[] for k in ('event_ids','session_ids','scene','host','time','signature','supervised_mask')};begin=time.monotonic()
        db=sqlite3.connect('file:'+str(folder/'sources.sqlite')+'?mode=ro',uri=True)
        for eid,t,host,sid,payload,norm_payload,rv,nv,cf_payload,cv in db.execute('SELECT id,t,host,sid,record,norm_record,raw_id,norm_raw_id,cf_record,cf_raw_id FROM events ORDER BY t,id'):
            record=json.loads(payload);normrecord=json.loads(norm_payload) if norm_payload else record;cf_record=json.loads(cf_payload) if cf_payload and sid in eligible else record
            event=self.dep.adapted(record,adapter);nevent=self.dep.adapted(normrecord,normadapter);before=env.before(event);nbefore=normenv.before(nevent)
            assert before['system_version']==counts['events']
            bounded,prior=ledger.before(before,event);sig=self.dep.normalize(record['text']);nsig=self.dep.normalize(normrecord['text']);csig=self.dep.normalize(cf_record['text'])
            if sid in active:
                row=rows[sid];n=row['norm'];l=row['ledger'];n['n']+=1;l['n']+=1;n['signatures'][nsig]+=1;l['signatures'][sig]+=1;n['raw_pairs'][(nsig,nv)]+=1;orig_sigs[sid][sig]+=1
                texts,numeric=graph_summary(nevent,nbefore,self.dep)
                for text in texts:n['state_pairs'][(nsig,self.bank.index(text))]+=1/len(texts)
                n['numeric_sum']+=numeric;n['numeric_max']=np.maximum(n['numeric_max'],numeric)
                texts,numeric=graph_summary(event,bounded,self.dep)
                for text in texts:l['ledger_state_pairs'][(sig,self.bank.index(text))]+=1/len(texts)
                l['ledger_numeric_sum']+=numeric;l['ledger_numeric_max']=np.maximum(l['ledger_numeric_max'],numeric)
                raw_prior=[e for e in prior.values() if e['raw_id']>=0]
                for e in raw_prior:l['ledger_raw_pairs'][(sig,e['raw_id'])]+=1/len(raw_prior)
                if sid in eligible:
                    c=row['cf'];c['ledger_numeric_sum']+=numeric;c['ledger_numeric_max']=np.maximum(c['ledger_numeric_max'],numeric)
                    for text in texts:c['ledger_state_pairs'][(csig,self.bank.index(text))]+=1/len(texts)
                    for e in raw_prior:
                        v=e['cf_raw_id'] if e['session_id']==sid else e['raw_id']
                        c['ledger_raw_pairs'][(csig,v)]+=1/len(raw_prior)
                    if cf_payload:
                        c['signatures'][sig]-=1;c['signatures'][csig]+=1;c['raw_pairs'][(sig,rv)]-=1;c['raw_pairs'][(csig,cv)]+=1
                        native_texts,_=graph_summary(event,before,self.dep)
                        for text in native_texts:
                            v=self.bank.index(text);c['state_pairs'][(sig,v)]-=1/len(native_texts);c['state_pairs'][(csig,v)]+=1/len(native_texts)
            for key,val in [('event_ids',eid),('session_ids',sid),('scene',scene),('host',host),('time',t),('signature',record['text']),('supervised_mask',sid in self.native)]:events[key].append(val)
            env.apply(event);normenv.apply(nevent);ledger.observe(event,sid,rv,cv if sid in eligible else rv);counts['events']+=1
            if counts['events']%10000==0:log(self.root,'common_ledger',scene=scene,events=counts['events'],seconds=time.monotonic()-begin)
        db.close()
        for sid,parts in rows.items():
            old=self.native[sid];l=parts['ledger'];n=parts['norm']
            if not max_events:assert l['n']==old['alerts'] and orig_sigs[sid]==old['signatures']
            if not l['n']:continue
            n['numeric_mean']=(n.pop('numeric_sum')/n['n']).astype(np.float32)
            l['ledger_numeric_mean']=(l.pop('ledger_numeric_sum')/l['n']).astype(np.float32)
            if parts['cf']:
                c=parts['cf'];c['ledger_numeric_mean']=(c.pop('ledger_numeric_sum')/l['n']).astype(np.float32)
                c['signatures']=+c['signatures']
                for key in ('raw_pairs','state_pairs','ledger_raw_pairs','ledger_state_pairs'):
                    c[key]=collections.Counter({k:v for k,v in c[key].items() if v>1e-8})
                assert sum(c['signatures'].values())==old['alerts']
        if max_events:rows={s:v for s,v in rows.items() if v['ledger']['n']}
        events={k:np.asarray(v,dtype=np.float64 if k=='time' else bool if k=='supervised_mask' else str) for k,v in events.items()}
        dump(folder/'rows.pkl.gz',{'rows':rows,'dns':dns,'qualification':qual,'source_receipt':source_receipt,
                                  'native_rows':{sid:self.native[sid] for sid in rows},
                                  'raw_rows':{sid:self.raw[sid] for sid in rows},
                                  'metadata':self.index.by_scene[scene]})
        npz(folder/'events.npz',**events);self.bank.register_texts()
        receipt={'scene':scene,'policy_sha256':object_sha(self.policy),'cache_sha256':sha(folder/'rows.pkl.gz'),'events_sha256':sha(folder/'events.npz'),'complete':not max_events,'events':counts['events'],'supervised_sessions':len(rows),'ledger':ledger.receipt(),'seconds':time.monotonic()-begin,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'partial_smoke' if max_events else 'full_scene'}
        save(done,receipt);return receipt

    def _denom(self,signatures,idf):
        weights={s:self.dep.weight(s,idf) for s in signatures};d=sum(weights[s]*n for s,n in signatures.items())
        if abs(d)<1e-8:weights={s:1. for s in signatures};d=sum(signatures.values())
        assert abs(d)>0;return weights,d+1e-8
    def _pool(self,pairs,signatures,idf):
        weights,denom=self._denom(signatures,idf);v=np.zeros(384,np.float64)
        for (sig,index),n in pairs.items():
            if index>=0:v+=self.bank.get(index)*(weights[sig]*n)
        return (v/denom).astype(np.float32)
    def _alert(self,signatures,idf):return self._pool({(s,self.bank.index(s)):n for s,n in signatures.items()},signatures,idf)
    def _native(self,sid,idf,scene_cache):
        old=self.native[sid];signatures=old['signatures'];raw={(s,int(self.bank.q_to_bank[i]) if i>=0 else -1):n for (s,i),n in self.raw[sid]['pairs'].items()};state={(s,self.bank.index(t)):n for (s,t),n in old['state_text'].items()}
        s0=np.r_[self._alert(signatures,idf),self._pool(raw,signatures,idf),scene_cache['dns'][sid]['original']].astype(np.float32)
        s3=np.r_[s0,self._pool(state,signatures,idf),old['numeric_mean'],old['numeric_max']].astype(np.float32)
        return s0,s3
    def _one(self,sid,idf,normidf,cache,cf=False):
        row=cache['rows'][sid];l=row['ledger'];n=row['norm']
        if cf:
            c=row['cf'];assert c is not None;sig=c['signatures']
            s0=np.r_[self._alert(sig,idf),self._pool(c['raw_pairs'],sig,idf),cache['dns'][sid]['cf']].astype(np.float32)
            s3=np.r_[s0,self._pool(c['state_pairs'],sig,idf),c['numeric_mean'],c['numeric_max']].astype(np.float32);h=c
        else:s0,s3=self._native(sid,idf,cache);h=l;sig=l['signatures']
        numeric=np.r_[h['ledger_numeric_mean'],h['ledger_numeric_max']].astype(np.float32)
        hcount=np.r_[s0,numeric].astype(np.float32)
        out={'S0':s0,'S3':s3,'Hcount':hcount,'Hflat':np.r_[hcount,self._pool(h['ledger_raw_pairs'],sig,idf)].astype(np.float32),'Hstate':np.r_[hcount,self._pool(h['ledger_state_pairs'],sig,idf)].astype(np.float32)}
        if not cf:
            ns=n['signatures'];norm0=np.r_[self._alert(ns,normidf),self._pool(n['raw_pairs'],ns,normidf),cache['dns'][sid]['norm']].astype(np.float32)
            out['Norm_S0']=norm0;out['Norm_S3']=np.r_[norm0,self._pool(n['state_pairs'],ns,normidf),n['numeric_mean'],n['numeric_max']].astype(np.float32)
        assert all(np.isfinite(v).all() for v in out.values());return out

    def build_fold(self,fold,phase='train-validation',seal=None):
        fold=next(x for x in folds() if x['id']==fold);folder=self.root/'data/folds'/fold['id'];folder.mkdir(parents=True,exist_ok=True)
        receipt_name='test_features_receipt.json' if phase=='test' else 'train_only_features_receipt.json' if phase=='train-only' else 'features_receipt.json'
        existing_receipt=folder/receipt_name
        if existing_receipt.exists():
            old=read(existing_receipt)
            assert old['policy_sha256']==object_sha(self.policy) and old['feature_sha256']==sha(old['feature_file'])
            return old
        if phase=='test':
            seal=pathlib.Path(seal or self.root/'results/global_validation_seal.json');assert seal.is_file(),'All-eight validation seal required'
            seal_doc=read(seal);assert all(seal_doc.get(k) is True for k in ('sealed','all_eight_validation_sealed','complete_plan_baseline_gate')), 'Complete-plan global validation seal required'
            save(folder/'test_features_started.json',{'global_validation_seal':str(seal),'sha256':sha(seal)})
        required=fold['train'] if phase=='train-only' else fold['train']+fold['validation'] if phase!='test' else fold['test']
        caches={s:load(self.root/'cache/scenes'/s/'rows.pkl.gz') for s in required}
        for s in required:assert read(self.root/'cache/scenes'/s/'features_complete.json')['complete'],'Smoke cache cannot form full fold'
        if self.pool_only:
            self.native={};self.raw={};self.index.meta={};self.index.by_scene={}
            for scene,cache in caches.items():
                self.native.update(cache['native_rows']);self.raw.update(cache['raw_rows']);self.index.by_scene[scene]=cache['metadata']
                self.index.meta.update({m['id']:dict(m,scene=scene) for m in cache['metadata']})
        ids={role:sorted((sid for sid,r in self.native.items() if r['scene'] in fold[role]),key=lambda sid:(SCENES.index(self.native[sid]['scene']),self.index.meta[sid]['start'],sid)) for role in ('train','validation','test')}
        if phase!='test':
            idf=self.dep.fit_idf([self.native[sid] for sid in ids['train']]);normidf=self.dep.fit_idf([caches[self.native[sid]['scene']]['rows'][sid]['norm'] for sid in ids['train']])
            save(folder/'idf_original.json',idf);save(folder/'idf_norm.json',normidf)
        else:idf=read(folder/'idf_original.json');normidf=read(folder/'idf_norm.json')
        # Signature encodings discovered at pooling must be known before encode.
        roles=('train',) if phase=='train-only' else ('train','validation') if phase!='test' else ('test',)
        for role in roles:
            for sid in ids[role]:
                r=self.native[sid];cache=caches[r['scene']]
                for sig in r['signatures']:self.bank.index(sig)
                for sig in cache['rows'][sid]['norm']['signatures']:self.bank.index(sig)
                c=cache['rows'][sid]['cf']
                if c:
                    for sig in c['signatures']:self.bank.index(sig)
        encoding=self.bank.encode();out={};begin=time.monotonic()
        for role in roles:
            values=collections.defaultdict(list)
            for sid in ids[role]:
                for state,x in self._one(sid,idf,normidf,caches[self.native[sid]['scene']]).items():values[state].append(x)
            for state,x in values.items():out['X_'+state+'_'+role]=np.stack(x).astype(np.float32)
            out['ids_'+role]=np.asarray(ids[role]);out['y_'+role]=np.asarray([self.native[s]['y'] for s in ids[role]],np.int8);out['scene_'+role]=np.asarray([self.native[s]['scene'] for s in ids[role]])
            event_parts=[]
            for s in fold[role]:
                with np.load(self.root/'cache/scenes'/s/'events.npz',allow_pickle=False) as z:event_parts.append({k:z[k] for k in z.files})
            ev={k:np.concatenate([v[k] for v in event_parts]) for k in event_parts[0]};order=np.lexsort((ev['event_ids'],ev['time']));npz(folder/(role+'_events.npz'),**{k:v[order] for k,v in ev.items()})
        parents=[]
        if phase!='test':
            parents=[sid for sid in ids['train'] if caches[self.native[sid]['scene']]['qualification'].get(sid,{}).get('eligible')]
            ix={sid:i for i,sid in enumerate(ids['train'])};out['parent_indices']=np.asarray([ix[s] for s in parents],np.int64);out['parent_ids']=np.asarray(parents,dtype=str);out['cf_y']=out['y_train'][out['parent_indices']]
            values=collections.defaultdict(list)
            for sid in parents:
                for state,x in self._one(sid,idf,normidf,caches[self.native[sid]['scene']],True).items():values[state].append(x)
            for state in ('S0','S3','Hcount','Hflat','Hstate'):out['CF_'+state]=np.stack(values[state]).astype(np.float32) if parents else np.empty((0,out['X_'+state+'_train'].shape[1]),np.float32)
        target=folder/('test.npz' if phase=='test' else 'train_only.npz' if phase=='train-only' else 'train_validation.npz');npz(target,**out)
        metadata={'fold':fold,'roles':{role:[dict(self.index.meta[s],attack_chain_id=None,attack_chain_status='unknown_not_independent_session_claim') for s in ids[role]] for role in roles},'pending':{role:[dict(m,attack_chain_id=None,supervised=False) for scene in fold[role] for m in self.index.by_scene[scene] if m['y'] not in (0,1)] for role in roles}}
        save(folder/('test_session_metadata.json' if phase=='test' else 'train_only_session_metadata.json' if phase=='train-only' else 'session_metadata.json'),metadata)
        checks={}
        for role in roles:
            shared=object_sha({'ids':ids[role],'labels':out['y_'+role].tolist(),'source_scenes':fold[role],
                              'source_hashes':{s:caches[s]['source_receipt']['source_hashes'] for s in fold[role]}})
            checks[role]={'native_id_label_source_sha256':shared,'norm_id_label_source_sha256':shared,'native_real_hashes':{s:array_sha(out['X_'+s+'_'+role]) for s in ('S0','S3')},'norm_real_hashes':{s:array_sha(out['X_Norm_'+s+'_'+role]) for s in ('S0','S3')},'common_H_numeric_sha256':array_sha(out['X_Hcount_'+role][:,811:])}
            assert np.array_equal(out['X_Hcount_'+role],out['X_Hflat_'+role][:,:1259]) and np.array_equal(out['X_Hcount_'+role],out['X_Hstate_'+role][:,:1259])
        receipt={'feature_version':FEATURE_VERSION,'fold':fold,'phase':phase,'complete':True,'policy_sha256':object_sha(self.policy),'feature_file':str(target),'feature_sha256':sha(target),'IDF_original_sha256':sha(folder/'idf_original.json'),'IDF_norm_sha256':sha(folder/'idf_norm.json'),'IDF_original_train_only':True,'CF_does_not_fit_IDF':True,'parents':len(parents),'parent_ids_sha256':array_sha(np.asarray(parents,dtype=str)),'roles':{role:{'n':len(ids[role]),'positive':int(out['y_'+role].sum())} for role in roles},'dimensions':{k:list(v.shape) for k,v in out.items()},'family_invariance':checks,'matched_exposure':'same parent_indices and cf_y for repeat, CF, Norm+repeat; original rows never modified by augmentation','encoding':encoding,'seconds_pooling':time.monotonic()-begin,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'all_scenes_prior_development':True,'CF_label_status':'conditional static field consistency; not sensor-certified','H_scope':'same retained record ledger; native S3 lifetime information is a separate representation'}
        save(folder/receipt_name,receipt);return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=DEFAULT_ROOT);p.add_argument('--base',default=DEFAULT_BASE);p.add_argument('--old-root',default=DEFAULT_OLD);p.add_argument('--data',default=DEFAULT_DATA);p.add_argument('--threads',type=int,default=2);p.add_argument('--stage',choices=['prepare-scenes','build-folds','train-only','test','smoke'],required=True);p.add_argument('--scene',choices=SCENES,action='append');p.add_argument('--fold',choices=[f'fold_{i:02d}' for i in range(1,9)],action='append');p.add_argument('--smoke-events',type=int,default=1000);p.add_argument('--seal');a=p.parse_args()
    if a.stage=='smoke':a.root=str(pathlib.Path(a.root)/'smoke'/('events_'+str(a.smoke_events)))
    r=RotationFeatures(a.root,a.base,a.old_root,a.data,a.threads,pool_only=a.stage in ('build-folds','train-only','test'))
    if a.stage in ('prepare-scenes','smoke'):
        for scene in a.scene or (['russellmitchell'] if a.stage=='smoke' else SCENES):
            receipt=r.prepare_scene(scene,a.smoke_events if a.stage=='smoke' else 0);print(json.dumps(receipt),flush=True)
        if a.stage=='smoke':print(json.dumps({'smoke_encoding':r.bank.encode()}),flush=True)
    else:
        for fold in a.fold or [f['id'] for f in folds()]:
            receipt=r.build_fold(fold,'test' if a.stage=='test' else 'train-only' if a.stage=='train-only' else 'train-validation',a.seal);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
