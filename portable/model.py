"""The selected 4/2/2 AUGUR numerical feature-to-score path.

This is an explicit port, checked against the supplied frozen cell. It does not
generate embeddings, certify labels, or implement a raw-alert production service.
"""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from current.history_features import views
from current.score_fusion import agreement

def digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def matrix(a,n,width,name):
    a=np.asarray(a)
    if a.shape!=(n,width) or not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected finite ({n},{width}) numerical matrix')
    return a

def branch_inputs(data):
    entity=np.asarray(data['entity935']);n=len(entity)
    matrix(entity,n,935,'entity935')
    host124=matrix(data['host124'],n,124,'host124')
    roles=matrix(data['roles54'],n,54,'roles54');links=matrix(data['links24'],n,24,'links24')
    all_views=views(np.c_[entity[:,:811],host124],entity)
    host=all_views['host_profile']
    return {'host':host,'cumulative':all_views['profile_contrast'],
            'direct':np.c_[host,roles,links]}

def score(models,matrices,local73,gates3,roles54,links24):
    n=len(matrices['host'])
    local=matrix(local73,n,73,'local73')
    gates=np.asarray(gates3)
    if gates.shape!=(n,3) or gates.dtype!=np.dtype(bool):raise ValueError('gates3 must be Boolean (n,3)')
    if len(models['experts'])!=3:raise ValueError('three protocol experts required')
    expert=np.column_stack([m.predict_proba(local)[:,1] for m in models['experts']])
    q=np.where(gates,expert,0.).max(1)
    adjusted={}
    for name,width in [('host',935),('cumulative',1059),('direct',1013)]:
        x=matrix(matrices[name],n,width,name)
        raw=models[name].predict_proba(x)[:,1]
        adjusted[name]=1.-(1.-raw)*(1.-.5*q)
    role=np.any(matrix(roles54,n,54,'roles54')>0,axis=1)
    relation=np.any(matrix(links24,n,24,'links24')>0,axis=1)
    anchor,host,direct=(adjusted[k] for k in ['cumulative','host','direct'])
    return {'anchor':anchor,'host':host,'direct':direct,
            'correction_off':anchor+.20*(host-anchor),
            'augur':agreement(anchor,host,direct,role,relation)}

def validation_cutoff(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if y.ndim!=1 or p.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all():
        raise ValueError('finite aligned binary validation labels/scores required')
    neg=np.sort(p[y==0])[::-1]
    if len(neg)==0:raise ValueError('validation requires policy negatives')
    return float(np.nextafter(neg[len(neg)//100],np.inf))

def load_frozen(bundle):
    import sklearn,joblib
    root=Path(bundle);receipt=read_json(root/'build_receipt.json')
    if sklearn.__version__!=receipt['software']['scikit_learn']:
        raise RuntimeError('Use the pinned scikit-learn version before loading the frozen estimator')
    for name,expected in receipt['output_sha256'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or digest(p)!=expected:raise ValueError('Frozen input identity mismatch: '+name)
    # joblib/pickle is executable content: only load this audited, trusted fixture.
    return joblib.load(root/'trusted_model.joblib')
