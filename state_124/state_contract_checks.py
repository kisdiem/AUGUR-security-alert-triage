"""Executable interface examples; no attack labels or detection measurements."""
from pathlib import Path
import collections
import importlib.util
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent

def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'implementation'/filename)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def event(i, seconds, action, outcome='unknown', actor='process:p', target='file:f'):
    focus=list(dict.fromkeys([actor,target]))
    return {'event_id':f'fixture-{i}','timestamp_ns':int(seconds*1e9),
            'actor':actor,'target':target,'focus':focus,
            'entity_observations':[{'key':k} for k in focus],
            'action':action,'outcome':outcome}

def main():
    engine=load('review_artifact_state','minimal_state_engine.py')
    env=engine.MinimalEnvironment()
    history=[]
    checks=[]
    stream=[event(0,1,'file_write'),event(1,2,'file_read'),event(2,3,'exec','failure')]
    stream += [event(i,i+1,'net_send',target='netflow:n') for i in range(3,13)]
    stream += [event(13,24*3600+1,'object_read'),event(14,24*3600+2,'file_write','success')]

    # Independent batch fold over the observation prefix, including observed
    # timestamps and action values. No use of the maintained counter dictionary.
    def from_ledger(key, now):
        rows=[e for e in history if key in e['focus']]
        if not rows:return np.zeros(31,dtype=np.float32)
        fields=[1.,len(rows),(now-rows[0]['timestamp_ns'])/1e9,
                (now-rows[-1]['timestamp_ns'])/1e9]
        counts=collections.Counter(e['action'] for e in rows)
        fields += [counts[a] for a in engine.ACTIONS]
        scaled=[1.]+[min(math.log1p(max(float(x),0))/10,3) for x in fields[1:]]
        return np.array(scaled,dtype=np.float16).astype(np.float32)

    pool=engine.SessionAccumulator()
    batch_vectors=[]
    for e in stream:
        keys,rows=env.before(e)
        expected=np.stack([from_ledger(k,e['timestamp_ns']) for k in keys])
        np.testing.assert_array_equal(rows,expected)
        pool.add(rows)
        batch_vectors.append(np.stack([rows.mean(0),rows.max(0)]))
        if e['event_id']=='fixture-0':assert np.count_nonzero(rows)==0
        if e['event_id']=='fixture-13':
            p=rows[keys.index('process:p')]
            assert p[4+engine.ACTIONS.index('file_write')]>0
            assert not any(r.action=='file_write' for r in env.entities['process:p'].recent)
        if e['event_id']=='fixture-14':
            assert 'last_successful_writer' not in env.entities['file:f'].links
        env.apply(e)
        history.append(e)
    assert env.entities['file:f'].links['last_successful_writer'].target=='process:p'
    checks += ['pre-event output equals independent prefix ledger at every fixture observation',
               'first observation does not enter its own history',
               'early action survives more than six later observations and more than twelve hours',
               'unknown and failed outcomes do not create successful-writer facts']

    batch=np.stack(batch_vectors)
    folded=np.concatenate([batch.astype(np.float64).mean(0).astype(np.float32),batch.max(0)])
    expected=np.r_[folded[:,:4].reshape(-1),folded[:,4:31].reshape(-1)]
    np.testing.assert_array_equal(pool.features('activity_actions'),expected)
    assert expected.shape==(124,)
    checks.append('online session accumulator equals batch two-level mean/max projection')

    second=engine.MinimalEnvironment()
    for e in stream:second.apply({**e,'label':'arbitrary-fixture-only'})
    query=event(15,24*3600+3,'unknown')
    a,va=env.before(query);b,vb=second.before(query)
    assert a==b;np.testing.assert_array_equal(va,vb)
    checks.append('label field does not affect observation state')

    bounded=engine.MinimalEnvironment(max_entities=1)
    keys,_=bounded.before(query)
    assert keys==['process:p']
    try:
        env.before(event(99,0,'unknown'))
        raise AssertionError('Out-of-order observation accepted')
    except ValueError:pass
    checks += ['query entity limit is enforced','out-of-order timestamps are rejected']
    cf=load('review_artifact_cf','strict_cf_norm.py')._self_test()
    assert cf['passed']
    return {'passed':True,'state_checks':checks,'fixture_observations':len(stream),
            'cf_contract':cf,'scope':'Deterministic implementation-contract examples only; synthetic unlabeled observations are not experimental attacks, new dataset labels, or detection evidence.'}

if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,indent=2))
