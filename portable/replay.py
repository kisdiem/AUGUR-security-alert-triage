"""Generate historical features from an ordered, adapted observation JSONL.

Input rows contain session_id, event, and optional close_session. All observations
update state, including rows with no supervised session. Labels are never read.
This consumes the adapter schema, not arbitrary provider files or raw messages.
"""
from pathlib import Path
import argparse,json,sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from state_124.implementation.minimal_state_engine import MinimalEnvironment,SessionAccumulator
from current.direct_entity_state import EntityHistory,ROLE_WIDTH,WIDTH

class DirectPool:
    def __init__(self):self.n=0;self.total=np.zeros(WIDTH,dtype=np.float64);self.peak=np.zeros(WIDTH,dtype=np.float32)
    def add(self,row):self.n+=1;self.total+=row;np.maximum(self.peak,row,out=self.peak)
    def features(self):
        mean=(self.total/self.n).astype(np.float32)
        return np.r_[mean[:ROLE_WIDTH],self.peak[:ROLE_WIDTH]],np.r_[mean[ROLE_WIDTH:],self.peak[ROLE_WIDTH:]]

def replay(rows):
    entity=MinimalEnvironment();host=MinimalEnvironment(max_entities=1);direct=EntityHistory()
    open_sessions={};closed=set();results=[];previous=None;count=0
    def finish(sid):
        h,e,d,n=open_sessions.pop(sid);roles,links=d.features()
        results.append((sid,h.features('activity_actions'),e.features('activity_actions'),roles,links,n))
        closed.add(sid)
    for row in rows:
        event=row['event'];now=event['timestamp_ns'];sid=row.get('session_id')
        if not isinstance(now,int) or (previous is not None and now<previous):raise ValueError('Input timestamps must be nondecreasing')
        previous=now;count+=1
        host_event=dict(event);host_event['focus']=[event['host']] if event.get('host') else []
        # Both readouts precede all writes for this observation. Equal timestamps
        # preserve the supplied source order, as required by the AIT stream.
        if sid is not None:
            if not isinstance(sid,str) or sid in closed:raise ValueError('Invalid or reopened session')
            if sid not in open_sessions:open_sessions[sid]=[SessionAccumulator(),SessionAccumulator(),DirectPool(),0]
            h,e,d,n=open_sessions[sid]
            h.add(host.before(host_event)[1]);e.add(entity.before(event)[1]);d.add(direct.before(event));open_sessions[sid][3]=n+1
        host.apply(host_event);entity.apply(event);direct.apply(event)
        if row.get('close_session'):
            if sid is None:raise ValueError('Closing requires session_id')
            finish(sid)
    for sid in list(open_sessions):finish(sid)
    if not results:raise ValueError('No sessions to emit')
    ids,h,e,r,l,n=zip(*results,strict=True)
    return dict(ids=np.asarray(ids),host124=np.stack(h),entity124=np.stack(e),roles54=np.stack(r),links24=np.stack(l),observations=np.asarray(n)),count

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    with a.input.open(encoding='utf-8') as f:arrays,n=replay(json.loads(line) for line in f if line.strip())
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**arrays)
    print(json.dumps({'status':'PASS','observations':n,'sessions':len(arrays['ids']),'labels_used':False,
        'scope':'Ordered adapter-schema observations to historical features; no text embedding, classifier or raw-source reconstruction.'}))
if __name__=='__main__':main()
