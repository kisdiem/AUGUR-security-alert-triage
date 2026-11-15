"""Small cumulative-state backend prototype; no text, graph edges, or labels.

Retains the old entity selector's relationship slots and two recent peer entries.
Six entries are required only when native recent-count readouts are requested.
Real AIT replay equivalence is pending; differential tests are structural checks.
"""
from dataclasses import dataclass, field
import math
import numpy as np

ACTIONS=('unknown','exec','proc_spawn','file_read','file_write','file_create','file_delete',
    'object_open','object_read','object_write','object_create','object_delete','process_exit',
    'net_connect','net_send','net_receive','authentication','session_start','session_end',
    'account_observation','audit_path_observation','cwd_observation','execution_arguments_observation',
    'dns_query','dns_forward','dns_observation','http_request')
WINDOW_NS=43200*10**9


@dataclass(slots=True)
class Link:
    target:str
    timestamp_ns:int
    event_id:str
    certainty:str


@dataclass(slots=True)
class Recent:
    timestamp_ns:int
    peer:str|None
    action:str


@dataclass(slots=True)
class Entity:
    first:int
    last:int
    observations:int=0
    counts:dict=field(default_factory=dict)
    links:dict=field(default_factory=dict)
    recent:list=field(default_factory=list)


def number(value):return min(math.log1p(max(float(value or 0),0))/10,3)


class MinimalEnvironment:
    def __init__(self,max_entities=12,recent_capacity=2,count_factory=dict):
        if recent_capacity<2:raise ValueError('Two peer slots are necessary to preserve the registered selector')
        self.entities={};self.max_entities=max_entities;self.recent_capacity=recent_capacity
        self.version=0;self.last_timestamp=None;self.count_factory=count_factory

    def before(self,event,include_recent=False):
        now=event['timestamp_ns']
        if self.last_timestamp is not None and now<self.last_timestamp:raise ValueError('Observations must be chronologically ordered')
        if include_recent and self.recent_capacity!=6:raise ValueError('Native recent-count projection requires six entries')
        queue=[(key,0) for key in event['focus'] if key];seen=set();keys=[];rows=[]
        while queue and len(keys)<self.max_entities:
            key,depth=queue.pop(0)
            if key in seen:continue
            seen.add(key);keys.append(key);obj=self.entities.get(key)
            if obj is None:rows.append([0.]*(58 if include_recent else 31));continue
            recent=[r for r in obj.recent if now-r.timestamp_ns<=WINDOW_NS]
            values=[1.,number(obj.observations),number((now-obj.first)/1e9),number((now-obj.last)/1e9)]
            counts=obj.counts.read() if hasattr(obj.counts,'read') else obj.counts
            values.extend(number(counts.get(a,0)) for a in ACTIONS)
            if include_recent:
                counts={a:0 for a in ACTIONS}
                for r in recent:
                    if r.action in counts:counts[r.action]+=1
                values.extend(number(counts[a]) for a in ACTIONS)
            rows.append(values)
            if depth<2:
                for link in obj.links.values():
                    if link.target:queue.append((link.target,depth+1))
            if depth<1:
                for r in reversed(recent[-2:]):
                    if r.peer:queue.append((r.peer,depth+1))
        # Match the archived graph projection's per-entity float16 rounding.
        return keys,np.asarray(rows,dtype=np.float16).astype(np.float32).reshape(-1,58 if include_recent else 31)

    def apply(self,event):
        t=event['timestamp_ns']
        if self.last_timestamp is not None and t<self.last_timestamp:raise ValueError('Observations must be chronologically ordered')
        self.last_timestamp=t;actor=event.get('actor');target=event.get('target');action=event['action']
        specs={x['key']:x for x in event['entity_observations']};changed=set()
        for key in event['focus']:
            if key not in specs:continue
            if key not in self.entities:self.entities[key]=Entity(t,t,counts=self.count_factory())
            obj=self.entities[key];obj.last=t;obj.observations+=1
            if action in ACTIONS:
                if hasattr(obj.counts,'increment'):obj.counts.increment(action,t)
                else:obj.counts[action]=obj.counts.get(action,0)+1
            peer=target if key==actor else actor if key==target else None
            obj.recent.append(Recent(t,peer,action));obj.recent=obj.recent[-self.recent_capacity:];changed.add(key)
        def link(key,name,other,certainty='directly_observed'):
            if key in changed and other:self.entities[key].links[name]=Link(other,t,event['event_id'],certainty)
        success=event.get('outcome')=='success'
        link(actor,'last_observed_target',target);link(target,'last_observed_actor',actor)
        if event.get('parent'):link(actor,'reported_parent',event['parent'],event.get('parent_quality','reported_identity'))
        if event.get('account'):link(actor,'observed_account',event['account'])
        if event.get('session'):link(actor,'observed_session',event['session'])
        if event.get('host'):link(actor,'observed_on_host',event['host']);link(target,'observed_on_host',event['host'])
        if action=='proc_spawn':
            link(target,'creation_attempt_from',actor)
            if success:link(target,'created_by',actor)
        if action in {'file_create','object_create'}:
            link(target,'creation_attempt_by',actor)
            if success:link(target,'created_by',actor)
        if action in {'file_write','object_write'}:
            link(target,'last_write_attempt_by',actor)
            if success:link(target,'last_successful_writer',actor)
        if action in {'file_read','object_read'}:
            link(target,'last_read_attempt_by',actor)
            if success:link(target,'last_successful_reader',actor)
        if action=='exec':
            link(actor,'last_execution_attempt',target)
            if success:link(actor,'last_successful_execution',target);link(target,'last_successful_executor',actor)
        if action in {'object_open','file_open'}:link(actor,'last_open_attempt',target)
        if action in {'net_connect','net_send','net_receive','dns_forward','dns_query'}:link(actor,'last_network_object',target)
        self.version+=1


class SessionAccumulator:
    """Pool observations online; retain no per-observation snapshots or text."""
    def __init__(self,width=31):
        self.width=width;self.n=0;self.sums=np.zeros((2,width),np.float64);self.maximum=np.zeros((2,width),np.float32)
    def add(self,node_values):
        assert node_values.ndim==2 and node_values.shape[1]==self.width
        v=np.stack([node_values.mean(0),node_values.max(0)]) if len(node_values) else np.zeros((2,self.width),np.float32)
        self.sums+=v;np.maximum(self.maximum,v,out=self.maximum);self.n+=1
    def features(self,arm):
        if self.n==0:raise ValueError('Empty session')
        pool=np.concatenate([(self.sums/self.n).astype(np.float32),self.maximum],axis=0)
        mean=pool[0,4:31];peak=pool[3,4:31]
        if arm=='activity':return pool[:,:4].reshape(-1)
        if arm=='activity_actions':return np.r_[pool[:,:4].reshape(-1),pool[:,4:31].reshape(-1)]
        if arm=='counts108':return pool[:,4:31].reshape(-1)
        if arm=='counts54_mean_peak':return np.r_[mean,peak]
        if arm=='counts27_peak':return peak.copy()
        if arm=='counts27_mean':return mean.copy()
        if arm=='ever27':return (peak>0).astype(np.float32)
        if arm=='counts27_peak_activity4':return np.r_[peak,pool[3,:4]]
        if arm in {'recent27_peak','counts27_peak_recent27'}:
            if self.width!=58:raise ValueError('Recent fields were not retained')
            recent=pool[3,31:58]
            return recent.copy() if arm=='recent27_peak' else np.r_[peak,recent]
        raise ValueError(arm)
