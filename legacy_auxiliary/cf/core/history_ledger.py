"""Bounded, label-free common history ledger for Hcount/Hflat/Hstate.

Selection uses the same native before entity set. Entity values are rebuilt from
only each selected entity's retained records, so native lifetime attributes and
counters cannot enter the three H representations.
"""
from __future__ import annotations
import collections, copy, hashlib, json

WINDOW_NS=43200*10**9
class MemoryStore:
    def __init__(self): self.data={}
    def get(self,key): return self.data.get(key)
    def put(self,key,value): self.data[key]=value

class CommonLedger:
    def __init__(self, Environment, cache_size=8192):
        self.Environment=Environment; self.events={}; self.queue=collections.deque()
        self.cache=collections.OrderedDict(); self.cache_size=cache_size
        self.digest=hashlib.sha256(); self.counts=collections.Counter()

    def observe(self,event,session_id,raw_id,cf_raw_id):
        eid=event['event_id']; assert eid not in self.events
        self.events[eid]={'event':event,'session_id':session_id,'raw_id':raw_id,'cf_raw_id':cf_raw_id}
        self.queue.append((event['timestamp_ns'],eid))

    def _entity(self, original, refs):
        key=original['key']; signature=(key,tuple(r['event_id'] for r in refs))
        if signature in self.cache:
            self.cache.move_to_end(signature); return self.cache[signature]
        if not refs:
            out={'key':key,'previously_observed':False}
            if 'kind' in original: out['kind']=original['kind']
        else:
            store=MemoryStore(); env=self.Environment(store)
            for ref in sorted(refs,key=lambda r:(r['timestamp_ns'],r['event_id'])):
                original_event=self.events[ref['event_id']]['event']
                specs=[s for s in original_event['entity_observations'] if s['key']==key]
                assert specs, 'Retained entity history must have an actual observation'
                event=dict(original_event,focus=[key],entity_observations=specs)
                env.apply(event)
            out=copy.deepcopy(store.get(key)); assert out is not None
            out['previously_observed']=True
            assert out['events_observed']<=6
        self.cache[signature]=out
        if len(self.cache)>self.cache_size: self.cache.popitem(last=False)
        return out

    def before(self, native_before, current):
        now=current['timestamp_ns']; eid=current['event_id']
        while self.queue and self.queue[0][0]<now-WINDOW_NS:
            _,old=self.queue.popleft(); self.events.pop(old,None)
        assert len(native_before['entities'])<=12
        entities=[]; unique={}; evidence=[]
        for entity in native_before['entities']:
            refs=[]
            for ref in entity.get('recent',[]):
                rid=ref['event_id']; delta=now-ref['timestamp_ns']
                assert rid!=eid and 0<=delta<=WINDOW_NS
                assert rid in self.events, 'Missing past observation in ledger'
                assert (ref['timestamp_ns'],rid)<(now,eid), 'Not a stable-order predecessor'
                refs.append(ref); unique[rid]=self.events[rid]
            assert len(refs)<=6
            entities.append(self._entity(entity,refs))
            evidence.append([entity['key'],[r['event_id'] for r in refs]])
        self.digest.update(json.dumps([eid,evidence],separators=(',',':')).encode()+b'\n')
        self.counts['current_events']+=1; self.counts['prior_references']+=len(unique)
        self.counts['max_unique_prior']=max(self.counts['max_unique_prior'],len(unique))
        self.counts['max_entities']=max(self.counts['max_entities'],len(entities))
        bounded={'system_version':native_before['system_version'],'entities':entities,
                 'relations':[],'context_limit':12,'pending_context_references':0,
                 'scope':'same_entity_retained_record_ledger','window_seconds':43200,
                 'missing_is':'unknown_not_benign','complete_operating_system_state':False}
        return bounded,unique

    def receipt(self):
        return {'schema':'common_ledger_v1','window_seconds':43200,'max_entities':12,
                'recent_per_entity':6,'ledger_stream_sha256':self.digest.hexdigest(),
                'statistics':dict(self.counts),'labels_read':False,
                'native_lifetime_attributes_reused':False,
                'selector_limitation':'The common entity set is selected by native before traversal; selected values are rebuilt only from retained records.'}
