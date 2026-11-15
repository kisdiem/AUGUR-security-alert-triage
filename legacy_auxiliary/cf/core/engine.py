"""Observed entity environment. No labels, attack rules, or future-event input."""
import collections,copy,json,pathlib,sqlite3

def dumps(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)

class Store:
 def __init__(self,path,cache_limit=16000):
  self.path=pathlib.Path(path);self.db=sqlite3.connect(str(path));self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA cache_size=-16384')
  self.db.execute('CREATE TABLE IF NOT EXISTS entities(k TEXT PRIMARY KEY,v TEXT NOT NULL)')
  self.db.execute('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT NOT NULL)');self.db.commit()
  self.cache=collections.OrderedDict();self.dirty=set();self.limit=cache_limit
 def get(self,key):
  if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
  row=self.db.execute('SELECT v FROM entities WHERE k=?',(key,)).fetchone()
  if row is None:return None
  value=json.loads(row[0]);self.put(key,value,dirty=False);return value
 def put(self,key,value,dirty=True):
  self.cache[key]=value;self.cache.move_to_end(key)
  if dirty:self.dirty.add(key)
  while len(self.cache)>self.limit:
   k,v=self.cache.popitem(last=False)
   if k in self.dirty:self.db.execute('INSERT OR REPLACE INTO entities VALUES(?,?)',(k,dumps(v)));self.dirty.remove(k)
 def reset(self):self.cache.clear();self.dirty.clear();self.db.execute('DELETE FROM entities')
 def flush(self):
  self.db.executemany('INSERT OR REPLACE INTO entities VALUES(?,?)',[(k,dumps(self.cache[k])) for k in self.dirty]);self.dirty.clear()
 def metadata(self,key,default=None):
  row=self.db.execute('SELECT v FROM meta WHERE k=?',(key,)).fetchone();return json.loads(row[0]) if row else default
 def commit(self,checkpoint):
  self.flush();self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('checkpoint',dumps(checkpoint)));self.db.commit()
 def backup(self,path):
  destination=sqlite3.connect(str(path));self.db.backup(destination);destination.close()
 def close(self):self.db.rollback();self.db.close()

class Environment:
 def __init__(self,store,max_entities=12,recent_limit=6):self.store=store;self.max_entities=max_entities;self.recent_limit=recent_limit;self.version=0
 def before(self,event):
  queue=[(x,0) for x in event['focus'] if x];seen=set();entities=[];relations=[]
  while queue and len(entities)<self.max_entities:
   key,depth=queue.pop(0)
   if key in seen:continue
   seen.add(key);obj=self.store.get(key)
   if obj is None:entities.append({'key':key,'previously_observed':False});continue
   view=copy.deepcopy(obj)
   # Short-term entries are views; their expiry never destroys the long-term facts.
   view['recent']=[r for r in view['recent'] if event['timestamp_ns']-r['timestamp_ns']<=43200*10**9]
   view['previously_observed']=True;entities.append(view)
   for relation,record in obj.get('links',{}).items():
    relations.append({'source':key,'relation':relation,**copy.deepcopy(record)})
    if depth<2 and record.get('target'):queue.append((record['target'],depth+1))
   if depth<1:
    for record in reversed(view['recent'][-2:]):
     if record.get('peer'):queue.append((record['peer'],depth+1))
  return {'system_version':self.version,'entities':entities,'relations':relations,'context_limit':self.max_entities,'pending_context_references':len({k for k,_ in queue if k not in seen}),'scope':'observed_local_entity_environment','window_seconds':43200,'missing_is':'unknown_not_benign','complete_operating_system_state':False}
 def apply(self,event):
  changed={};t=event['timestamp_ns'];actor=event.get('actor');target=event.get('target');action=event['action'];evidence={'event_id':event['event_id'],'timestamp_ns':t}
  specs={x['key']:x for x in event['entity_observations']}
  for key in event['focus']:
   if key not in specs:continue
   spec=specs[key];old=self.store.get(key)
   obj=old if old is not None else {'key':key,'kind':spec['kind'],'identity_quality':spec.get('identity_quality','observed_identifier'),'first_seen_ns':t,'last_seen_ns':t,'attributes':{},'links':{},'event_counts':{},'events_observed':0,'recent':[]}
   obj['last_seen_ns']=t;obj['events_observed']+=1;obj['event_counts'][action]=obj['event_counts'].get(action,0)+1
   obj['attributes'].update({k:v for k,v in spec.get('attributes',{}).items() if v is not None})
   peer=target if key==actor else actor if key==target else None
   obj['recent'].append({**evidence,'action':action,'peer':peer,'outcome':event.get('outcome','unknown'),'audit_group':event.get('audit_group')});obj['recent']=obj['recent'][-self.recent_limit:]
   self.store.put(key,obj);changed[key]=obj
  def link(key,name,other,certainty='directly_observed'):
   if key in changed and other:changed[key]['links'][name]={'target':other,'certainty':certainty,**evidence}
  def fact(key,name,value):
   if key in changed:changed[key]['attributes'][name]={'value':value,**evidence}
  success=event.get('outcome')=='success'
  link(actor,'last_observed_target',target);link(target,'last_observed_actor',actor)
  if event.get('parent'):link(actor,'reported_parent',event['parent'],event.get('parent_quality','reported_identity'))
  if event.get('account'):link(actor,'observed_account',event['account'])
  if event.get('session'):link(actor,'observed_session',event['session'])
  if event.get('host'):link(actor,'observed_on_host',event['host']);link(target,'observed_on_host',event['host'])
  if action=='proc_spawn' and actor and target:
   link(target,'creation_attempt_from',actor)
   if success:link(target,'created_by',actor);fact(target,'creation_time_ns',t)
  if action in {'file_create','object_create'}:
   link(target,'creation_attempt_by',actor)
   if success:link(target,'created_by',actor)
  if action in {'file_write','object_write'}:
   link(target,'last_write_attempt_by',actor)
   if success:
    link(target,'last_successful_writer',actor)
    if target in changed:changed[target]['attributes']['observed_write_revision']=changed[target]['attributes'].get('observed_write_revision',0)+1
  if action in {'file_read','object_read'}:
   link(target,'last_read_attempt_by',actor)
   if success:link(target,'last_successful_reader',actor)
  if action=='exec':
   link(actor,'last_execution_attempt',target)
   fact(actor,'last_execution_arguments',event.get('arguments'))
   if success:link(actor,'last_successful_execution',target);link(target,'last_successful_executor',actor)
  if action in {'object_open','file_open'}:
   link(actor,'last_open_attempt',target)
   if success and actor in changed and target and isinstance(event.get('return_value'),int):
    fd=event['return_value'];bindings=changed[actor]['attributes'].setdefault('observed_fd_bindings',{})
    bindings.pop(str(fd),None);bindings[str(fd)]={'target':target,'status':'last_observed_binding_not_proof_still_open',**evidence}
    while len(bindings)>32:bindings.pop(next(iter(bindings)))
  if action in {'net_connect','net_send','net_receive','dns_forward','dns_query'}:
   link(actor,'last_network_object',target)
   if target in changed and action in {'net_send','net_receive'} and success:
    attr='observed_sent_bytes' if action=='net_send' else 'observed_received_bytes';n=event.get('return_value')
    if isinstance(n,int) and n>=0:changed[target]['attributes'][attr]=changed[target]['attributes'].get(attr,0)+n
  if action in {'file_delete','object_delete','process_exit'}:fact(target if action!='process_exit' else actor,'last_lifecycle_attempt',action)
  self.version+=1
  # Full upserts make global-state reconstruction independent of feature code.
  return {'system_version_after':self.version,'entity_upserts':list(changed.values())}
