"""Label-blind projection of v3 observed event and PRIOR environment only."""
import re,math,collections,hashlib
import numpy as np
KINDS=('process','file','netflow','account','session','host','resource_observation')
ACTIONS=('unknown','exec','proc_spawn','file_read','file_write','file_create','file_delete','object_open','object_read','object_write','object_create','object_delete','process_exit','net_connect','net_send','net_receive','authentication','session_start','session_end','account_observation','audit_path_observation','cwd_observation','execution_arguments_observation','dns_query','dns_forward','dns_observation','http_request')
LINKS=('last_observed_target','last_observed_actor','reported_parent','observed_account','observed_session','observed_on_host','creation_attempt_from','creation_attempt_by','created_by','last_write_attempt_by','last_successful_writer','last_read_attempt_by','last_successful_reader','last_execution_attempt','last_successful_execution','last_successful_executor','last_open_attempt','last_network_object')
BASE=('present','log_events','log_since_first','log_since_last','log_write_revision','uid_known','root_uid','pid_ambiguous')
NAMES=tuple(f'{kind}:{name}' for kind in KINDS for name in BASE+tuple('long:'+a for a in ACTIONS)+tuple('recent:'+a for a in ACTIONS)+tuple('relation:'+r for r in LINKS))
INDEX={name:i for i,name in enumerate(NAMES)}
TOKEN=re.compile(r'[a-zA-Z_][a-zA-Z_0-9.-]*|/|[^\w\s]',re.ASCII)
UUID=re.compile(r'(?i)\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b|\b[0-9a-f]{24,}\b')
IP=re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
def kind(value):return 'process' if value in ('process','pid_observation','subject') else value
def tokens(text,limit):
 text=UUID.sub('identifier',str(text or '')[:4096]);text=IP.sub('ipaddress',text);text=re.sub(r'\b\d+\b','number',text)
 return TOKEN.findall(text.lower())[:limit]
def sensor_alert(row):
 member=str(row.get('source',{}).get('archive_member','')).lower()
 return bool(row.get('quality',{}).get('sensor_derived_alert')) or 'suricata' in member
def stratum(row):
 current=row['current_event'];o=current.get('observed') or {};state=row['state_before'];entities=state['entities']
 host=str(o.get('host') or 'unknown');family=str(current.get('native_type') or 'unknown')
 # Source path used for sampling audit only; never added to model features.
 if family in ('unknown','unstructured_text'):
  member=str(row.get('source',{}).get('archive_member',''));family=re.sub(r'(?:\.\d+|\.gz)$','',member.rsplit('/',1)[-1])
 types=','.join(sorted(set(kind(x) for x in current.get('entity_types',[]))))
 coverage=sum(bool(x.get('previously_observed')) for x in entities);coverage='0' if coverage==0 else '1' if coverage==1 else '2+'
 return '|'.join((host,family,types,coverage))
def project(row):
 state=row.get('state_before');assert state is not None,'Unknown-time events not supervised'
 current=row['current_event'];o=current.get('observed') or {};t=row['timestamp_ns'];x=np.zeros(len(NAMES),np.float32);history=[]
 def add(k,name,value):
  ix=INDEX.get(k+':'+name)
  if ix is not None:x[ix]+=float(value)
 def count(v):return min(math.log1p(max(float(v or 0),0))/10,3)
 for entity in state['entities']:
  k=kind(entity.get('kind') or str(entity.get('key','')).split(':',1)[0])
  if k not in KINDS or not entity.get('previously_observed'):continue
  a=entity.get('attributes') or {};add(k,'present',1);add(k,'log_events',count(entity.get('events_observed')))
  for field,name in [('first_seen_ns','log_since_first'),('last_seen_ns','log_since_last')]:
   if field in entity:add(k,name,count((t-entity[field])/1e9))
  add(k,'log_write_revision',count(a.get('observed_write_revision')))
  uid=a.get('observed_euid') if a.get('observed_euid') is not None else a.get('observed_uid')
  if uid is not None:add(k,'uid_known',1);add(k,'root_uid',str(uid)=='0')
  add(k,'pid_ambiguous',entity.get('identity_quality')=='pid_only_lifetime_unresolved')
  for action,n in (entity.get('event_counts') or {}).items():add(k,'long:'+action,count(n))
  recent=collections.Counter(e['action'] for e in entity.get('recent',[]) if 0<=t-e['timestamp_ns']<=43200*10**9)
  for action,n in recent.items():add(k,'recent:'+action,count(n))
  for relation in entity.get('links',{}):add(k,'relation:'+relation,1);history.extend((k,relation))
  # No entity keys, UUIDs, host/scenario names, annotation, or global benchmark metadata.
  for key in ('program','image','path','observed_object_path'):
   if a.get(key):history.extend([k]+tokens(a[key],12))
  arguments=a.get('last_execution_arguments')
  if isinstance(arguments,dict):history.extend(tokens(arguments.get('value'),16))
 x=np.minimum(x,5).astype(np.float16)
 present={kind(v) for v in current.get('entity_types',[])};types=[int(k in present) for k in ('process','file','netflow')]
 text=str(current.get('statement') or '')
 # Remove volatile syslog header; normalize exact host name and numeric identifiers.
 text=re.sub(r'^[A-Z][a-z]{2}\s+\d+\s+[\d:]+\s+\S+\s+','',text)
 if o.get('host'):text=text.replace(str(o['host']),'host')
 subject=o.get('image') or o.get('command') or current.get('subject_description') or ''
 if not subject:
  match=re.match(r'([\w./-]+)(?:\[\d+\])?:',text)
  if match:subject=match[1]
 file=o.get('resource') or current.get('path_argument') or current.get('object_description') or ''
 endpoint=o.get('dest_ip') or ''
 action=current.get('action') or 'unknown';action_ix=ACTIONS.index(action) if action in ACTIONS else 0
 bags=[['subject']+tokens(subject,24) if types[0] else [],['file']+tokens(file,24) if types[1] else [],['netflow']+tokens(endpoint,16) if types[2] else [],['verb']+action.split('_'),['event']+tokens(text,96),['history']+history[:96]]
 return {'id':row['event_id'],'bags':bags,'types':types,'action':action_ix,'x':x}
def stable_fraction(eid):return int.from_bytes(hashlib.blake2b(eid.encode(),digest_size=8).digest(),'big')/2**64
