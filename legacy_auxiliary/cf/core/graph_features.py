"""Observed local environment graph; all identities are join-only, never features."""
import math
import numpy as np
from features import project,tokens,kind,KINDS,BASE,ACTIONS,LINKS
NODE_DIM=len(BASE)+2*len(ACTIONS)+len(LINKS)+1
RELATIONS=LINKS+tuple('recent:'+x for x in ACTIONS)+('observed_fd_binding','unknown_relation')
def index(values,value):return values.index(value) if value in values else len(values)-1
def age(t,previous):return min(math.log1p(max((t-previous)/1e9,0))/10,3) if isinstance(previous,int) else 0.
def number(v):return min(math.log1p(max(float(v or 0),0))/10,3)
def graph(row):
 old=project(row);state=row['state_before'];entities=state['entities'];assert len(entities)<=12
 keys={e['key']:i for i,e in enumerate(entities)};t=row['timestamp_ns'];node_bags=[];node_x=[];node_types=[];edges=[];omitted=0
 def edge(source,target,relation,when,outcome='unknown',certainty=None):
  target_kind=kind(str(target or '').split(':',1)[0]);destination=keys.get(target,-1)
  edge_type=index(RELATIONS,relation);target_type=KINDS.index(target_kind)+1 if target_kind in KINDS else 0
  # Retain links outside the exported 12-entity scope as typed boundary links.
  edges.append((source,destination,edge_type,target_type,age(t,when),int(isinstance(when,int)),int(outcome=='success'),int(outcome=='failure'),int(certainty in ('directly_observed','observed_identifier'))))
 for i,e in enumerate(entities):
  k=kind(e.get('kind') or e['key'].split(':',1)[0]);typ=KINDS.index(k)+1 if k in KINDS else 0;node_types.append(typ);a=e.get('attributes') or {};known=bool(e.get('previously_observed'));uid=a.get('observed_euid') if a.get('observed_euid') is not None else a.get('observed_uid')
  v=[float(known),number(e.get('events_observed')),age(t,e.get('first_seen_ns')),age(t,e.get('last_seen_ns')),number(a.get('observed_write_revision')),float(uid is not None),float(str(uid)=='0' if uid is not None else False),float(e.get('identity_quality')=='pid_only_lifetime_unresolved')]
  counts=e.get('event_counts') or {};v.extend(number(counts.get(action,0)) for action in ACTIONS)
  recent={action:0 for action in ACTIONS}
  for event in e.get('recent',[]):
   if 0<=t-event['timestamp_ns']<=43200*10**9:
    if event['action'] in recent:recent[event['action']]+=1
    edge(i,event.get('peer'),'recent:'+event['action'],event['timestamp_ns'],event.get('outcome','unknown'))
  v.extend(number(recent[action]) for action in ACTIONS);v.extend(float(r in e.get('links',{})) for r in LINKS)
  fd=a.get('observed_fd_bindings') or {};v.append(number(len(fd)));assert len(v)==NODE_DIM;node_x.append(v)
  words=[k,'previously_observed' if known else 'previously_unobserved']
  for name in ('program','image','path','observed_object_path'):
   if a.get(name):words.extend([name]+tokens(a[name],16))
  args=a.get('last_execution_arguments')
  if isinstance(args,dict) and args.get('value'):words.extend(['execution_arguments']+tokens(args['value'],24))
  if uid is not None:words.extend(['privilege','root' if str(uid)=='0' else 'nonroot'])
  node_bags.append(words[:80])
  for relation,record in e.get('links',{}).items():edge(i,record.get('target'),relation,record.get('timestamp_ns'),certainty=record.get('certainty'))
  for record in fd.values():edge(i,record.get('target'),'observed_fd_binding',record.get('timestamp_ns'))
 # Full exported entity records are projected; no state_delta/current-event update.
 return {'id':old['id'],'bags':old['bags'][:5],'types':old['types'],'action':old['action'],'node_bags':node_bags,'node_x':np.asarray(node_x,np.float16).reshape(-1,NODE_DIM),'node_types':node_types,'edges':np.asarray(edges,np.float32).reshape(-1,9),'boundary_links':sum(e[1]<0 for e in edges)}
