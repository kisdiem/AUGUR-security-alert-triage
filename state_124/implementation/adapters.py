import re,hashlib,json,collections,datetime

def hid(*parts):return hashlib.sha256('\x1f'.join(map(str,parts)).encode()).hexdigest()[:32]
def valid(x):return x is not None and str(x).strip() not in {'','None','null','?','-','4294967295'}
def key(scope,host,kind,value):return kind+':'+hid(scope,host,kind,value)
def kv(text):return {k:(a if a else b) for k,a,b in re.findall(r'(?<!\w)([\w.]+)=(?:"([^"]*)"|([^\s]+))',text)}
def integer(x):
 try:return int(str(x))
 except (TypeError,ValueError):return None
def outcome(value,explicit=None):
 if explicit in {'yes','success',True}:return 'success'
 if explicit in {'no','failed','failure',False}:return 'failure'
 return 'success' if isinstance(value,int) and value>=0 else 'failure' if isinstance(value,int) else 'unknown'
def make(scope,event_id,t,split,actor,target,observations,action,**extra):
 focus=list(dict.fromkeys([actor,target,*[x['key'] for x in observations]]));focus=[x for x in focus if x]
 return {'event_id':event_id,'timestamp_ns':t,'split':split,'actor':actor,'target':target,'focus':focus,'entity_observations':observations,'action':action,**extra}

class AITAdapter:
 def __init__(self,scope):self.scope=scope;self.groups=collections.OrderedDict()
 def checkpoint(self):return list(self.groups.items())[-2048:]
 def restore(self,data):self.groups=collections.OrderedDict(data or [])
 def adapt(self,row):
  text=row['statement'];o=dict(row.get('observed') or {});provenance=row['provenance'];member=str(provenance['source'].get('archive_member','')).split('/');host=o.get('host') or (member[1] if len(member)>1 else 'unknown-host');fields=kv(text)
  observations=[]
  def add(kind,value,attrs=None,quality='observed_identifier'):
   if not valid(value):return None
   k=key(self.scope,host,kind,value);observations.append({'key':k,'kind':kind,'identity_quality':quality,'attributes':attrs or {}});return k
  hostkey=add('host',host,{'name':host});pid=o.get('pid') or fields.get('pid');program=None
  m=re.search(r'\b([\w./-]+)\[(\d+)\]:',text)
  if m:program=m[1];pid=pid or m[2]
  m=re.search(r'\[pid (\d+)(?::tid \d+)?\]',text)
  if m:pid=pid or m[1]
  process_attrs={'pid':str(pid) if pid is not None else None,'program':program,'image':o.get('image') or fields.get('exe'),'observed_uid':fields.get('uid'),'observed_euid':fields.get('euid'),'audit_login_uid':fields.get('auid')}
  actor=add('process',o.get('process_guid'),process_attrs) if valid(o.get('process_guid')) else add('pid_observation',pid,process_attrs,'pid_only_lifetime_unresolved')
  account=add('account',o.get('account') or fields.get('acct') or fields.get('uid'),{'observed_uid':fields.get('uid')})
  session=add('session',o.get('session') or fields.get('ses'));parent=add('pid_observation',fields.get('ppid'),{'pid':fields.get('ppid')},'pid_only_lifetime_unresolved')
  action=o.get('action') or 'unknown';ret=integer(fields.get('exit'));resource=o.get('resource');args=None;audit_group=None;native=o.get('native_type');target=None
  audit=re.search(r'msg=audit\((\d+(?:\.\d+)?):(\d+)\)',text)
  if audit:
   audit_group=host+':'+audit[1]+':'+audit[2];typ=fields.get('type');group=self.groups.get(audit_group,{})
   if actor:group.update(actor=actor,pid=str(pid),image=o.get('image') or fields.get('exe'))
   elif group.get('actor'):
    actor=group['actor'];observations.append({'key':actor,'kind':'pid_observation','identity_quality':'pid_only_lifetime_unresolved','attributes':{'pid':group.get('pid'),'image':group.get('image')}})
   if typ=='SYSCALL' and fields.get('arch')=='c000003e':
    action={'0':'object_read','1':'object_write','2':'object_open','257':'object_open','59':'exec','56':'proc_spawn','57':'proc_spawn','58':'proc_spawn','42':'net_connect','87':'file_delete','60':'process_exit','231':'process_exit'}.get(fields.get('syscall'),action)
    group.update(syscall_action=action,return_value=ret,success=fields.get('success'))
   elif typ=='PATH':resource=fields.get('name');action='audit_path_observation'
   elif typ=='CWD':action='cwd_observation';resource=None
   elif typ=='EXECVE':
    action='execution_arguments_observation';args=[fields.get('a'+str(i)) for i in range(min(integer(fields.get('argc')) or 0,64))]
   elif typ in {'USER_AUTH','USER_ACCT','CRED_ACQ','USER_START','USER_END','USER_LOGIN'}:action={'USER_START':'session_start','USER_END':'session_end','USER_AUTH':'authentication'}.get(typ,'account_observation')
   self.groups[audit_group]=group;self.groups.move_to_end(audit_group)
   while len(self.groups)>2048:self.groups.popitem(last=False)
   native='linux_audit:'+str(typ)
  else:
   m=re.search(r'(query\[[^]]+\]|forwarded|reply|cached)\s+(\S+)\s+(?:from|to|is)\s+(\S+)',text)
   if m and 'dnsmasq' in text:
    action='dns_forward' if m[1]=='forwarded' else 'dns_query' if m[1].startswith('query') else 'dns_observation';o['domain']=m[2];o['dest_ip']=m[3] if m[1]=='forwarded' else o.get('dest_ip');o['dest_port']=53 if m[1]=='forwarded' else o.get('dest_port');native='dnsmasq'
   m=re.search(r'"(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+(\S+)\s+HTTP/[\d.]+"\s+(\d{3})',text)
   if m:action='http_request';o.update(http_method=m[1],http_path=m[2],http_status=int(m[3]));native='http_access'
   m=re.search(r'(Accepted|Failed)\s+(\S+)\s+for\s+(?:invalid user\s+)?(\S+)\s+from\s+(\S+)\s+port\s+(\d+)',text)
   if m:action='authentication';account=add('account',m[3]);o.update(auth_result=m[1],source_ip=m[4],source_port=int(m[5]))
  if valid(resource):
   kind='file' if str(resource).startswith(('/','./','../')) or o.get('object_kind') in {'file','directory'} else 'resource_observation'
   target=add(kind,resource,{'path':resource,'path_is_absolute':str(resource).startswith('/')})
  if valid(o.get('dest_ip')) and str(o['dest_ip']) not in {'0','0.0.0.0'}:
   target=add('netflow',str(o['dest_ip'])+':'+str(o.get('dest_port')),{'destination_ip':o['dest_ip'],'destination_port':o.get('dest_port'),'domain':o.get('domain'),'observation_kind':'endpoint_only_connection_identity_unavailable'},'endpoint_only_not_connection_identity')
  if action=='proc_spawn' and isinstance(ret,int) and ret>0:target=add('pid_observation',str(ret),{'pid':str(ret)},'pid_only_lifetime_unresolved')
  if action=='exec' and not target and valid(o.get('image') or fields.get('exe')):target=add('file',o.get('image') or fields.get('exe'),{'path':o.get('image') or fields.get('exe'),'path_role':'audit_reported_executable_not_verified_exec_target'})
  result=outcome(ret,fields.get('success') or fields.get('res'))
  if o.get('auth_result'):result='success' if o['auth_result']=='Accepted' else 'failure'
  # Every row keeps its original text; derived fields are explicit observations.
  current={'statement':text,'native_type':native,'action':action,'observed':{**o,'audit_fields':{k:v for k,v in fields.items() if k in {'pid','ppid','uid','euid','auid','ses','syscall','success','exit','name','cwd','exe','type','argc'}},'arguments':args},'entity_types':[x['kind'] for x in observations]}
  member=str(provenance['source'].get('archive_member') or '')
  return make(self.scope,row['event_id'],row['timestamp_ns'],provenance['split'],actor,target,observations,action,parent=parent,parent_quality='reported_pid_identity_ambiguous',account=account,session=session,host=hostkey,arguments=args,return_value=ret,outcome=result,audit_group=audit_group,current_event=current,source=provenance['source'],quality={'pid_reuse_unresolved':any(x['kind']=='pid_observation' for x in observations),'sensor_derived_alert':'suricata' in member and 'fast.log' in member,'history_input_quality':row.get('state_quality'),'audit_group_join':'previously_observed_companions_only','raw_source':'retained_original_statement'})

OPS=['EVENT_CONNECT','EVENT_EXECUTE','EVENT_OPEN','EVENT_READ','EVENT_RECVFROM','EVENT_RECVMSG','EVENT_SENDMSG','EVENT_SENDTO','EVENT_WRITE','EVENT_CLONE']
ACTIONS=['net_connect','exec','object_open','object_read','net_receive','net_receive','net_send','net_send','object_write','proc_spawn']

def decode_parameter(p):
 b=p.get('valueBytes');b=bytes.fromhex(b['bytes_hex']) if isinstance(b,dict) and 'bytes_hex' in b else None
 if b is None:return None
 if p.get('valueDataType')=='VALUE_DATA_TYPE_CHAR':
  try:return b.decode('utf-32-be')
  except UnicodeError:return None
 if p.get('valueDataType') in {'VALUE_DATA_TYPE_INT','VALUE_DATA_TYPE_LONG','VALUE_DATA_TYPE_SHORT'} and len(b) in (1,2,4,8):return int.from_bytes(b,'big',signed=True)
 return None

class DARPAAdapter:
 def __init__(self,nodes,raw):self.nodes=nodes;self.raw=raw
 def adapt(self,s,d,op,t,eid,split,source):
  sv,dv=self.nodes[s],self.nodes[d];actor_id=s if sv['type']=='subject' else d if dv['type']=='subject' else None;object_id=d if actor_id==s else s;actor=self.nodes[actor_id] if actor_id is not None else None;obj=self.nodes[object_id];observations=[]
  host='capture-host-unresolved';scope='CLEARSCOPE_E5'
  def add(nid,node):
   kind={'subject':'process','file':'file','netflow':'netflow'}[node['type']];exported={k:v for k,v in node['attrs'].items() if valid(v) and k!='type' and not (k in {'remote_ip','remote_port'} and str(v) in {'0','0.0.0.0'})};attrs={'benchmark_metadata':exported,'metadata_observation_time':'not_exported_by_benchmark'}
   # Current node descriptions are current-event observations, not preloaded past state.
   k=kind+':CLEARSCOPE_E5:'+str(nid);observations.append({'key':k,'kind':kind,'identity_quality':'benchmark_node_identity_host_not_exported','attributes':attrs});return k
  ak=add(actor_id,actor) if actor else None;ok=add(object_id,obj)
  if actor_id is None and d!=object_id:add(d,dv)
  native=OPS[op];action=ACTIONS[op]
  if op==3 and obj['type']=='file':action='file_read'
  if op==8 and obj['type']=='file':action='file_write'
  raw=self.raw.get(eid);args=ret=path=None;detail=None;hk=None
  if raw:
   rd=raw['raw_record']['datum'];assert rd['type']==native and rd['timestampNanos']==t
   ps={p.get('name'):decode_parameter(p) for p in rd.get('parameters') or [] if p.get('name') in {'path','argv','r','byteCount','count','fd','flags','mode'}}
   args=ps.get('argv');args=args.rstrip('\x00').split('\x00') if isinstance(args,str) else None;ret=ps.get('r');path=ps.get('path')
   detail={'parameters':ps,'names':rd.get('names'),'program_point':rd.get('programPoint'),'size':rd.get('size'),'source_file':raw['raw_source_file'],'block_offset':raw['avro_block_offset'],'block_sha256':raw['avro_block_sha256']}
   rawhost=raw['raw_record'].get('hostId')
   if rawhost and rawhost.get('bytes_hex'):
    hk='host:CDM:'+rawhost['bytes_hex'];observations.append({'key':hk,'kind':'host','identity_quality':'raw_host_uuid','attributes':{'uuid':rawhost['bytes_hex']}})
   if observations and rd.get('subject'):
    for observation in observations:
     if observation['key']==ak:observation['attributes']['raw_subject_uuid']=rd['subject']['bytes_hex']
   if rd.get('predicateObjectPath'):
    for observation in observations:
     if observation['key']==ok:observation['attributes']['observed_object_path']=rd['predicateObjectPath']
  current={'native_type':native,'action':action,'source_node':s,'destination_node':d,'subject_description':actor['label'] if actor else None,'object_description':obj['label'],'entity_types':[sv['type'],dv['type']],'arguments':args,'path_argument':path,'return_value':ret,'raw_parameters':detail,'projected_cdm_fields':raw['raw_record'] if raw else None,'statement':json.dumps({'subject':actor['label'] if actor else None,'operation':native,'object':obj['label'],'arguments':args,'path_argument':path,'return_value':ret},ensure_ascii=False)}
  return make(scope,eid,t,split,ak,ok,observations,action,host=hk,arguments=args,return_value=ret,outcome=outcome(ret),current_event=current,source=source,quality={'raw_parameter_recovery':bool(raw),'raw_parameters_otherwise_unknown':not bool(raw),'history_source':'Velox_selected_fused_events_not_all_raw_CDM','benchmark_metadata_time_unknown':True,'host_link_not_inferred':True,'same_timestamp_order':'time_then_uuid_within_original_window'})
