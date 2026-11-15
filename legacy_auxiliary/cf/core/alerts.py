"""Read only an explicit whitelist of observable alert fields, never attack labels."""
import datetime,re,hashlib,json
UTC=datetime.timezone.utc
def iso(value):
 stamp=re.sub(r'([+-]\d{2})(\d{2})$',r'\1:\2',str(value).replace('Z','+00:00'))
 parsed=datetime.datetime.fromisoformat(stamp)
 return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).timestamp()
def parse(row,detector_family,scene,line):
 if detector_family=='aminer':
  t=float(row['LogData']['DetectionTimestamp'][-1]);host=str(row['AMiner']['ID']);name=str(row['AnalysisComponent']['AnalysisComponentName']);detector='AMiner';raw=str(row['LogData'].get('RawLogData',[''])[0]);location=str(row['LogData'].get('LogResources',[''])[0]);program=None
 else:
  data=row.get('data') or {};pre=row.get('predecoder') or {};detector='Suricata' if str(row['rule']['description']).startswith('Suricata:') else 'Wazuh'
  host=str(row['agent']['ip']);name=str(row['rule']['description']);raw=str(row.get('full_log') or '');location=str(row.get('location') or '');program=pre.get('program_name')
  if data.get('timestamp'):t=iso(data['timestamp'])
  elif pre.get('timestamp'):
   stamp=str(pre['timestamp']);fmt='%m/%d/%Y-%H:%M:%S.%f' if '/' in stamp else '%Y %b %d %H:%M:%S';t=datetime.datetime.strptime(stamp if '/' in stamp else '2022 '+stamp,fmt).replace(tzinfo=UTC).timestamp()
  elif row.get('@timestamp'):t=iso(row['@timestamp'])
  elif row.get('timestamp') and str(row['timestamp']).startswith('2022'):t=iso(row['timestamp'])
  else:
   match=re.search(r'msg=audit\((\d+(?:\.\d+)?):',raw)
   if match:t=float(match[1])
   else:raise ValueError('No supported event-time source; do not guess')
 if not host or host in ('None','null','unknown'):raise ValueError('Unknown alert source host')
 # Signature is the only raw textual event sentence; identities join state/session only.
 text=name if name.startswith(detector+':') else detector+': '+name
 eid=hashlib.sha256(f'{scene}:{detector_family}:{line}'.encode()).hexdigest()[:32]
 return {'id':eid,'t':t,'host':host,'text':text,'raw':raw,'location':location,'program':program,'detector':detector}
def adapted(record,adapter):
 observed={'host':record['host'],'action':'unknown','native_type':'alert_record','image':record.get('program')}
 raw=record['raw']
 # An alert without embedded raw evidence only updates the observed host, not invented process facts.
 event=adapter.adapt({'statement':raw,'observed':observed,'event_id':record['id'],'timestamp_ns':int(record['t']*1e9),'provenance':{'split':'unassigned','source':{'archive_member':'alerts/'+record['host']+'/'+record['location']}}})
 event['current_event']['statement']=record['text'];event['current_event']['native_type']='alert_'+record['detector'].lower()
 return event
