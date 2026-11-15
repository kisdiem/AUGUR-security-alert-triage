import math,re,collections
import numpy as np
DOMAIN_NAMES=('length_without_tld','label_count_without_tld','max_label_length','first_label_length','entropy_longest_label','entropy_without_tld','digit_fraction','upper_fraction','non_hostname_char_fraction','exfil_filename_marker','encoding_special_marker','payload_prefix_length')
QTYPES=('A','AAAA','TXT','CNAME','MX','PTR','SOA','OTHER')
FEATURE_NAMES=tuple('mean:'+n for n in DOMAIN_NAMES)+tuple('max:'+n for n in DOMAIN_NAMES)+tuple('qtype_fraction:'+q for q in QTYPES)+('log_alerts','dns_alert_fraction','log_query_observations','log_unique_query_observations_per_second_name_type','log_unique_domains','unique_domain_fraction','log_max_queries_per_second','log_mean_interval','log_max_interval','observed_query_to_dns_alert_ratio','has_observed_query')
FILE=re.compile(r'\.(?:xlsx?|csv|sql|db|pdf|docx?|zip|gz|tar|txt|pem|key)(?:\.|$)',re.I)
def entropy(s):
 c=collections.Counter(s);n=len(s);return -sum((v/n)*math.log2(v/n) for v in c.values()) if n else 0.
def domain_features(name):
 # No TLD or exact hostname features. The payload before the zone is retained statistically.
 labels=str(name).strip('.').split('.');labels=labels[:-1] if len(labels)>1 else labels;s='.'.join(labels);longest=max(labels,key=len) if labels else '';prefix='.'.join(labels[:-1]);n=max(len(s),1)
 return np.array([len(s),len(labels),len(longest),len(labels[0]) if labels else 0,entropy(longest),entropy(s),sum(x.isdigit() for x in s)/n,sum(x.isupper() for x in s)/n,sum(not(x.isalnum() or x in '.-_') for x in s)/n,int(bool(FILE.search(s))),int(any(c in s for c in '+/=*')),len(prefix)],np.float64)
def queries(source,record):
 data=source.get('data') or {};dns=data.get('dns') or {};out=[]
 def walk(value):
  if isinstance(value,dict):
   if isinstance(value.get('rrname'),str):out.append((value['rrname'],str(value.get('rrtype') or 'OTHER').upper()))
   for k,v in value.items():
    if isinstance(v,(list,dict)):walk(v)
  elif isinstance(value,list):
   for v in value:walk(v)
 if isinstance(dns,dict):
  values=dns.get('query') or dns.get('queries') or (dns if dns.get('type')=='query' else None)
  if values:walk(values)
 if not out:
  m=re.search(r'\bquery\[([^]]+)\]\s+(\S+)\s+from\s+',record.get('raw',''))
  if m and 'dnsmasq' in record.get('raw',''):out.append((m[2],m[1].upper()))
 return out
class Aggregate:
 def __init__(self):self.alerts=0;self.dns_alerts=0;self.query_observations=0;self.seen=set();self.domains=set();self.total=np.zeros(len(DOMAIN_NAMES));self.maximum=np.zeros(len(DOMAIN_NAMES));self.qtypes=collections.Counter();self.seconds=collections.Counter()
 def add(self,record,source):
  self.alerts+=1;self.dns_alerts+=int('dns' in record['text'].lower());qq=queries(source,record);self.query_observations+=len(qq)
  for name,qtype in qq:
   anchor=(int(record['t']),name.lower(),qtype)
   if anchor in self.seen:continue
   self.seen.add(anchor);self.domains.add(name.lower());v=domain_features(name);self.total+=v;self.maximum=np.maximum(self.maximum,v);self.qtypes[qtype if qtype in QTYPES else 'OTHER']+=1;self.seconds[int(record['t'])]+=1
 def vector(self):
  count=len(self.seen);times=sorted(self.seconds);interval=np.diff(times);n=max(count,1)
  extra=[math.log1p(self.alerts),self.dns_alerts/max(self.alerts,1),math.log1p(self.query_observations),math.log1p(count),math.log1p(len(self.domains)),len(self.domains)/n,math.log1p(max(self.seconds.values(),default=0)),math.log1p(float(interval.mean())) if len(interval) else 0.,math.log1p(float(interval.max())) if len(interval) else 0.,count/max(self.dns_alerts,1),float(count>0)]
  v=np.r_[self.total/n,self.maximum,[self.qtypes[q]/n for q in QTYPES],extra].astype(np.float32);assert v.shape==(len(FEATURE_NAMES),) and np.isfinite(v).all();return v
