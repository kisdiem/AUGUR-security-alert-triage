import re,collections,math
import numpy as np
from graph_features import NODE_DIM,RELATIONS
from features import KINDS,UUID,IP
WORDS=re.compile(r'[a-z_][a-z0-9_.-]*',re.I)
NUMERIC_DIM=NODE_DIM*2+len(KINDS)+1+len(RELATIONS)+7
def normalize(text):
 text=UUID.sub('<ID>',str(text));text=IP.sub('<IP>',text);text=re.sub(r'\b\d+\b','<NUM>',text);return ' '.join(text.lower().split())
def words(text):return set(WORDS.findall(text.lower()))
def fit_idf(rows):
 counts=collections.Counter()
 for row in rows:counts.update(row['signatures'])
 n=sum(counts.values());df=collections.Counter()
 for text,count in counts.items():
  for word in words(text):df[word]+=count
 assert n>0
 return {'N':n,'df':dict(df)}
def weight(text,idf):
 tokens=WORDS.findall(text.lower());return sum(math.log(idf['N']/(1+idf['df'].get(t,0))) for t in tokens)/len(tokens) if tokens else 1.
def state_summary(record):
 nodes=np.asarray(record['node_x'],np.float32);edges=np.asarray(record['edges'],np.float32)
 mean=nodes.mean(0) if len(nodes) else np.zeros(NODE_DIM);maximum=nodes.max(0) if len(nodes) else np.zeros(NODE_DIM)
 kinds=np.bincount(record['node_types'],minlength=len(KINDS)+1).astype(float)/max(len(nodes),1)
 relation=np.bincount(edges[:,2].astype(int),minlength=len(RELATIONS)).astype(float) if len(edges) else np.zeros(len(RELATIONS))
 flags=edges[:,4:].mean(0) if len(edges) else np.zeros(5)
 out=np.concatenate([mean,maximum,kinds,np.log1p(relation),flags,[math.log1p(len(nodes)),math.log1p(len(edges))]])
 assert out.shape==(NUMERIC_DIM,) and np.isfinite(out).all();return out.astype(np.float32)
def pool(row,idf,embedding):
 ws={text:weight(text,idf) for text in row['signatures']};denom=sum(ws[t]*n for t,n in row['signatures'].items());fallback=abs(denom)<1e-8
 if fallback:ws={text:1. for text in row['signatures']};denom=sum(row['signatures'].values())
 base=sum((embedding[t]*(ws[t]*n) for t,n in row['signatures'].items()),np.zeros(384,np.float64))/(denom+1e-8)
 state=sum((embedding[text]*(ws[sig]*n) for (sig,text),n in row['state_text'].items()),np.zeros(384,np.float64))/(denom+1e-8)
 augmented=np.concatenate([base,state,row['numeric_mean'],row['numeric_max']]);assert np.isfinite(augmented).all()
 return base.astype(np.float32),augmented.astype(np.float32),fallback
