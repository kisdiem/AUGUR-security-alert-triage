"""Recompute CERT score-only metrics locally from the supplied result tree.

Input files remain local. Output contains aggregated metrics only.
File naming: <root>/<arm>_<seed>/test_predictions.npz and metrics.json.
This audit does not establish data redistribution rights.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, json, sys
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_curve

ARMS=('et_current','et_session','et_state_12h','et_state_full')
SEEDS=(42,1337,31415,2025,8675309)

def audit(root, arm, seed):
    p=root/(arm+'_'+str(seed))
    with np.load(p/'test_predictions.npz',allow_pickle=False) as z:
        y=z['y']; score=z['score']; valid=y>=0
        y=y[valid]; score=score[valid]
        ap=float(average_precision_score(y,score))
        fpr,tpr,_=roc_curve(y,score,drop_intermediate=False)
        recall=float(tpr[fpr<=0.01].max())
        count_pos=int((y==1).sum()); count_neg=int((y==0).sum())
    reference=json.loads((p/'metrics.json').read_text(encoding='utf-8'))
    return dict(arm=arm,seed=seed,ap=ap,recall_at_1pct_fpr=recall,
        positives=count_pos,negatives=count_neg,
        ap_matches_metrics=bool(abs(ap-reference['AP'])<1e-10))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(lambda x:audit(args.root,*x),[(a,s) for a in ARMS for s in SEEDS]))
    result=dict(method='sklearn average_precision_score; roc_curve(drop_intermediate=False); max TPR where FPR <= 0.01; exclude y < 0',
      numpy_version=np.__version__,sklearn_version=sklearn.__version__,rows=rows)
    result['group_means']={a:dict(AP=float(np.mean([r['ap'] for r in rows if r['arm']==a])),recall=float(np.mean([r['recall_at_1pct_fpr'] for r in rows if r['arm']==a]))) for a in ARMS}
    result['all_ap_match']=all(r['ap_matches_metrics'] for r in rows)
    text=json.dumps(result,indent=2)+'\n'
    if args.out: args.out.write_text(text,encoding='utf-8')
    else: sys.stdout.write(text)
    if not result['all_ap_match']: raise SystemExit(1)
if __name__=='__main__': main()
