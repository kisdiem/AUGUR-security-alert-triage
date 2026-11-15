"""Recompute AutoLabel window metrics from supplied, locally held unit NPZ files.

Reads only; emits group aggregate results without row IDs or per-window values.
Input schema: <root>/<kind>__<group>/seed_<seed>/<arm>/test/*/*_units.npz.
"""
import argparse,json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score,roc_curve

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path)
    args=ap.parse_args(); rows=[]
    for fold in sorted(args.root.iterdir()):
        if not (fold.name.startswith('application__') or fold.name.startswith('cve_group__')): continue
        for seed in (42,1337,31415):
            for arm in ('current','state'):
                d=fold/('seed_'+str(seed))/arm
                files=sorted((d/'test').glob('*/*_units.npz'))
                if not files: raise ValueError('No test units: '+fold.name+' '+str(seed)+' '+arm)
                ys=[]; ss=[]
                for p in files:
                    with np.load(p,allow_pickle=False) as z:
                        ys.append(z['window_y']); ss.append(z['window_score'])
                y=np.concatenate(ys);s=np.concatenate(ss); valid=y>=0;y=y[valid];s=s[valid]
                score=float(average_precision_score(y,s))
                f,t,_=roc_curve(y,s,drop_intermediate=False)
                rec=float(t[f<=.01].max())
                reference=json.loads((d/'metrics.json').read_text(encoding='utf8'))
                expected=reference['test_pooled']['window']['ap']
                rows.append(dict(fold=fold.name,seed=seed,arm=arm,ap=score,recall_at_1pct_fpr=rec,
                   positive=int((y==1).sum()),negative=int((y==0).sum()),unit_files=len(files),
                   ap_matches_metrics=bool(abs(score-expected)<1e-10)))
    assert len(rows)==54 and all(r['ap_matches_metrics'] for r in rows), 'missing folds or AP mismatch'
    summary={}
    for kind in ('application','cve_group'):
        for arm in ('current','state'):
            group=[r for r in rows if r['fold'].startswith(kind+'__') and r['arm']==arm]
            summary[kind+'/'+arm]=dict(ap=float(np.mean([r['ap'] for r in group])),recall_at_1pct_fpr=float(np.mean([r['recall_at_1pct_fpr'] for r in group])))
    result=dict(method='window-level sklearn AP and max TPR where FPR <= 0.01; test units pooled per fold',numpy_version=np.__version__,sklearn_version=sklearn.__version__,rows=rows,summary=summary)
    text=json.dumps(result,indent=2)+'\n'
    if args.out: args.out.write_text(text,encoding='utf8')
    else: print(text,end='')
if __name__=='__main__': main()
