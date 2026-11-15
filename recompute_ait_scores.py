"""Verify exported AIT per-session scores against frozen per-scene metrics.

Extract data/ait_test_scores.zip and pass its test_scores directory using
--prediction-dir. Observe the attribution terms in the archive.
No individual labels or scores are printed or saved by this script.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

ARMS=('000','001','010','011','100','101','110','111','current_only')
SEEDS=(42,1337,31415,2025,8675309)
SCENES={'fox','harrison','russellmitchell','santos','shaw','wardbeck','wheeler','wilson'}

def metrics(y,s):
    if not np.all(np.isfinite(s)) or not np.all(np.isin(y,[0,1])):
        raise ValueError('non-finite scores or non-binary labels')
    npos=int(np.sum(y)); nneg=len(y)-npos
    if not npos or not nneg: raise ValueError('each scene requires positives and negatives')
    order=np.argsort(-s,kind='mergesort')
    scores=s[order]; label=y[order]
    ends=np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1]
    tp=np.cumsum(label,dtype=np.int64)[ends]
    fp=(ends+1)-tp
    recall=tp/npos
    precision=tp/(tp+fp)
    ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
    eligible=np.flatnonzero(fp<=math.floor(nneg*0.01))
    budget_recall=float(recall[eligible[-1]]) if len(eligible) else 0.
    return ap,budget_recall

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir',required=True)
    parser.add_argument('--reference-csv',default=str(Path(__file__).parent/'results/ait_per_scene.csv'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    by={}
    with Path(args.reference_csv).open(encoding='utf8',newline='') as f:
        for row in csv.DictReader(f):
            key=(row['protocol'],row['fold'],int(row['seed']),row['scene'].lower(),row['arm'])
            if key in by: raise ValueError('duplicate reference cell '+str(key))
            by[key]=(float(row['AP']),float(row['Recall@1% FPR']))
    rows=[]; maxerr=0.
    for protocol in ('611','422'):
        for fold in range(1,9):
            foldname=f'fold_{fold:02}'
            for seed in SEEDS:
                path=Path(args.prediction_dir)/f'{protocol}_{foldname}_{seed}.test.npz'
                with np.load(path,allow_pickle=False) as data:
                    ids=np.asarray(data['ids']).astype(str)
                    y=np.asarray(data['y'])
                    names=np.asarray([x.split('_',1)[0].lower() for x in ids])
                    unique=set(names)
                    if not unique.issubset(SCENES) or len(unique)!=(1 if protocol=='611' else 2): raise ValueError('unexpected scene allocation in '+path.name)
                    if len(set(ids)) != len(ids): raise ValueError('duplicate session IDs in '+path.name)
                    for scene in sorted(unique):
                        mask=names==scene
                        for arm in ARMS:
                            ap,rec=metrics(y[mask],np.asarray(data[arm])[mask])
                            key=protocol,foldname,seed,scene,arm
                            if key not in by: raise ValueError('missing reference cell '+str(key))
                            diff=max(abs(ap-by[key][0]),abs(rec-by[key][1]))
                            maxerr=max(maxerr,diff)
                            if diff>1e-10: raise ValueError('score/reference mismatch '+str(key)+' diff='+str(diff))
                            rows.append({'protocol':protocol,'fold':foldname,'seed':seed,'scene':scene,'arm':arm,'AP':ap,'Recall@1% FPR':rec})
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['protocol','fold','seed','scene','arm','AP','Recall@1% FPR']);w.writeheader();w.writerows(rows)
    print(json.dumps({'NPZ_files_checked':80,'per_scene_arm_rows_checked':len(rows),'maximum_absolute_metric_error':maxerr,'output':str(out)},indent=2))
if __name__=='__main__': main()
