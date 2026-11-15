"""Recompute saved-score AIT metrics; does not retrain or verify ground truth."""
from collections import defaultdict
from pathlib import Path
import csv, io, json, zipfile
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

root=Path(__file__).resolve().parent
arms=('111','011','current_only')
rows=defaultdict(list); budgets=defaultdict(lambda:np.zeros(4,dtype=np.int64))
with zipfile.ZipFile(root/'ait_validation_test_minimal.zip') as z:
    reference=json.loads(z.read('references.json'))
    costs={r['id']:(int(r['y']),int(r['alerts'])) for r in csv.DictReader(io.StringIO(z.read('session_costs.csv').decode('utf8')))}
    cells=sorted(n.removeprefix('cells/').removesuffix('.thresholds.json') for n in z.namelist() if n.endswith('.thresholds.json'))
    assert len(cells)==80 and len(costs)==11482 and sum(y for y,_ in costs.values())==403
    for prefix in cells:
        parts=prefix.split('_');tag,seed=parts[0],parts[3]
        cutoff=json.loads(z.read(f'cells/{prefix}.thresholds.json'))
        with np.load(io.BytesIO(z.read(f'cells/{prefix}.validation.npz')),allow_pickle=False) as val, np.load(io.BytesIO(z.read(f'cells/{prefix}.test.npz')),allow_pickle=False) as test:
            assert not set(val['ids']) & set(test['ids'])
            label_cost=np.asarray([costs[str(s)] for s in test['ids']])
            np.testing.assert_array_equal(test['y'],label_cost[:,0]);c=label_cost[:,1]
            scenes=np.asarray([str(s).split('_',1)[0] for s in test['ids']])
            for a in arms:
                negatives=np.sort(val[a][val['y']==0])[::-1]
                threshold=float(np.nextafter(negatives[len(negatives)//100],np.inf))
                assert threshold==cutoff[a],(prefix,a)
                for scene in np.unique(scenes):
                    m=scenes==scene;y=test['y'][m];score=test[a][m];called=score>=threshold
                    fpr,tpr,_=roc_curve(y,score,drop_intermediate=False)
                    result=[average_precision_score(y,score),float(tpr[fpr<=.01].max()),float(called[y==1].mean()),float(called[y==0].mean())]
                    rows[tag,a,str(scene),seed].append(result)
                if a in ('111','current_only'):
                    budget=int(c.sum())*2//100;order=np.lexsort((test['ids'],-test[a]));picked=[];spent=0
                    for i in order:
                        cost=int(c[i])
                        if spent+cost<=budget:picked.append(i);spent+=cost
                    budgets[tag,a]+=np.array([int(test['y'][np.array(picked,dtype=int)].sum()),len(picked),spent,budget])
result={}
for tag in ('611','422'):
    result[tag]={}
    for a in arms:
        scenes=sorted({k[2] for k in rows if k[:2]==(tag,a)});seeds=sorted({k[3] for k in rows if k[:2]==(tag,a)})
        assert len(scenes)==8 and len(seeds)==5
        for scene in scenes:
            for seed in seeds: assert len(rows[tag,a,scene,seed])==(1 if tag=='611' else 2)
        matrix=np.array([[np.mean(rows[tag,a,scene,seed],axis=0) for seed in seeds] for scene in scenes])
        for scene,vec in zip(scenes,matrix.mean(axis=1)): np.testing.assert_allclose(vec,reference[tag]['scenes'][a][scene],atol=1e-13,rtol=0)
        means=matrix.mean(axis=(0,1));np.testing.assert_allclose(means,reference[tag]['means'][a],atol=1e-13,rtol=0)
        result[tag][a]={'scene_macro_AP':float(means[0]),'recall_at_test_1pct_fpr':float(means[1]),'frozen_positive_recall':float(means[2]),'frozen_negative_review_rate':float(means[3])}
        if a in ('111','current_only'):
            np.testing.assert_array_equal(budgets[tag,a],reference[tag]['record_budget'][a])
            result[tag][a]['record_budget_sums']=budgets[tag,a].tolist()
out={'status':'PASS','cells':len(cells),'evaluation':'240 validation thresholds across 80 cells and scene-level AP, ranking recall, frozen-threshold rates, and 2% record-budget totals; no retraining or ground-truth verification','results':result}
(root/'verification_result.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
