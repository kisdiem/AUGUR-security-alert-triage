"""Regenerate selected paper tables from exported per-fold/seed metrics.

This is NOT model inference or training. No per-observation scores or labels
are included here. Output goes into an explicitly chosen directory.
"""
from collections import defaultdict
from pathlib import Path
from statistics import fmean
import argparse, csv, json, math

ROOT=Path(__file__).resolve().parent
SEEDS={42,1337,31415,2025,8675309}
SCENES={'fox','harrison','russellmitchell','santos','shaw','wardbeck','wheeler','wilson'}
METHODS=['current','state','LightGBM','NLLog','DeepCASE','UnseenLog','Velox']

def write(path, fieldnames, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def ait(out):
    groups=defaultdict(list)
    with (ROOT/'results/ait_per_scene.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            protocol,scene,seed,arm=row['protocol'],row['scene'].lower(),int(row['seed']),row['arm']
            if protocol not in ('611','422') or scene not in SCENES or seed not in SEEDS: raise ValueError('Invalid AIT split')
            groups[protocol,arm,scene,seed].append((float(row['AP']),float(row['Recall@1% FPR'])))
    arms=['000','001','010','011','100','101','110','111','current_only','independent_0.3','common_0.3','common_1']
    rows=[]
    saved=json.loads((ROOT/'results/ait_aggregate.json').read_text(encoding='utf-8'))
    for protocol in ['611','422']:
        for arm in arms:
            cells=[]
            for scene in sorted(SCENES):
                for seed in sorted(SEEDS):
                    values=groups[protocol,arm,scene,seed]
                    if len(values)!=(1 if protocol=='611' else 2): raise ValueError((protocol,arm,scene,seed,len(values)))
                    cells.append([fmean(x[j] for x in values) for j in (0,1)])
            means=[fmean(x[j] for x in cells) for j in (0,1)]
            ref=saved['results'][protocol]['means'][arm]
            if any(abs(means[j]-ref[j])>1e-10 for j in (0,1)): raise ValueError(('AIT summary mismatch',protocol,arm))
            rows.append({'protocol':protocol,'arm':arm,'AP_percent':f'{100*means[0]:.2f}','Recall1FPR_percent':f'{100*means[1]:.2f}','paper_table':'5' if len(arm)==3 and arm.isdigit() else ('2' if arm in ['current_only','111'] else '7' if arm in ['independent_0.3','common_0.3','common_1'] else '')})
    write(out/'ait_tables_2_5_7.csv',['protocol','arm','AP_percent','Recall1FPR_percent','paper_table'],rows)
    return len(rows)

def autolabel(out):
    data=json.loads((ROOT/'results/autolabel_aggregate.json').read_text(encoding='utf-8'))
    cells=defaultdict(dict)
    for row in data['rows']:
        k=(row['kind'],row['method']); sub=(row['fold'],int(row['seed']))
        if sub in cells[k]: raise ValueError(('duplicate AutoLabel row',k,sub))
        cells[k][sub]=[float(row['AP']),float(row['Recall1FPR'])]
    rows=[]
    for kind,fold_count in [('cve_group',5),('application',4)]:
        for method in METHODS:
            d=cells[kind,method]
            folds={k[0] for k in d}
            if len(folds)!=fold_count or any({s for f,s in d if f==fold}!={42,1337,31415} for fold in folds): raise ValueError(('incomplete AutoLabel cells',kind,method))
            means=[fmean(v[j] for v in d.values()) for j in (0,1)]
            ref=data['summary'][kind][method]['means']
            if any(abs(means[j]-ref[j])>1e-10 for j in (0,1)): raise ValueError(('AutoLabel summary mismatch',kind,method))
            rows.append({'holdout':kind,'method':method,'fold_seed_cells':len(d),'AP_percent':f'{100*means[0]:.2f}','Recall1FPR_percent':f'{100*means[1]:.2f}'})
    write(out/'autolabel_table_3.csv',['holdout','method','fold_seed_cells','AP_percent','Recall1FPR_percent'],rows)
    return len(rows)

def cert(out):
    data=json.loads((ROOT/'results/cert_aggregate.json').read_text(encoding='utf8'))
    rows=[]
    for method in ('et_current','et_session','et_state_12h','et_state_full'):
        cells=[data['score_file_audit'][f'{method}/{s}'] for s in sorted(SEEDS)]
        ap=fmean(c['AP'] for c in cells)
        recall=fmean(c['tp']/1830 for c in cells)
        rows.append({'method':method,'seeds':5,'AP_percent':f'{100*ap:.2f}','Recall1FPR_percent':f'{100*recall:.2f}'})
    write(out/'cert_table_4.csv',['method','seeds','AP_percent','Recall1FPR_percent'],rows)
    return len(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',required=True,help='New or existing output directory (outside results/ recommended)')
    args=ap.parse_args(); out=Path(args.output_dir).resolve()
    if out == (ROOT/'results').resolve(): raise SystemExit('Refusing to overwrite results/')
    out.mkdir(parents=True,exist_ok=True)
    counts={'AIT':ait(out),'AutoLabel':autolabel(out),'CERT':cert(out)}
    actual={}
    for dataset,filename,a,b in [('AIT','ait_tables_2_5_7.csv','protocol','arm'),('AutoLabel','autolabel_table_3.csv','holdout','method'),('CERT','cert_table_4.csv',None,'method')]:
        with (out/filename).open(encoding='utf8',newline='') as f:
            for row in csv.DictReader(f): actual[dataset,row[a] if a else 'temporal',row[b]]=(row['AP_percent'],row['Recall1FPR_percent'])
    with (ROOT/'expected_core_tables.csv').open(encoding='utf8',newline='') as f:
        expected=list(csv.DictReader(f))
    for row in expected:
        key=(row['dataset'],row['protocol_or_holdout'],row['arm_or_method'])
        if actual.get(key)!=(row['AP_percent'],row['Recall1FPR_percent']):
            raise ValueError(('paper table mismatch',key,actual.get(key),(row['AP_percent'],row['Recall1FPR_percent'])))
    print(json.dumps({'source':'exported fold/seed metrics, NOT predictions or training','rows_written':counts,'paper_rows_matched':len(expected),'output_dir':str(out)},indent=2))
if __name__=='__main__': main()
