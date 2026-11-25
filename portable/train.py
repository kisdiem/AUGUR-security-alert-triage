"""Retrain the selected 422 cell from original prepared numerical training data.

All parameters and edits are frozen inputs. Validation determines cutoffs before
the optional test inference step. No test score selects a hyperparameter.
"""
from pathlib import Path
import argparse,json,sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from portable.model import digest,read_json,score,validation_cutoff,load_frozen

def run(bundle,output,verify_test=False):
    import sklearn,joblib
    from sklearn.ensemble import ExtraTreesClassifier
    root=Path(bundle);output=Path(output)
    reference=load_frozen(root)
    receipt=read_json(root/'training_manifest.json')
    for name,expected in receipt['files'].items():
        if digest(root/name)!=expected:raise ValueError('Training input hash: '+name)
    params=read_json(root/'training_params.json')
    with np.load(root/'train.npz',allow_pickle=False) as train, np.load(root/'validation.npz',allow_pickle=False) as val:
        if set(train['ids']) & set(val['ids']):raise ValueError('Train/validation overlap')
        y=train['y'];w=train['sample_weight'];w=None if w.size==0 else w
        matrices={'host':train['host'],'cumulative':np.c_[train['host'],train['cumulative_extra']],
                  'direct':np.c_[train['host'],train['direct_extra']]}
        models={}
        for name in ['host','cumulative','direct']:
            if params[name]!=reference[name].get_params(deep=False):raise ValueError('Parameters differ from frozen reference')
            models[name]=ExtraTreesClassifier(**params[name]).fit(matrices[name],y,sample_weight=w)
        use=train['expert_use'].astype(bool);targets=train['expert_union'].astype(bool)
        models['experts']=[]
        for i,param in enumerate(params['experts']):
            if param!=reference['experts'][i].get_params(deep=False):raise ValueError('Expert parameters differ')
            models['experts'].append(ExtraTreesClassifier(**param).fit(train['expert_x'][use],targets[use]))
        val_matrices={name:val[name] for name in ['host','cumulative','direct']}
        predictions=score(models,val_matrices,val['local73'],val['gates3'],val['roles54'],val['links24'])
        cutoffs={k:validation_cutoff(val['y'],p) for k,p in predictions.items() if k in ['correction_off','augur']}
        ntrain,nval=len(y),len(val['y'])
    output.mkdir(parents=True,exist_ok=True)
    model_file=output/'model.joblib';joblib.dump(models,model_file,compress=3)
    sealed={'protocol':'422','fold':'fold_01','seed':42,'cutoffs':cutoffs,'model_sha256':digest(model_file),
            'training_files_sha256':receipt['files'],'test_features_used_for_fitting_or_cutoff':False}
    (output/'validation_seal.json').write_text(json.dumps(sealed,indent=2)+'\n',encoding='utf-8')
    result={'status':'PASS','training_rows':ntrain,'validation_rows':nval,'main_trees':sum(models[n].n_estimators for n in ['host','cumulative','direct']),
            'expert_trees':sum(m.n_estimators for m in models['experts']),'scikit_learn':sklearn.__version__,
            'scope':'One original prepared numerical cell, all three main heads and three protocol experts. Raw preprocessing and all 80-cell retraining are not reproduced.'}
    if verify_test:
        from portable.infer import run as infer
        result['test_inference']=infer(root,output/'test',model_file)
    (output/'training_receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--verify-test',action='store_true');a=p.parse_args();print(json.dumps(run(a.bundle,a.output_dir,a.verify_test),indent=2))
if __name__=='__main__':main()
