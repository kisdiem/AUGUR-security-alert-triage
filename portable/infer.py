"""Recompute the original 3,520-session frozen 4/2/2 numerical inference cell."""
from pathlib import Path
import argparse,json,sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from portable.model import branch_inputs,load_frozen,score,digest

def run(bundle,output,model_override=None):
    root=Path(bundle);output=Path(output)
    models=load_frozen(root)
    if model_override is not None:
        import joblib
        # The optional override is the output of the trusted local training entry.
        models=joblib.load(model_override)
    with np.load(root/'numeric_inputs.npz',allow_pickle=False) as d:
        result=score(models,branch_inputs(d),d['local73'],d['gates3'],d['roles54'],d['links24'])
        expected=d['expected111'].copy()
    error=float(np.max(np.abs(result['augur']-expected)))
    np.testing.assert_allclose(result['augur'],expected,rtol=0,atol=1e-12)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'scores.npz',**result)
    report={'status':'PASS','sessions':len(expected),'exact_array_equal':bool(np.array_equal(result['augur'],expected)),
            'max_abs_score_error':error,'absolute_tolerance':1e-12,
            'scope':'One 422 fold/seed, numeric features to three heads, protocol experts and fusion. No raw-log or label reconstruction.',
            'model_source':'retrained locally' if model_override else 'frozen reference',
            'input_sha256':digest(root/'numeric_inputs.npz')}
    (output/'inference_receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--model',type=Path);a=p.parse_args();print(json.dumps(run(a.bundle,a.output_dir,a.model),indent=2))
if __name__=='__main__':main()
