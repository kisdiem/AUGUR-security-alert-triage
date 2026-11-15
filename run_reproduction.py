"""One-command synthetic contract checks and exported-metric table aggregation."""
import argparse, os, subprocess, sys
from pathlib import Path

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',required=True)
args=parser.parse_args()
env=os.environ.copy()
env['PYTHONDONTWRITEBYTECODE']='1'
for script,args_ in [('state_124/state_contract_checks.py',[]),('test_current.py',[]),('reproduce_core_tables.py',['--output-dir',args.output_dir])]:
    print('\nRUN:',script,flush=True)
    subprocess.run([sys.executable,str(root/script),*args_],cwd=root,env=env,check=True)
print('\nPASS: synthetic implementation contracts and aggregation of exported metrics.')
print('NOT checked: raw data preparation, model training, per-sample predictions, end-to-end performance.')
