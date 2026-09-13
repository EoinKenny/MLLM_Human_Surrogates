"""Collection requires explicit approval tied to the exact reviewed code/input bundle."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
VERSIONS={'adult':'adult-v2-paired-demonstrations-target-saliency','bios':'bios-v2-target-saliency',
'recourse':'recourse-v2-codebook-model-verified','teaching':'teaching-v14-simple-random300-local-causal',
'engagement':'engagement-v2-supplied-scenario','model_improvement':'model-improvement-v2-strict-config'}

def snapshot():
    paths=sorted(p for p in ROOT.rglob('*.py') if not any(x in p.parts for x in ['vendor','__pycache__']))
    paths+=sorted((ROOT/'teaching/data_v12').glob('*'))
    paths+=[ROOT/'teaching/original_script_reference.txt',ROOT/'teaching/boundary_originals.json']
    paths+=sorted((ROOT/'appropriate_reliance/data_v2').glob('*.csv'))
    paths+=sorted((ROOT/'recourse/datasets').glob('*'))
    paths+=sorted((ROOT/'recourse/data_v2').glob('*'))
    paths+=[ROOT/'model_improv/pima-indians-diabetes.csv',ROOT/'model_improv/feature_distributions.png']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def check_approval():
    path=ROOT/'COLLECTION_APPROVAL.json'
    if not path.exists():raise RuntimeError('Collection remains stopped. Researcher approval of the rendered prompt audit is required.')
    approval=json.loads(path.read_text())
    if approval.get('approved') is not True or approval.get('files')!=snapshot() or approval.get('versions')!=VERSIONS:
        raise RuntimeError('Collection approval does not match the current code and inputs; rerun preflight and review.')

# Direct experiment runners use snapshot approval rather than historical server scopes.
def reserve_pilot_requests(*args, **kwargs):
    return None
