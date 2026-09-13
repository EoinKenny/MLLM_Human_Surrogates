"""Record explicit local consent for inference against current code and inputs."""
import argparse,json
from pathlib import Path
from audit_gate import snapshot,VERSIONS
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--approve',action='store_true',required=True)
    p.parse_args()
    target=Path(__file__).resolve().parent/'COLLECTION_APPROVAL.json'
    target.write_text(json.dumps({'approved':True,'files':snapshot(),'versions':VERSIONS},indent=2)+'\n')
    print('Collection authorized for this local code/input snapshot.')
