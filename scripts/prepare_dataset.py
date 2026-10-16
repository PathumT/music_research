import _bootstrap
import argparse
from musicai.data import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Import licensed local datasets and write grouped segment manifests.')
    p.add_argument('--config',required=True,help='Dataset source YAML path (paths relative to working directory)')
    prepare(p.parse_args().config)
