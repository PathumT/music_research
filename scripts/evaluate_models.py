import _bootstrap
import argparse
from musicai.training import evaluate
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Evaluate saved models on held-out test; export actual metrics')
    p.add_argument('--config',required=True,help='Training YAML path')
    evaluate(p.parse_args().config)
