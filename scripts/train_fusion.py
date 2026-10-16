import _bootstrap
import argparse
from musicai.training import train
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Fit confidence-guided CNN/attention logit fusion; requires both expert checkpoints')
    p.add_argument('--config',required=True,help='Training YAML path')
    train(p.parse_args().config,'fusion')
