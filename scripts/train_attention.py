import _bootstrap
import argparse
from musicai.training import train
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Train CNN, bidirectional GRU and temporal attention classifier')
    p.add_argument('--config',required=True,help='Training YAML path')
    train(p.parse_args().config,'attention')
