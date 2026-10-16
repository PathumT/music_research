"""CPU-only integration verification with explicitly synthetic, non-research fixtures."""
import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import yaml
from musicai.adapters import CLASSES,import_records
from musicai.data import digest,dump,validate_manifest,Segments
from musicai.training import train,evaluate,load_checkpoint,masked_loss

def smoke(output):
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=True)
    rows=[]; sr=8000
    for split in ('train','validation','test'):
        for i,name in enumerate(CLASSES):
            path=root/f'{split}_{i}.wav'
            t=np.arange(sr*3)/sr
            # These tones exercise plumbing. Instrument names are fixture targets, not acoustic truth.
            phase={'train':0,'validation':.31,'test':.67}[split]
            sf.write(path,.2*np.sin(2*np.pi*(110+37*i)*t+phase)+.01*np.sin(2*np.pi*13*t),sr)
            rows.append(dict(id=f'{split}_{i}',audio=str(path),audio_sha256=digest(path),start=0,end=3,
                group_id=f'fixture:{split}:{i}',split=split,labels=[name],known_classes=CLASSES,synthetic=True))
    validate_manifest(rows); dump(root/'segments.json',rows)
    for split in ('train','validation','test'): dump(root/f'{split}.json',[r for r in rows if r['split']==split])
    c=dict(manifest=str(root/'segments.json'),output=str(root/'results'),registry=str(root/'registry.json'),
        seed=42,device='cpu',epochs=2,batch_size=5,learning_rate=.003,patience=3,max_batches=2,allow_synthetic=True,
        preprocessing=dict(sample_rate=8000,segment_seconds=3,n_fft=256,hop_length=128,n_mels=32))
    config=root/'training.yaml'; config.write_text(yaml.safe_dump(c))
    for kind in ('baseline','attention','fusion'): train(config,kind)
    results=evaluate(config)
    x,y,m=Segments(rows,c['preprocessing'])[0]
    assert x.shape[0:2]==(1,32) and torch.isfinite(x).all()
    for kind in results:
        model,ck=load_checkpoint(root/'results'/kind/'best.pt')
        with torch.no_grad(): pred=model(x.unsqueeze(0))
        assert pred.shape==(1,10) and torch.isfinite(pred).all()
        assert ck['optimizer_updates']==4
    assert not (root/'registry.json').exists(),'Synthetic checkpoints must not be production ready'
    # Check unknown targets contribute zero gradient.
    z=torch.zeros(1,10,requires_grad=True); mask=torch.zeros(1,10); mask[0,0]=1
    masked_loss(z,torch.ones_like(z),mask).backward()
    assert z.grad[0,1:].abs().sum()==0
    leaky=[dict(rows[0]),dict(rows[1],group_id=rows[0]['group_id'],split='test')]
    try: validate_manifest(leaky)
    except ValueError: pass
    else: raise AssertionError('Leakage not detected')
    from fastapi.testclient import TestClient
    from backend.main import app
    client=TestClient(app)
    assert client.get('/api/models').status_code==200
    with Path(rows[0]['audio']).open('rb') as f:
        assert client.post('/api/analysis?model=unavailable',files={'audio':('test.wav',f,'audio/wav')}).status_code==409
    report=dict(status='passed',synthetic=True,warning='Not instrument-recognition research evidence',
        segments=len(rows),splits={s:10 for s in ('train','validation','test')},
        checks=['loader','split leakage rejection','masked BCE gradients','optimizer updates',
                'baseline/attention/fusion training','checkpoint save/load','inference','held-out evaluation',
                'synthetic registry exclusion','API unavailable-model response'],
        results={k:{s:v[s] for s in ('loss','macro_f1','micro_f1','samples')} for k,v in results.items()})
    dump(root/'verification.json',report)
    dump('research/results/smoke_verification.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/smoke',help='Synthetic fixture/checkpoint output directory')
    smoke(p.parse_args().output)
