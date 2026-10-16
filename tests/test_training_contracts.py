import copy
import numpy as np
import pytest
import torch
from musicai.adapters import CLASSES
from musicai.data import features,waveform,validate_manifest
from musicai.models import Recognizer,Fusion
from musicai.training import masked_loss,metrics,thresholds_from_validation,seed_all

def test_unknown_target_mask_and_optimizer():
    seed_all(42)
    model=Recognizer(); optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    before=model.head.weight.detach().clone()
    x=torch.randn(2,1,32,64); y=torch.zeros(2,10); y[:,0]=1; mask=torch.ones_like(y)
    loss=masked_loss(model(x),y,mask); loss.backward(); optimizer.step()
    assert not torch.equal(before,model.head.weight)
    z=torch.zeros(2,10,requires_grad=True); mask[:,1:]=0
    masked_loss(z,y,mask).backward(); assert z.grad[:,1:].abs().sum()==0

def test_metric_masks_and_confusion():
    y=np.zeros((2,10)); y[0,0]=1
    m=np.zeros_like(y); m[:,0]=1
    p=np.ones_like(y)*.8
    result=metrics(y,p,m,[.5]*10)
    v=result['per_instrument']['piano']
    assert v['tp']==1 and v['fp']==1 and v['fn']==0
    assert result['per_instrument']['violin']['f1'] is None
    assert len(thresholds_from_validation(y,p,m))==10

def test_audio_feature_edge_cases(tmp_path):
    import soundfile as sf
    p=dict(sample_rate=8000,segment_seconds=3,n_fft=256,hop_length=128,n_mels=32)
    path=tmp_path/'stereo.wav'; sf.write(path,np.zeros((16000,2)),16000)
    x=waveform(path,0,1,p)
    assert x.shape==(24000,) and torch.isfinite(features(x,p)).all()
    with pytest.raises(ValueError): waveform(path,1,1,p)
    bad=tmp_path/'bad.wav'; bad.write_text('not audio')
    with pytest.raises(RuntimeError): waveform(bad,0,1,p)

def test_fusion_experts_frozen():
    seed_all(42); model=Fusion(); model.train()
    assert not model.baseline.training and not model.attention.training
    assert all(not p.requires_grad for p in model.baseline.parameters())
    z=model(torch.randn(2,1,32,64)); z.sum().backward()
    assert model.gate.grad is not None and torch.isfinite(model.gate.grad).all()

def test_duplicate_audio_leakage():
    a=dict(id='a',group_id='one',audio_sha256='same',start=0,end=3,split='train',labels=['piano'],known_classes=CLASSES)
    b=dict(a,id='b',group_id='two',split='test')
    with pytest.raises(ValueError,match='leakage'): validate_manifest([a,b])
