import copy
import json
import platform
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from .adapters import CLASSES
from .data import Segments, digest, dump, validate_manifest
from .models import create, Fusion

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)

def load_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=True)
    if ck['classes']!=CLASSES: raise ValueError('Checkpoint taxonomy mismatch')
    model=create(ck['kind']).to(device); model.load_state_dict(ck['state_dict']); model.eval()
    return model,ck

def masked_loss(logits,y,mask):
    return (torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*mask).sum()/mask.sum().clamp(min=1)

def metrics(y,p,mask,thresholds):
    pred=p>=np.asarray(thresholds)
    result={'per_instrument':{}}; total=np.zeros(3); fs=[]
    for i,name in enumerate(CLASSES):
        valid=mask[:,i]>0; target=y[valid,i]; estimate=pred[valid,i]
        if not valid.any():
            result['per_instrument'][name]={'support':0,'known':0,'precision':None,'recall':None,'f1':None,'average_precision':None}; continue
        precision,recall,f1,_=precision_recall_fscore_support(target,estimate,average='binary',zero_division=0)
        tp=int(((target==1)&estimate).sum()); fp=int(((target==0)&estimate).sum()); fn=int(((target==1)&~estimate).sum())
        total+=np.array([tp,fp,fn]); fs.append(float(f1))
        result['per_instrument'][name]=dict(precision=float(precision),recall=float(recall),f1=float(f1),
            average_precision=float(average_precision_score(target,p[valid,i])) if target.sum() else None,
            support=int(target.sum()),known=int(valid.sum()),tp=tp,fp=fp,fn=fn,tn=int(((target==0)&~estimate).sum()))
    tp,fp,fn=total
    result.update(macro_f1=float(np.mean(fs)) if fs else None,micro_f1=float(2*tp/max(1,2*tp+fp+fn)),
                  annotation_policy='Unknown class targets excluded; metrics on observed labels only')
    return result

def collect(model,loader,device,max_batches=None):
    model.eval(); ys=[]; ps=[]; masks=[]; losses=[]
    with torch.no_grad():
        for i,(x,y,m) in enumerate(loader):
            if max_batches and i>=max_batches: break
            z=model(x.to(device)); losses.append((float(masked_loss(z,y.to(device),m.to(device))),float(m.sum())))
            ys.append(y.numpy()); ps.append(z.sigmoid().cpu().numpy()); masks.append(m.numpy())
    if not ys: raise ValueError('Empty evaluation partition')
    loss=sum(v*n for v,n in losses)/max(1,sum(n for _,n in losses))
    return np.concatenate(ys),np.concatenate(ps),np.concatenate(masks),loss

def thresholds_from_validation(y,p,m):
    thresholds=[]
    for i in range(10):
        valid=m[:,i]>0
        if not valid.any() or y[valid,i].sum()==0: thresholds.append(.5); continue
        scores=[]
        for t in np.arange(.1,.91,.05):
            f=precision_recall_fscore_support(y[valid,i],p[valid,i]>=t,average='binary',zero_division=0)[2]
            scores.append((f,-abs(t-.5),float(t)))
        thresholds.append(max(scores)[2])
    return thresholds

def loaders(c):
    rows=json.loads(Path(c['manifest']).read_text()); validate_manifest(rows)
    if any(r.get('synthetic',False) for r in rows) and not c.get('allow_synthetic',False): raise ValueError('Synthetic fixtures are not research data')
    checked=set()
    import soundfile as sf
    for r in rows:
        if r['audio'] not in checked:
            if digest(r['audio'])!=r['audio_sha256']: raise ValueError('Raw audio changed since preparation: '+r['audio'])
            checked.add(r['audio'])
        info=sf.info(r['audio'])
        if r['end']>info.frames/info.samplerate+1e-6: raise ValueError('Segment exceeds source duration')
    result={}
    for split in ('train','validation','test'):
        subset=[r for r in rows if r['split']==split]
        if not subset: raise ValueError(f'Missing {split} partition')
        result[split]=torch.utils.data.DataLoader(Segments(subset,c['preprocessing'],split=='train'),
            batch_size=c['batch_size'],shuffle=split=='train',num_workers=0)
    return rows,result

def register(path,checkpoint):
    model,ck=load_checkpoint(checkpoint)
    p=ck['preprocessing']; frames=1+int(p['sample_rate']*p['segment_seconds'])//p['hop_length']
    with torch.no_grad():
        output=model(torch.zeros(1,1,p['n_mels'],frames))
    if not torch.isfinite(output).all() or output.shape!=(1,10): raise ValueError('Checkpoint inference validation failed')
    if ck['synthetic']: return # Fixture checkpoints never enter production registry.
    registry=json.loads(Path(path).read_text()) if Path(path).exists() else {}
    registry[ck['kind']]={'checkpoint':str(Path(checkpoint).resolve()),'sha256':digest(checkpoint),'ready':True,
        'classes':ck['classes'],'dataset_sha256':ck['manifest_sha256'],'kind':ck['kind']}
    dump(path,registry)

def train(config,kind):
    c=yaml.safe_load(Path(config).read_text()); seed_all(c['seed']); device=c.get('device','cpu')
    rows,dl=loaders(c); run=Path(c['output'])/kind; run.mkdir(parents=True,exist_ok=True)
    model=create(kind).to(device)
    if kind=='fusion':
        a,ac=load_checkpoint(Path(c['output'])/'baseline/best.pt',device)
        b,bc=load_checkpoint(Path(c['output'])/'attention/best.pt',device)
        for ck in (ac,bc):
            if ck['manifest_sha256']!=digest(c['manifest']) or ck['preprocessing']!=c['preprocessing']: raise ValueError('Expert dataset/preprocessing mismatch')
        model=Fusion(a,b).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['learning_rate'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2)
    initial={k:v.detach().clone() for k,v in model.named_parameters() if v.requires_grad}
    history=[]; best=float('inf'); stale=0; started=time.time(); updates=0
    for epoch in range(c['epochs']):
        model.train(); losses=[]
        for batch,(x,y,m) in enumerate(dl['train']):
            if c.get('max_batches') and batch>=c['max_batches']: break
            optimizer.zero_grad(); z=model(x.to(device)); loss=masked_loss(z,y.to(device),m.to(device))
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            updates+=1; losses.append(float(loss.detach()))
        y,p,m,val=collect(model,dl['validation'],device,c.get('max_batches'))
        measured=metrics(y,p,m,[.5]*10)
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_loss=val,
                            macro_f1=measured['macro_f1'],micro_f1=measured['micro_f1']))
        scheduler.step(val)
        if val<best:
            best=val; stale=0
            ck=dict(kind=kind,state_dict=copy.deepcopy(model.cpu().state_dict()),classes=CLASSES,
                preprocessing=c['preprocessing'],thresholds=[.5]*10,config=c,manifest_sha256=digest(c['manifest']),
                synthetic=any(r.get('synthetic',False) for r in rows),epoch=epoch+1,
                software={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__},
                hardware={'device':device,'machine':platform.machine(),'processor':platform.processor(),
                    'cuda_device':torch.cuda.get_device_name() if device.startswith('cuda') and torch.cuda.is_available() else None},
                train_positive_support={name:sum(name in r['labels'] for r in rows if r['split']=='train') for name in CLASSES},
                parameters=sum(p.numel() for p in model.parameters()),activity_supervision='weak clip labels',
                fusion_inputs='CNN and attention, original audio only' if kind=='fusion' else None)
            torch.save(ck,run/'best.pt'); model.to(device)
        else: stale+=1
        dump(run/'history.json',history)
        print(f'{kind} epoch {epoch+1}: train={history[-1]["train_loss"]:.4f} validation={val:.4f}',flush=True)
        if stale>=c.get('patience',5): break
    changed=any(not torch.equal(initial[k].cpu(),v.detach().cpu()) for k,v in model.named_parameters() if k in initial)
    if not updates or not changed: raise RuntimeError('No optimizer update detected')
    model,ck=load_checkpoint(run/'best.pt',device)
    y,p,m,_=collect(model,dl['validation'],device) # Thresholds never see held-out test.
    ck['thresholds']=thresholds_from_validation(y,p,m); ck['training_seconds']=time.time()-started
    ck['optimizer_updates']=updates; torch.save(ck,run/'best.pt')
    dump(run/'metadata.json',{k:v for k,v in ck.items() if k!='state_dict'})
    cache=Path(c['output'])/'.matplotlib'; cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache.resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot([h['train_loss'] for h in history],label='train'); plt.plot([h['validation_loss'] for h in history],label='validation')
    plt.xlabel('Epoch (zero indexed)'); plt.ylabel('Masked BCE'); plt.legend(); plt.tight_layout(); plt.savefig(run/'loss.png'); plt.close()
    register(c['registry'],run/'best.pt')
    return run/'best.pt'

def evaluate(config):
    c=yaml.safe_load(Path(config).read_text()); seed_all(c['seed']); rows,dl=loaders(c); summary={}
    for kind in ('baseline','attention','fusion'):
        path=Path(c['output'])/kind/'best.pt'
        if not path.exists(): summary[kind]={'status':'checkpoint_unavailable'}; continue
        model,ck=load_checkpoint(path,c['device'])
        if ck['manifest_sha256']!=digest(c['manifest']): raise ValueError('Evaluation manifest differs from training')
        y,p,m,loss=collect(model,dl['test'],c['device'])
        result=metrics(y,p,m,ck['thresholds']); result.update(loss=loss,checkpoint_sha256=digest(path),
            synthetic=ck['synthetic'],samples=len(y),thresholds=ck['thresholds'])
        test=[r for r in rows if r['split']=='test']; errors=[]
        for r,target,probs,mask in zip(test,y,p,m):
            wrong=[name for i,name in enumerate(CLASSES) if mask[i] and bool(target[i])!=bool(probs[i]>=ck['thresholds'][i])]
            if wrong: errors.append({'segment_id':r['id'],'misclassified':wrong,'probabilities':probs.tolist()})
        result['errors']=errors; dump(path.parent/'evaluation.json',result)
        import csv
        with (path.parent/'per_instrument.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['instrument','precision','recall','f1','support']); w.writeheader()
            for name,v in result['per_instrument'].items(): w.writerow({'instrument':name,**{k:v[k] for k in w.fieldnames if k!='instrument'}})
        summary[kind]=result
    dump(Path(c['output'])/'comparison.json',summary)
    return summary
