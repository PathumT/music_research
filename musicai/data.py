import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pretty_midi
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly
from .adapters import CLASSES, import_records

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def validate_manifest(rows):
    groups, contents, ids = {}, {}, set()
    for r in rows:
        if r['split'] not in ('train','validation','test'): raise ValueError('Invalid split')
        if not 0 <= r['start'] < r['end']: raise ValueError('Invalid segment offsets')
        if not set(r['labels']) <= set(r['known_classes']) <= set(CLASSES): raise ValueError('Invalid known-target mask')
        if r['id'] in ids: raise ValueError('Duplicate segment')
        ids.add(r['id'])
        for mapping,key in ((groups,r['group_id']),(contents,r['audio_sha256'])):
            if key in mapping and mapping[key] != r['split']: raise ValueError('Split leakage: '+key)
            mapping[key]=r['split']
        if not r['labels'] or not set(r['labels']) <= set(CLASSES): raise ValueError('Invalid labels')
    return True

def prepare(config):
    c=yaml.safe_load(Path(config).read_text()); out=Path(c['output']); out.mkdir(parents=True,exist_ok=True)
    seconds=float(c.get('segment_seconds',5))
    if not 3 <= seconds <= 10: raise ValueError('segment_seconds must be 3–10')
    report={'sources':{},'errors':[],'unsupported_labels':{}}; tracks=[]
    for source,opt in c['sources'].items():
        if not opt.get('enabled',True): continue
        if not Path(opt['root']).exists():
            report['sources'][source]={'status':'unavailable','root':opt['root']}; continue
        if not opt.get('license_accepted',False):
            report['sources'][source]={'status':'license_acceptance_required'}; continue
        try:
            imported=import_records(source,opt['root'],opt)
            for r in imported:
                r['source_version']=str(opt.get('version','unspecified'))
                for label in r['unsupported_labels']:
                    key=source+':'+label
                    report['unsupported_labels'][key]=report['unsupported_labels'].get(key,0)+1
            tracks.extend(imported)
            report['sources'][source]={'status':'imported','tracks':len(imported)}
        except Exception as e: report['errors'].append({'source':source,'error':str(e)})
    rows=[]; seen=set(); group_splits={}; content_splits={}
    for r in tracks:
        try:
            info=sf.info(r['audio'])
            sha=digest(r['audio'])
            if sha in seen: continue
            seen.add(sha)
            if r['midi'] and not Path(r['midi']).is_file(): raise ValueError('Missing aligned MIDI')
            midi=pretty_midi.PrettyMIDI(r['midi']) if r['midi'] else None
            aligned=[(pretty_midi.PrettyMIDI(item['path']),item['label']) for item in r.get('midi_sources',[])]
            value=int(hashlib.sha256(f"{c['seed']}:{r['group_id']}".encode()).hexdigest()[:8],16)/2**32
            split=r['official_split'] or ('train' if value<.7 else 'validation' if value<.85 else 'test')
            if r['group_id'] in group_splits and group_splits[r['group_id']]!=split: raise ValueError('Official split conflicts within group')
            group_splits[r['group_id']]=split; content_splits[sha]=split
            duration=info.frames/info.samplerate
            for start in np.arange(0,duration,seconds):
                end=min(float(start)+seconds,duration)
                if end-start < .25: continue
                # IRMAS modification restriction: retain original excerpts intact.
                if r['source']=='irmas' and (start>0 or end<duration): continue
                audio,_=sf.read(r['audio'],start=int(start*info.samplerate),stop=int(end*info.samplerate),always_2d=True)
                if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2))<1e-5: continue
                notes=[]
                window_labels=set()
                for annotation,label in ([(midi,r['labels'][0] if r['labels'] else None)] if midi else aligned):
                    for inst in annotation.instruments:
                        for n in inst.notes:
                            if n.start<end and n.end>start:
                                notes.append(dict(start=float(max(0,n.start-start)),end=float(min(end-start,n.end-start)),
                                    pitch=int(n.pitch),velocity=int(n.velocity),program=int(inst.program),is_drum=bool(inst.is_drum),label=label))
                                if label: window_labels.add(label)
                if (midi or aligned) and not window_labels: continue
                if not r['labels']: continue
                row=dict(r,start=float(start),end=end,split=split,audio_sha256=sha,notes=notes,
                         synthetic=False, sample_rate=info.samplerate,channels=info.channels)
                if midi or aligned: row['labels']=sorted(window_labels)
                row['id']=hashlib.sha256(f'{sha}:{start}:{end}'.encode()).hexdigest()
                rows.append(row)
        except Exception as e: report['errors'].append({'audio':r['audio'],'error':str(e)})
    # Deterministic hash sampling, independent of adapter enumeration order.
    rows.sort(key=lambda r:hashlib.sha256(f"{c['seed']}:{r['id']}".encode()).hexdigest())
    rows=rows[:int(c.get('target_segments',10000))]
    validate_manifest(rows)
    dump(out/'segments.json',rows)
    for split in ('train','validation','test'):
        subset=[r for r in rows if r['split']==split]; dump(out/f'{split}.json',subset)
        with (out/f'{split}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['id','audio','group_id','start','end','labels','midi'])
            writer.writeheader()
            writer.writerows({k:json.dumps(r[k]) if isinstance(r[k],list) else r[k] for k in writer.fieldnames} for r in subset)
    report.update(segments=len(rows),target=c.get('target_segments',10000),
        target_shortfall=max(0,int(c.get('target_segments',10000))-len(rows)),
        splits={s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},
        labels={k:sum(k in r['labels'] for r in rows) for k in CLASSES},
        manifest_sha256=digest(out/'segments.json'),config=c)
    dump(out/'report.json',report)
    print(json.dumps(report,indent=2))
    if not rows: raise RuntimeError('No research audio prepared. See report.json and docs/dataset_setup.md')
    return rows

def waveform(path,start,end,p):
    info=sf.info(path)
    x,sr=sf.read(path,start=int(start*info.samplerate),stop=int(end*info.samplerate),always_2d=True,dtype='float32')
    if not len(x) or not np.isfinite(x).all(): raise ValueError('Empty or invalid audio')
    x=x.mean(1); target=p['sample_rate']; divisor=math.gcd(sr,target)
    if sr!=target: x=resample_poly(x,target//divisor,sr//divisor)
    n=int(target*p['segment_seconds']); x=x[:n]
    x=np.pad(x,(0,max(0,n-len(x))))
    return torch.from_numpy(x.copy())

def features(x,p):
    # Triangular HTK Mel filterbank, shared by training and inference.
    sr,nfft,hop,mels=p['sample_rate'],p['n_fft'],p['hop_length'],p['n_mels']
    spec=torch.stft(x,nfft,hop,window=torch.hann_window(nfft),return_complex=True).abs().square()
    hz=lambda m:700*(10**(m/2595)-1)
    edges=hz(torch.linspace(0,2595*math.log10(1+sr/1400),mels+2))
    freq=torch.linspace(0,sr/2,nfft//2+1)
    bank=torch.minimum((freq[None]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),
                       (edges[2:,None]-freq[None])/(edges[2:,None]-edges[1:-1,None])).clamp(min=0)
    mel=torch.log1p(bank@spec)
    return ((mel-mel.mean())/mel.std().clamp(min=1e-6)).unsqueeze(0)

class Segments(torch.utils.data.Dataset):
    def __init__(self,rows,p,augment=False): self.rows,self.p,self.augment=rows,p,augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; x=waveform(r['audio'],r['start'],r['end'],self.p)
        if self.augment and r.get('source')!='irmas':
            x=x*(.8+.4*torch.rand(()))
            x=x+torch.randn_like(x)*.001
        y=torch.tensor([float(k in r['labels']) for k in CLASSES])
        mask=torch.tensor([float(k in r['known_classes']) for k in CLASSES])
        return features(x,self.p),y,mask
