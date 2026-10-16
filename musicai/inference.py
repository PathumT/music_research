import math
import soundfile as sf
import torch
from .adapters import CLASSES
from .data import features,waveform
from .training import load_checkpoint

def analyze(path,checkpoint):
    model,ck=load_checkpoint(checkpoint)
    if ck['synthetic']: raise ValueError('Synthetic smoke checkpoint cannot analyze research songs')
    p=ck['preprocessing']; seconds=p['segment_seconds']; info=sf.info(path)
    duration=info.frames/info.samplerate
    if duration<=0: raise ValueError('Empty audio')
    timeline=[]; total=torch.zeros(10); weight=0
    with torch.no_grad():
        for i in range(math.ceil(duration/seconds)):
            start=i*seconds; end=min(duration,start+seconds)
            x=features(waveform(path,start,end,p),p).unsqueeze(0)
            probs=model(x).sigmoid()[0]; total+=probs*(end-start); weight+=end-start
            timeline.append(dict(start=start,end=end,probabilities=dict(zip(CLASSES,probs.tolist()))))
    probabilities=(total/weight).tolist()
    supported=[name for name in CLASSES if ck['train_positive_support'].get(name,0)>0]
    detected=[k for k,v,t in zip(CLASSES,probabilities,ck['thresholds']) if k in supported and v>=t]
    return dict(probabilities=dict(zip(CLASSES,probabilities)),
        instruments=detected,unsupported_classes=[name for name in CLASSES if name not in supported],
        unknown=not detected,
        timeline=timeline,activity_supervision='weak clip estimates, not frame-ground-truth detection',
        aggregation='duration-weighted mean window probabilities',duration=duration)
