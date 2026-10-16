"""Small model-serving API. Registry is populated only by verified real-data training."""
import json
import os
import tempfile
from pathlib import Path
from fastapi import FastAPI,File,HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from musicai.data import digest
from musicai.inference import analyze

app=FastAPI(title='MusicAI Research Lab')
app.add_middleware(CORSMiddleware,allow_origins=os.environ.get('MUSICAI_CORS_ORIGINS','http://localhost:3000').split(','),
                   allow_methods=['GET','POST'],allow_headers=['Content-Type'])
REGISTRY=Path(os.environ.get('MUSICAI_REGISTRY','checkpoints/registry.json'))

def registry():
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}

@app.get('/api/models')
def models():
    result=[]
    for name,item in registry().items():
        valid=Path(item['checkpoint']).is_file() and digest(item['checkpoint'])==item['sha256']
        result.append(dict(id=name,**{k:v for k,v in item.items() if k!='checkpoint'},available=valid))
    return result

@app.post('/api/analysis')
def analysis(model: str, audio: UploadFile=File(...)):
    entry=registry().get(model)
    if not entry: raise HTTPException(409,'No trained research checkpoint for this model. Import real data and train first.')
    if not Path(entry['checkpoint']).is_file() or digest(entry['checkpoint'])!=entry['sha256']:
        raise HTTPException(409,'Checkpoint missing or checksum mismatch')
    suffix=Path(audio.filename or '').suffix.lower()
    if suffix not in ('.wav','.flac','.mp3','.ogg'): raise HTTPException(415,'Use WAV, FLAC, MP3 or OGG supported by libsndfile')
    temp=None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as f:
            temp=Path(f.name); size=0
            while chunk:=audio.file.read(1024*1024):
                size+=len(chunk)
                if size>100*1024*1024: raise HTTPException(413,'Upload exceeds 100 MiB')
                f.write(chunk)
        return analyze(temp,entry['checkpoint'])
    except (ValueError,RuntimeError) as e: raise HTTPException(422,str(e)) from e
    finally:
        if temp: temp.unlink(missing_ok=True)
