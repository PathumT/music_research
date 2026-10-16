"""Download provider-hosted archives without extracting or accepting licenses implicitly."""
import _bootstrap
import argparse
import hashlib
import urllib.request
from pathlib import Path

URLS={
 'nsynth-train':('https://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz',None),
 'nsynth-valid':('https://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz',None),
 'nsynth-test':('https://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz',None),
 'maestro-v3':('https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip',
              '6680fea5be2339ea15091a249fbd70e49551246ddbd5ca50f1b2352c08c95291')}

def download(name,output,accepted):
    if not accepted: raise ValueError('Review official license first; pass --accept-license explicitly')
    url,expected=URLS[name]; dest=Path(output)/url.rsplit('/',1)[1]
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise FileExistsError(dest)
    partial=dest.with_suffix(dest.suffix+'.part'); h=hashlib.sha256()
    try:
        with urllib.request.urlopen(url,timeout=60) as response, partial.open('xb') as f:
            while chunk:=response.read(1024*1024): f.write(chunk); h.update(chunk)
        if expected and h.hexdigest()!=expected: raise ValueError('Provider SHA256 mismatch')
        partial.rename(dest)
    except Exception:
        partial.unlink(missing_ok=True); raise
    dest.with_suffix(dest.suffix+'.sha256').write_text(h.hexdigest())
    print(f'Downloaded {dest}; SHA256 {h.hexdigest()}. Extract manually, then configure local import.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=URLS,required=True)
    p.add_argument('--output',default='data/downloads',help='Archive destination directory')
    p.add_argument('--accept-license',action='store_true',help='Confirm you reviewed and accept the official dataset license')
    a=p.parse_args(); download(a.dataset,a.output,a.accept_license)
