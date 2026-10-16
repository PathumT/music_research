"""Format-contract fixtures, not downloaded research data."""
import json
from pathlib import Path
import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import yaml
from musicai.adapters import import_records,CLASSES,gm,safe
from musicai.data import prepare,validate_manifest,Segments

def audio(path,seconds=6):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=np.arange(8000*seconds)/8000
    sf.write(path,.1*np.sin(t*2*np.pi*220),8000)

def midi(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    m=pretty_midi.PrettyMIDI(); i=pretty_midi.Instrument(0)
    i.notes=[pretty_midi.Note(80,60,0,4)]; m.instruments=[i]; m.write(str(path))

def test_irmas_requires_real_groups(tmp_path):
    root=tmp_path/'irmas'; audio(root/'pia'/'[pia][dru]001.wav',3)
    with pytest.raises(ValueError,match='group_map'): import_records('irmas',root,{})
    group=tmp_path/'groups.json'; group.write_text(json.dumps({'pia/[pia][dru]001.wav':'song-one'}))
    r=import_records('irmas',root,{'group_map':str(group)})[0]
    assert r['labels']==['drums','piano'] and r['known_classes']==r['labels']

def test_nsynth_conservative_mapping(tmp_path):
    folder=tmp_path/'nsynth-train'; folder.mkdir()
    data={}
    for family in ('brass','string','reed','bass','keyboard','guitar','organ','flute'):
        key=family+'-060-100'; audio(folder/'audio'/f'{key}.wav',4)
        data[key]=dict(instrument_family_str=family,instrument_source_str='acoustic',instrument_str=family+'_acoustic_001',pitch=60,velocity=100)
    (folder/'examples.json').write_text(json.dumps(data))
    records=import_records('nsynth',tmp_path,{})
    assert len(records)==8
    assert sum(bool(r['labels']) for r in records)==3
    assert all(r['official_split']=='train' for r in records)

def test_maestro_composition_and_alignment(tmp_path):
    audio(tmp_path/'2018'/'a.wav'); midi(tmp_path/'2018'/'a.midi')
    (tmp_path/'maestro-v3.0.0.csv').write_text('canonical_composer,canonical_title,split,audio_filename,midi_filename\nBach,Prelude,train,2018/a.wav,2018/a.midi\n')
    r=import_records('maestro',tmp_path,{})[0]
    assert r['group_id']=='maestro:Bach|Prelude' and r['midi'].endswith('a.midi')

def test_slakh_mixture_and_prepare(tmp_path):
    root=tmp_path/'raw'; track=root/'train'/'Track00001'; track.mkdir(parents=True)
    audio(track/'stems/S00.flac'); audio(track/'mix.flac'); midi(track/'MIDI/S00.mid')
    # Different byte encodings avoid intentionally identical content deduplication.
    x,sr=sf.read(track/'mix.flac'); sf.write(track/'mix.flac',x*.8,sr)
    (track/'metadata.yaml').write_text(yaml.safe_dump({'UUID':'original-lakh-id','stems':{'S00':dict(program_num=0,is_drum=False,audio_rendered=True,inst_class='Piano')}}))
    records=import_records('slakh',root,{})
    assert len(records)==2 and records[0]['group_id']==records[1]['group_id']
    config=tmp_path/'config.yaml'; out=tmp_path/'manifests'
    config.write_text(yaml.safe_dump(dict(seed=42,segment_seconds=3,target_segments=10000,output=str(out),sources={
        'slakh':dict(root=str(root),license_accepted=True)})))
    rows=prepare(config); assert len(rows)==4
    assert rows[0]['notes'] and all(r['split']=='train' for r in rows)
    original=(out/'segments.json').read_bytes(); prepare(config)
    assert (out/'segments.json').read_bytes()==original
    assert all(n['start']>=0 and n['end']<=3 for r in rows for n in r['notes'])
    dataset=Segments(rows,dict(sample_rate=8000,segment_seconds=3,n_fft=256,hop_length=128,n_mels=32))
    assert len(dataset)==len(rows)
    for x,y,m in dataset: assert x.shape[1]==32 and y.shape==(10,) and m.shape==(10,)

def test_medleydb_stems(tmp_path):
    song='Artist_Song'; folder=tmp_path/song; folder.mkdir()
    audio(folder/(song+'_STEMS')/'s1.wav')
    (folder/(song+'_METADATA.yaml')).write_text(yaml.safe_dump(dict(stem_dir=song+'_STEMS',has_bleed=True,
        stems={'S01':dict(filename='s1.wav',instrument='violin')})))
    r=import_records('medleydb',tmp_path,{})[0]
    assert r['labels']==['violin'] and r['known_classes']==['violin']

def test_unsafe_paths_and_unknown_programs(tmp_path):
    with pytest.raises(ValueError): safe(tmp_path,'../escape.wav')
    assert gm(48) is None and gm(32) is None and gm(0,True)=='drums'

def test_corrupt_audio_report(tmp_path):
    folder=tmp_path/'nsynth-train'; folder.mkdir()
    (folder/'examples.json').write_text(json.dumps({'bad':dict(instrument_family_str='organ',instrument_source_str='electronic',instrument_str='organ_1')}))
    config=tmp_path/'config.yaml'; out=tmp_path/'out'
    config.write_text(yaml.safe_dump(dict(seed=42,output=str(out),sources={'nsynth':dict(root=str(tmp_path),license_accepted=True)})))
    with pytest.raises(RuntimeError): prepare(config)
    assert json.loads((out/'report.json').read_text())['errors']
