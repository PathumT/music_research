"""Local official-format adapters. No inferred fine-grained family labels."""
import csv
import json
import re
from pathlib import Path
import yaml

CLASSES = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/instruments.yaml').read_text())['classes']
SOURCES = {
    'irmas': ('https://www.upf.edu/web/mtg/irmas', 'CC BY-NC-SA 3.0; provider additionally prohibits modification/redistribution'),
    'slakh': ('https://www.slakh.com/', 'CC BY 4.0'),
    'medleydb': ('https://medleydb.weebly.com/downloads.html', 'CC BY-NC-SA 4.0'),
    'nsynth': ('https://magenta.tensorflow.org/datasets/nsynth', 'CC BY 4.0'),
    'maestro': ('https://magenta.tensorflow.org/datasets/maestro', 'CC BY-NC-SA 4.0'),
}
ALIASES = dict(zip(['pia','gac','gel','dru','vio','flu','tru','sax','org'],
                   ['piano','acoustic_guitar','electric_guitar','drums','violin','flute','trumpet','saxophone','organ_keyboard']))
ALIASES.update({'piano':'piano','acoustic guitar':'acoustic_guitar','electric guitar':'electric_guitar',
 'electric bass':'bass_guitar','drum set':'drums','violin':'violin','flute':'flute','trumpet':'trumpet',
 'alto saxophone':'saxophone','tenor saxophone':'saxophone','soprano saxophone':'saxophone',
 'baritone saxophone':'saxophone','organ':'organ_keyboard','electric piano':'organ_keyboard'})

def gm(program, drums=False):
    p = int(program)
    if drums: return 'drums'
    if 0 <= p <= 3: return 'piano'
    if 4 <= p <= 5 or 16 <= p <= 20: return 'organ_keyboard'
    if 24 <= p <= 25: return 'acoustic_guitar'
    if 26 <= p <= 31: return 'electric_guitar'
    if 33 <= p <= 37: return 'bass_guitar'
    if p == 40: return 'violin'
    if p == 56: return 'trumpet'
    if 64 <= p <= 67: return 'saxophone'
    if p == 73: return 'flute'
    return None

def safe(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError(f'Path escapes dataset: {relative}')
    return path

def split_for(path):
    for part in path.parts:
        name = part.lower().replace('nsynth-', '')
        if name in ('train','training'): return 'train'
        if name in ('valid','validation'): return 'validation'
        if name in ('test','testing'): return 'test'
    return None

def record(source, root, audio, group, raw, mapped, split=None, midi=None, complete=True, **extra):
    labels = sorted(set(x for x in mapped if x))
    return dict(source=source, audio=str(audio.resolve()), track_id=str(audio.relative_to(root)),
                group_id=f'{source}:{group}', labels=labels, raw_annotations=raw,
                unsupported_labels=[str(x) for x,y in zip(raw,mapped) if y is None],
                known_classes=CLASSES if complete else labels,
                official_split=split, midi=str(midi.resolve()) if midi else None,
                source_url=SOURCES[source][0], license=SOURCES[source][1], **extra)

def import_records(source, root, options):
    root = Path(root).resolve()
    records = []
    if source == 'irmas':
        # Training excerpts do not expose original song IDs: require a researcher-supplied grouping.
        groups = json.loads(Path(options['group_map']).read_text()) if options.get('group_map') else {}
        for audio in sorted(root.rglob('*.wav')):
            side = audio.with_suffix('.txt')
            raw = side.read_text().split() if side.exists() else [audio.parent.name]
            if '[dru]' in audio.name: raw.append('dru')
            key = audio.relative_to(root).as_posix()
            if key not in groups:
                raise ValueError('IRMAS requires group_map JSON {relative audio path: original recording ID}; no guessed song IDs')
            records.append(record(source,root,audio,groups[key],raw,[ALIASES.get(x) for x in raw],
                                  'test' if side.exists() else None,complete=False, annotation_scope='predominant_positive_only'))
    elif source == 'nsynth':
        for meta in sorted(root.rglob('examples.json')):
            for key, item in sorted(json.loads(meta.read_text()).items()):
                family, origin = item['instrument_family_str'], item['instrument_source_str']
                label = 'flute' if family == 'flute' and origin != 'synthetic' else None
                if family == 'organ': label = 'organ_keyboard'
                if family == 'guitar' and origin in ('acoustic','electronic'):
                    label = 'acoustic_guitar' if origin == 'acoustic' else 'electric_guitar'
                # keyboard, bass, brass, reed, string do not identify piano/guitar/trumpet/sax/violin.
                records.append(record(source,root,safe(meta.parent,'audio/'+key+'.wav'),item['instrument_str'],
                    [family],[label],split_for(meta), note_metadata=item))
    elif source == 'maestro':
        files = sorted(root.glob('maestro-v*.csv'))
        for meta in files:
            for item in csv.DictReader(meta.open(encoding='utf-8')):
                group = item['canonical_composer']+'|'+item['canonical_title']
                records.append(record(source,root,safe(root,item['audio_filename']),group,['piano'],['piano'],
                    item['split'],safe(root,item['midi_filename']),composition=item))
    elif source == 'slakh':
        for meta in sorted(root.rglob('metadata.yaml')):
            item = yaml.safe_load(meta.read_text())
            # Stems permit exact program labels and segment-level MIDI activity.
            midi_sources=[]; mix_raw=[]; mix_labels=[]
            for stem, info in sorted(item['stems'].items()):
                if not info.get('audio_rendered',True): continue
                label = gm(info['program_num'], info.get('is_drum',False))
                audio = meta.parent/'stems'/f'{stem}.flac'
                midi_path=meta.parent/'MIDI'/f'{stem}.mid'
                group=item.get('UUID',meta.parent.name)
                raw_label=info.get('inst_class',str(info['program_num']))
                records.append(record(source,root,audio,group,[raw_label],
                    [label],split_for(meta),midi_path,stem_metadata=info))
                midi_sources.append({'path':str(midi_path.resolve()),'label':label})
                mix_raw.append(raw_label); mix_labels.append(label)
            records.append(record(source,root,meta.parent/'mix.flac',item.get('UUID',meta.parent.name),
                mix_raw,mix_labels,split_for(meta),midi_sources=midi_sources,annotation_scope='MIDI-aligned mixture'))
    elif source == 'medleydb':
        for meta in sorted(root.rglob('*_METADATA.yaml')):
            item = yaml.safe_load(meta.read_text())
            song = meta.name.removesuffix('_METADATA.yaml')
            for stem, info in sorted(item['stems'].items()):
                raw = info['instrument']
                raw = raw if isinstance(raw,list) else [raw]
                # Import annotated stems, not whole-song labels copied onto every mix window.
                audio = safe(meta.parent, item.get('stem_dir',song+'_STEMS')+'/'+info['filename'])
                records.append(record(source,root,audio,song,raw,[ALIASES.get(x) for x in raw],
                    complete=False,annotation_scope='stem_instrument_weak'))
    else: raise ValueError(f'Unknown adapter {source}')
    if not records: raise ValueError(f'No supported metadata/audio structure found for {source}')
    return records
