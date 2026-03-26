import json, os, re

def extract_outputs(path, label):
    nb = json.load(open(path, 'r', encoding='utf-8'))
    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"{'#'*70}")
    for i, c in enumerate(nb['cells']):
        outputs = c.get('outputs', [])
        src = ''.join(c.get('source', []))
        if not outputs and c['cell_type'] != 'markdown':
            continue
        # Print markdown cells too for section context
        if c['cell_type'] == 'markdown':
            md = ''.join(c.get('source', []))[:200]
            if md.strip().startswith('#'):
                print(f"\n--- Cell {i} [MARKDOWN] ---")
                print(md[:300])
            continue
        
        print(f"\n--- Cell {i} [CODE OUTPUT] ---")
        print(f"Source preview: {src[:100]}...")
        for o in outputs:
            if 'text' in o:
                text = o['text'] if isinstance(o['text'], str) else ''.join(o['text'])
                print(text[:2000])
            if 'data' in o:
                for k, v in o['data'].items():
                    if k == 'text/plain':
                        val = v if isinstance(v, str) else ''.join(v)
                        print(val[:500])

extract_outputs(r'f:\files\work\btp\simpletm\simpleTMG\simpletm-eda.ipynb', 'EDA NOTEBOOK')
extract_outputs(r'f:\files\work\btp\simpletm\simpleTMG\simpletm-kaggle.ipynb', 'TRAINING NOTEBOOK')
