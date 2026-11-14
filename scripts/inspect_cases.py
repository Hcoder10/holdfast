import json, sys, collections
root = sys.argv[1]
out = open(sys.argv[2], 'w', encoding='utf8')
for d in ['gc', 'if', 'insight', 'outsight']:
    data = json.load(open(f'{root}/{d}/data.json', encoding='utf8'))
    out.write(f'\n===== {d}: {len(data)} cases\n')
    keys = collections.Counter(k for c in data for k in c)
    out.write(f'keys: {dict(keys)}\n')
    out.write(f"perspective: {dict(collections.Counter(c.get('perspective') for c in data))}\n")
    big = {k: sum(len(json.dumps(c.get(k))) for c in data) for k in keys}
    out.write(f'bytes per key: {big}\n')
    for c in data:
        small = {k: v for k, v in c.items() if len(json.dumps(v)) < 3000}
        out.write(json.dumps(small, ensure_ascii=False) + '\n')
out.close()
