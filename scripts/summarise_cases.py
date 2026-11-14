import json, sys, collections, re
root = sys.argv[1]
w = sys.stdout.buffer.write
def p(s): w((s + '\n').encode('ascii', 'replace'))
for d in ['gc', 'if', 'insight', 'outsight']:
    data = json.load(open(f'{root}/{d}/data.json', encoding='utf8'))
    p(f'\n===== {d}: {len(data)}')
    p('sub_category: ' + str(dict(collections.Counter(c.get('sub_category') for c in data))))
    q0 = data[0]['questions']
    p('questions type: ' + type(q0).__name__ + ' n=' + str(len(q0)))
    first = q0[0] if isinstance(q0, list) else q0
    p('first question: ' + json.dumps(first)[:1500])
    nq = [len(c['questions']) for c in data]
    p(f'questions per case: min {min(nq)} max {max(nq)} mean {sum(nq)/len(nq):.1f}')
    fam = collections.Counter()
    for c in data:
        for q in c['questions']:
            if isinstance(q, dict):
                for k in ('family', 'evidence_family', 'aspect', 'group', 'type', 'dimension'):
                    if k in q: fam[(k, str(q[k])[:40])] += 1
    p('question tags: ' + str(fam.most_common(25)))
    acts = collections.Counter(re.sub(r'\*\d+', '', c['action']) for c in data)
    p('action patterns: ' + str(acts.most_common(12)))
    for c in data:
        p(f"  {c['task_id']} [{c.get('sub_category','')}] {c['action'][:40]} :: {c['prompt'][:150]}")
