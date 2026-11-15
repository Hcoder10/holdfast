import re,sys,hashlib,json,statistics as st
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
def h(d):
    m=hashlib.md5()
    for f in sorted(p for p in Path(d).rglob('*') if p.is_file()): m.update(f.read_bytes())
    return m.hexdigest()
def dim(c): return c[:2] if c[:2] in('GC','IF') else ('IE' if c in IE else 'OE')
IE={'OE002','OE005','OE006','OE008','OE009','OE010'}
base={}; trials={}
for t in (2,3,4):
    mode='memory+shadow' if t==2 else 'open'
    trials[t]={}
    for l in open(f'runs/hillclimb/trial_{t:02d}.txt',encoding='utf8').read().splitlines()[6:]:
        m=re.match(r'(\w+) \((\w+)\) baseline ([\d.]+) -> ([\d.]+)',l)
        if not m: continue
        c=m.group(1); base.setdefault(c,[float(m.group(3))])
        a=Path(f'runs/jin_hc{t:02d}/lingbot2_14b_{mode}')/c/'sheets'; b=Path('runs/jin_half1/lingbot2_14b_open')/c/'sheets'
        same=a.is_dir() and h(a)==h(b)
        if same: base[c].append(float(m.group(4)))
        trials[t][c]=(float(m.group(4)),same)
pool={c:st.mean(v) for c,v in base.items()}
json.dump({c:{'judgings':v,'mean':round(pool[c],2)} for c,v in sorted(base.items())},open('runs/hillclimb/baseline_pooled.json','w'),indent=1)
for c,v in sorted(base.items()):
    if len(v)>1: print(c,v)
def overall(sc): return {d:round(st.mean(s for c,s in sc.items() if dim(c)==d),2) for d in('GC','IF','IE','OE')}
ob=overall(pool); print('pooled baseline',ob,round(st.mean(ob.values()),2))
for t,sc in trials.items():
    eff={c:(pool[c] if same else s) for c,(s,same) in sc.items()}
    o=overall(eff); print('trial',t,o,round(st.mean(o.values()),2),'changed videos:',sum(not s for _,s in sc.values()))
