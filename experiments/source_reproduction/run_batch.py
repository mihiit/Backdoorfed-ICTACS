import sys,csv,os,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'paper'))
import fl
alpha=float(sys.argv[1]); start_seed=int(sys.argv[2]); end_seed=int(sys.argv[3])
out=Path(__file__).resolve().parents[1]/'results_rerun'/'fresh20_seed_level.csv'
fields=['alpha','m','attack','defense','seed','acc','asr','tpr','fpr']
methods=['fedavg','probeguard','probeguard_k','clean_oracle']
out.parent.mkdir(parents=True, exist_ok=True)
t0=time.time(); rows=[]
for seed in range(start_seed,end_seed):
 for d in methods:
  kw={'k':2.0,'delta':0.1} if d=='probeguard' else {}
  r=fl.run(alpha,0.2,'fixed',d,seed,**kw)
  rows.append({k:r.get(k,'') for k in fields})
  print(f'alpha={alpha} seed={seed} method={d} acc={r["acc"]:.4f} asr={r["asr"]:.4f}',flush=True)
write_header=not out.exists() or out.stat().st_size==0
with open(out,'a',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields)
 if write_header:w.writeheader()
 w.writerows(rows)
print(f'APPENDED {len(rows)} runs in {time.time()-t0:.1f}s',flush=True)
