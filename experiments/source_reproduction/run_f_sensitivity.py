import sys, time, os
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fl_instrumented as fl
out=Path(__file__).resolve().parents[1]/'results_rerun'/'f_sensitivity_raw.csv'
# args: seed_start seed_end (exclusive)
s0=int(sys.argv[1]); s1=int(sys.argv[2]); rows=[]
if out.exists(): rows=pd.read_csv(out).to_dict('records')
done={(float(r['alpha']),int(r['seed']),str(r['defense']),int(r['f_assumed']) if pd.notna(r['f_assumed']) else -1) for r in rows}
out.parent.mkdir(parents=True, exist_ok=True)
t0=time.time()
for alpha in [.5,1.0]:
 for seed in range(s0,s1):
  for defense, f_assumed in [('probeguard',None)]+[('probeguard_k',f) for f in [2,3,4,5,6]]:
   key=(float(alpha),int(seed),defense,int(f_assumed) if f_assumed is not None else -1)
   if key in done: continue
   r=fl.run(alpha,.2,'fixed',defense,seed,k=2.0,delta=.1,f_assumed=f_assumed)
   rows.append({'alpha':alpha,'seed':seed,'defense':defense,'f_assumed':f_assumed,'actual_malicious':4,'asr':r['asr'],'clean_accuracy':r['acc'],'cumulative_tpr':r.get('tpr'),'cumulative_fpr':r.get('fpr')})
   pd.DataFrame(rows).to_csv(out,index=False)
   print(f'{alpha=} {seed=} {defense=} {f_assumed=} ASR={r["asr"]:.3f} elapsed={time.time()-t0:.1f}s',flush=True)
