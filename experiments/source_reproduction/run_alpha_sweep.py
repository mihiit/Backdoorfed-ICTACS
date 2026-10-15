
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fl_instrumented as fl
out=Path(__file__).resolve().parents[1]/'results_rerun'/'alpha_sweep_raw.csv'
alphas=[0.1,0.25,0.5,0.75,1.0,1.5,2.0,5.0]
seeds=range(int(sys.argv[1]),int(sys.argv[2]))
methods=[('fedavg',None),('probeguard',None),('probeguard_k',4)]
out.parent.mkdir(parents=True, exist_ok=True)
rows=pd.read_csv(out).to_dict('records') if out.exists() else []
done={(round(float(r['alpha']),4),int(r['seed']),str(r['defense'])) for r in rows}
out.parent.mkdir(parents=True, exist_ok=True)
t0=time.time()
for a in alphas:
 for seed in seeds:
  for defense,f_assumed in methods:
   key=(round(a,4),int(seed),defense)
   if key in done: continue
   r=fl.run(a,0.2,'fixed',defense,seed,k=2.0,delta=0.1,f_assumed=f_assumed)
   rows.append({'alpha':a,'seed':seed,'defense':defense,'f_assumed':f_assumed,'actual_malicious':4,
                'asr':r['asr'],'clean_accuracy':r['acc'],'cumulative_tpr':r.get('tpr'),'cumulative_fpr':r.get('fpr')})
   pd.DataFrame(rows).to_csv(out,index=False)
   print(f'alpha={a} seed={seed} defense={defense} ASR={r["asr"]:.4f} elapsed={time.time()-t0:.1f}s',flush=True)
