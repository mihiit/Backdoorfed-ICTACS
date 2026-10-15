
import sys, pathlib, pandas as pd, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fl_instrumented as fl
out=pathlib.Path(__file__).resolve().parents[1]/'results_rerun'/'mad_threshold_sensitivity_raw.csv'
alphas=[0.1,0.5,1.0]; ks=[1.0,1.5,2.0,2.5,3.0]
rows=pd.read_csv(out).to_dict("records") if out.exists() else []
done={(round(float(r["alpha"]),4),int(r["seed"]),str(r["defense"]),None if pd.isna(r["k"]) else float(r["k"])) for r in rows}
out.parent.mkdir(parents=True, exist_ok=True)
t0=time.time()
for alpha in alphas:
 for seed in range(int(sys.argv[1]),int(sys.argv[2])):
  for k in ks:
   key=(round(alpha,4),seed,"MAD",k)
   if key in done: continue
   r=fl.run(alpha,0.2,"fixed","probeguard",seed,k=k,delta=0.1)
   rows.append({"alpha":alpha,"seed":seed,"defense":"MAD","k":k,"delta":0.1,"asr":r["asr"],"clean_accuracy":r["acc"],"tpr":r.get("tpr"),"fpr":r.get("fpr")})
   pd.DataFrame(rows).to_csv(out,index=False); done.add(key)
  key=(round(alpha,4),seed,"Top-f",None)
  if key not in done:
   r=fl.run(alpha,0.2,"fixed","probeguard_k",seed,k=2.0,delta=0.1,f_assumed=4)
   rows.append({"alpha":alpha,"seed":seed,"defense":"Top-f","k":None,"delta":None,"asr":r["asr"],"clean_accuracy":r["acc"],"tpr":r.get("tpr"),"fpr":r.get("fpr")})
   pd.DataFrame(rows).to_csv(out,index=False); done.add(key)
  print(alpha,seed,len(rows),round(time.time()-t0,1),flush=True)
