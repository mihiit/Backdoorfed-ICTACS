
import sys, pathlib, pandas as pd, time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import fl_instrumented as fl
out=pathlib.Path(__file__).resolve().parents[1]/'results_rerun'/'malicious_fraction_sweep_raw.csv'
rows=pd.read_csv(out).to_dict("records") if out.exists() else []
done={(float(r["alpha"]),float(r["m_frac"]),int(r["seed"]),str(r["defense"])) for r in rows}
t=time.time()
for a in [0.1,0.5,1.0]:
 for mf in [0.1,0.2,0.3]:
  for seed in range(int(sys.argv[1]),int(sys.argv[2])):
   for d in ["fedavg","probeguard","probeguard_k"]:
    key=(a,mf,seed,d)
    if key in done: continue
    r=fl.run(a,mf,"fixed",d,seed,k=2.0,delta=0.1,f_assumed=None if d!="probeguard_k" else int(round(mf*20)))
    rows.append({"alpha":a,"m_frac":mf,"actual_malicious":int(round(mf*20)),"seed":seed,"defense":d,"asr":r["asr"],"clean_accuracy":r["acc"],"tpr":r.get("tpr"),"fpr":r.get("fpr")})
    pd.DataFrame(rows).to_csv(out,index=False); done.add(key)
   print(a,mf,seed,len(rows),round(time.time()-t,1),flush=True)
