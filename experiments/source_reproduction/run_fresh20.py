import sys, time, csv, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'paper'))
import fl
import numpy as np
out=str(Path(__file__).resolve().parents[1]/'results_rerun'/'fresh20_seed_level.csv')
summary=str(Path(__file__).resolve().parents[1]/'results_rerun'/'fresh20_summary.csv')
seeds=range(100,120)
alphas=[0.1,0.5,1.0,10.0]
methods=['fedavg','probeguard','probeguard_k','clean_oracle']
rows=[]; start=time.time(); total=len(seeds)*len(alphas)*len(methods); count=0
for a in alphas:
  for seed in seeds:
    for d in methods:
      kw={'k':2.0,'delta':0.1} if d=='probeguard' else {}
      r=fl.run(a,0.2,'fixed',d,seed,**kw)
      rows.append({k:r.get(k,'') for k in ['alpha','m','attack','defense','seed','acc','asr','tpr','fpr']})
      count+=1
      if count%20==0: print(f'{count}/{total} runs; elapsed {time.time()-start:.1f}s',flush=True)
Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['alpha','m','attack','defense','seed','acc','asr','tpr','fpr']);w.writeheader();w.writerows(rows)
# summaries: mean, sd, t CI; use t critical approx 2.093 for n=20
import pandas as pd
from scipy.stats import t
D=pd.DataFrame(rows)
ss=[]
for (a,d),g in D.groupby(['alpha','defense']):
  n=len(g); crit=t.ppf(.975,n-1)
  for metric in ['acc','asr','tpr','fpr']:
    vals=pd.to_numeric(g[metric],errors='coerce').dropna()
    if len(vals):
      mean=vals.mean(); sd=vals.std(ddof=1); half= t.ppf(.975,len(vals)-1)*sd/np.sqrt(len(vals))
    else: mean=sd=half=np.nan
    if metric in ['acc','asr']:
      ss.append({'alpha':a,'defense':d,'n':n,'metric':metric,'mean':mean,'sd':sd,'ci95_low':mean-half,'ci95_high':mean+half})
S=pd.DataFrame(ss)
S.to_csv(summary,index=False)
print('\nSUMMARY')
print(S[S.metric.eq('asr')].pivot(index='alpha',columns='defense',values='mean').to_string(float_format=lambda x:f'{100*x:.2f}%'))
print('\nASR paired comparisons (mean difference in percentage points, paired t CI, and Wilcoxon p):')
from scipy.stats import wilcoxon, ttest_rel
for a in alphas:
  q=D[D.alpha==a].pivot(index='seed',columns='defense',values='asr')
  for d in ['probeguard_k','probeguard','fedavg']:
    if d=='probeguard_k': continue
    diff=100*(q['probeguard_k']-q[d]); n=len(diff); half=t.ppf(.975,n-1)*diff.std(ddof=1)/np.sqrt(n)
    try: p=wilcoxon(diff).pvalue
    except: p=np.nan
    print(f'alpha={a:g}, ProbeGuard-K minus {d}: {diff.mean():+.2f} pp; 95% paired t CI [{diff.mean()-half:+.2f}, {diff.mean()+half:+.2f}], Wilcoxon p={p:.4g}')
print(f'Wrote {out} and {summary}; elapsed {time.time()-start:.1f}s')
