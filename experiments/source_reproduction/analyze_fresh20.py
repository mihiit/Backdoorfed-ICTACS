import pandas as pd, numpy as np
from scipy.stats import t, wilcoxon
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'results_rerun'/'fresh20_seed_level.csv'
d=pd.read_csv(p)
print('rows',len(d)); print(d.groupby(['alpha','defense']).size().unstack())
assert len(d)==320 and d.groupby(['alpha','defense']).size().eq(20).all()
summary=[]
for (a,defn),g in d.groupby(['alpha','defense']):
 for met in ['acc','asr','tpr','fpr']:
  x=pd.to_numeric(g[met],errors='coerce').dropna().to_numpy()
  if len(x):
   m=x.mean(); sd=x.std(ddof=1); half=t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x))
   summary.append({'alpha':a,'defense':defn,'metric':met,'n':len(x),'mean':m,'sd':sd,'ci95_low':m-half,'ci95_high':m+half})
s=pd.DataFrame(summary)
s.to_csv(str(Path(__file__).resolve().parents[1]/'results_rerun'/'fresh20_summary.csv'),index=False)
print('\nASR means (%), 95% t CIs')
for a in sorted(d.alpha.unique()):
 print('\nalpha',a)
 for defn in ['fedavg','probeguard','probeguard_k','clean_oracle']:
  r=s[(s.alpha==a)&(s.defense==defn)&(s.metric=='asr')].iloc[0]
  acc=s[(s.alpha==a)&(s.defense==defn)&(s.metric=='acc')].iloc[0]
  print(f'{defn:14s} ASR={r["mean"]*100:6.2f}% [{r["ci95_low"]*100:6.2f},{r["ci95_high"]*100:6.2f}] clean={acc["mean"]*100:6.2f}%')
print('\nPaired comparisons: ProbeGuard-K minus comparison, percentage points; paired bootstrap percentile CI (10000 resamples), Wilcoxon signed-rank p')
rng=np.random.default_rng(20261012)
comps=[]
for a in sorted(d.alpha.unique()):
 q=d[d.alpha==a].pivot(index='seed',columns='defense',values='asr').sort_index()
 for other in ['probeguard','fedavg']:
  dif=100*(q['probeguard_k'].to_numpy()-q[other].to_numpy())
  bs=np.array([rng.choice(dif,size=len(dif),replace=True).mean() for _ in range(10000)])
  lo,hi=np.quantile(bs,[.025,.975])
  pval=wilcoxon(dif,zero_method='wilcox',alternative='two-sided').pvalue if np.any(dif) else 1.0
  comps.append({'alpha':a,'comparison':f'ProbeGuard-K minus {other}','mean_diff_pp':dif.mean(),'bootstrap_ci95_low_pp':lo,'bootstrap_ci95_high_pp':hi,'wilcoxon_p_unadjusted':pval,'n_pairs':len(dif)})
  print(f'alpha={a:g}, vs {other:9s}: {dif.mean():+6.2f} pp [{lo:+6.2f},{hi:+6.2f}], p={pval:.5g}')
c=pd.DataFrame(comps)
# Holm correction over all 8 paired tests
order=np.argsort(c.wilcoxon_p_unadjusted.to_numpy()); m=len(c); adj=np.empty(m); running=0
for rank,idx in enumerate(order):
 val=min(1,(m-rank)*c.wilcoxon_p_unadjusted.iloc[idx]); running=max(running,val); adj[idx]=running
c['wilcoxon_p_holm_8tests']=adj
c.to_csv(str(Path(__file__).resolve().parents[1]/'results_rerun'/'fresh20_paired_tests.csv'),index=False)
print('\nHolm adjusted p values:'); print(c[['alpha','comparison','wilcoxon_p_unadjusted','wilcoxon_p_holm_8tests']].to_string(index=False))
# Plot ASR means with 95% t intervals
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,5))
methods=['fedavg','probeguard','probeguard_k','clean_oracle']
labels={'fedavg':'FedAvg','probeguard':'ProbeGuard (MAD)','probeguard_k':'ProbeGuard-K','clean_oracle':'Benign-only oracle'}
for defn in methods:
 g=s[(s.defense==defn)&(s.metric=='asr')].sort_values('alpha')
 ax.errorbar(g.alpha,100*g['mean'],yerr=[100*(g['mean']-g.ci95_low),100*(g.ci95_high-g['mean'])],marker='o',capsize=3,label=labels[defn])
ax.set_xscale('log'); ax.set_xticks([.1,.5,1,10],labels=['0.1','0.5','1','10']); ax.set_xlabel('Dirichlet concentration α (larger = more IID)'); ax.set_ylabel('Attack success rate (%)'); ax.set_ylim(-5,105); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(str(Path(__file__).resolve().parents[1]/'results_rerun'/'fresh20_asr_plot.png'),dpi=180); plt.close(fig)
print('\nSaved summary, paired tests and plot.')
