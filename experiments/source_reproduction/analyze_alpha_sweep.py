import pandas as pd, numpy as np, scipy.stats as st, matplotlib.pyplot as plt, zipfile, os
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'results_rerun'; raw=pd.read_csv(out/'alpha_sweep_raw.csv')
# run-level summary
summ=raw.groupby(['alpha','defense']).agg(n=('seed','nunique'),asr_mean=('asr','mean'),asr_sd=('asr','std'),acc_mean=('clean_accuracy','mean'),acc_sd=('clean_accuracy','std'),tpr_mean=('cumulative_tpr','mean'),fpr_mean=('cumulative_fpr','mean')).reset_index()
for metric in ['asr','clean_accuracy','cumulative_tpr','cumulative_fpr']:
    g=raw.groupby(['alpha','defense'])[metric]
    ci=g.agg(['mean','std','count']).reset_index(); ci[metric+'_ci_low']=ci['mean']-st.t.ppf(.975,ci['count']-1)*ci['std']/np.sqrt(ci['count']); ci[metric+'_ci_high']=ci['mean']+st.t.ppf(.975,ci['count']-1)*ci['std']/np.sqrt(ci['count'])
    summ=summ.merge(ci[['alpha','defense',metric+'_ci_low',metric+'_ci_high']],on=['alpha','defense'],how='left')
summ.to_csv(out/'alpha_sweep_summary.csv',index=False)
# paired stats per alpha
rows=[]
for a in sorted(raw.alpha.unique()):
 sub=raw[raw.alpha==a].pivot(index='seed',columns='defense',values='asr').dropna()
 for method in ['probeguard','probeguard_k']:
  if method not in sub: continue
  diff=sub[method]-sub['fedavg']
  try: w=st.wilcoxon(diff,alternative='two-sided',zero_method='wilcox')
  except Exception: w=(np.nan,np.nan)
  rows.append({'alpha':a,'comparison':f'{method} - fedavg','n_paired':len(diff),'mean_difference':diff.mean(),'median_difference':diff.median(),'ci95_low':diff.mean()-st.t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)),'ci95_high':diff.mean()+st.t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)),'wilcoxon_p':w.pvalue if hasattr(w,'pvalue') else w[1]})
 if 'probeguard_k' in sub and 'probeguard' in sub:
  diff=sub['probeguard_k']-sub['probeguard']
  w=st.wilcoxon(diff)
  rows.append({'alpha':a,'comparison':'Top-f - MAD','n_paired':len(diff),'mean_difference':diff.mean(),'median_difference':diff.median(),'ci95_low':diff.mean()-st.t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)),'ci95_high':diff.mean()+st.t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)),'wilcoxon_p':w.pvalue})
tests=pd.DataFrame(rows)
# Holm correction across all tests
if len(tests):
 p=tests.wilcoxon_p.to_numpy(); order=np.argsort(p); adj=np.empty(len(p)); running=0
 for rank,idx in enumerate(order): running=max(running,(len(p)-rank)*p[idx]); adj[idx]=min(running,1)
 tests['holm_adjusted_p']=adj
tests.to_csv(out/'alpha_sweep_paired_tests.csv',index=False)
# plot means + t CI
fig,ax=plt.subplots(figsize=(9,5.5))
labels={'fedavg':'FedAvg','probeguard':'ProbeGuard (MAD)','probeguard_k':'ProbeGuard (Top-f, f=4)'}
for d in ['fedavg','probeguard','probeguard_k']:
 s=summ[summ.defense==d].sort_values('alpha')
 x=s.alpha.to_numpy(); y=s.asr_mean.to_numpy(); lo=s.asr_ci_low.to_numpy(); hi=s.asr_ci_high.to_numpy()
 ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),marker='o',capsize=3,label=labels[d])
ax.set_xscale('log'); ax.set_xticks(sorted(raw.alpha.unique()),[str(a) for a in sorted(raw.alpha.unique())]); ax.set_xlabel('Dirichlet concentration α (log scale)'); ax.set_ylabel('Final attack success rate (ASR)'); ax.set_ylim(0,1.03); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/'alpha_sweep_asr_plot.png',dpi=200); plt.close(fig)
# report
with open(out/'alpha_sweep_report.txt','w') as f:
 f.write('ProbeGuard alpha-sweep sensitivity study\n')
 f.write(f'Rows={len(raw)}; complete alpha-seed groups={(raw.groupby(["alpha","seed"]).size()==3).sum()}; expected={len(raw.alpha.unique())*raw.seed.nunique()}\n')
 f.write('Dataset: sklearn digits; 20 clients; actual malicious clients=4/20; 30 rounds; fixed trigger; MLP; MAD k=2, delta=0.1; Top-f assumes f=4; seeds 100-119 intended.\n')
 f.write('Reported CIs are t-based 95% confidence intervals over independent seed-level runs. Paired comparisons use seed-matched Wilcoxon tests with Holm adjustment across the reported tests.\n\n')
 f.write(summ.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
 f.write('\n\nPAIRED TESTS\n'+tests.to_string(index=False,float_format=lambda x:f'{x:.6g}'))
 f.write('\n\nCaveat: These results apply to this controlled synthetic/benchmark setup and do not establish performance on real federated deployments. Incomplete cells, if any, are explicitly visible via n in summary.\n')
# bundle
files=['alpha_sweep_raw.csv','alpha_sweep_summary.csv','alpha_sweep_paired_tests.csv','alpha_sweep_asr_plot.png','alpha_sweep_report.txt','run_alpha_sweep.py','analyze_alpha_sweep.py']
with zipfile.ZipFile(out/'probeguard_alpha_sweep_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in files:
  p=(Path(__file__).resolve().parent/name) if name.endswith('.py') else out/name
  if p.exists(): z.write(p,arcname=p.name)
print('rows',len(raw),'seeds',raw.seed.nunique(),'alphas',sorted(raw.alpha.unique()))
print(summ[['alpha','defense','n','asr_mean','asr_sd','acc_mean','tpr_mean','fpr_mean']].to_string(index=False))
print(tests.to_string(index=False))
