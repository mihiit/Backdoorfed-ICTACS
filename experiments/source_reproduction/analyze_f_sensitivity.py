from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt, zipfile, os
from scipy import stats
base=Path(__file__).resolve().parents[1]/'results_rerun'; raw=pd.read_csv(base/'f_sensitivity_raw.csv')
# summaries and 95% t intervals
summ=[]
for (a,d,f),g in raw.groupby(['alpha','defense','f_assumed'],dropna=False):
 row={'alpha':a,'defense':'MAD' if d=='probeguard' else 'Top-f','assumed_f':None if pd.isna(f) else int(f),'n_seeds':len(g)}
 for col,label in [('asr','final_ASR'),('clean_accuracy','clean_accuracy'),('cumulative_tpr','cumulative_TPR'),('cumulative_fpr','cumulative_FPR')]:
  vals=g[col].dropna().astype(float); n=len(vals); mean=vals.mean(); sd=vals.std(ddof=1); se=sd/np.sqrt(n); tcrit=stats.t.ppf(.975,n-1)
  row[label+'_mean']=mean; row[label+'_sd']=sd; row[label+'_ci95_low']=mean-tcrit*se; row[label+'_ci95_high']=mean+tcrit*se
 summ.append(row)
s=pd.DataFrame(summ).sort_values(['alpha','defense','assumed_f'],na_position='first'); s.to_csv(base/'f_sensitivity_summary.csv',index=False)
# paired comparisons Top-f vs MAD, by alpha and assumed f
paired=[]
for a in [.5,1.0]:
 mad=raw[(raw.alpha==a)&(raw.defense=='probeguard')].set_index('seed')
 for f in [2,3,4,5,6]:
  g=raw[(raw.alpha==a)&(raw.defense=='probeguard_k')&(raw.f_assumed==f)].set_index('seed')
  x=g.asr-mad.asr; n=len(x); m=x.mean(); sd=x.std(ddof=1); ci=stats.t.interval(.95,n-1,loc=m,scale=stats.sem(x))
  try: wp=stats.wilcoxon(x).pvalue
  except ValueError: wp=1.0
  paired.append({'alpha':a,'assumed_f':f,'n_paired_seeds':n,'mean_ASR_difference_Topf_minus_MAD':m,'ci95_low':ci[0],'ci95_high':ci[1],'wilcoxon_p':wp,'topf_lower_ASR_seed_count':int((x<0).sum()),'mean_clean_accuracy_topf':g.clean_accuracy.mean(),'mean_TPR':g.cumulative_tpr.mean(),'mean_FPR':g.cumulative_fpr.mean()})
pair=pd.DataFrame(paired); pair.to_csv(base/'f_sensitivity_paired_tests.csv',index=False)
# monotonicity: compare ASR across assumed f paired by seed
mono=[]
for a in [.5,1.0]:
 g=raw[(raw.alpha==a)&(raw.defense=='probeguard_k')]
 for f1,f2 in [(2,3),(3,4),(4,5),(5,6),(2,4),(4,6)]:
  x=g[g.f_assumed==f1].set_index('seed').asr; y=g[g.f_assumed==f2].set_index('seed').asr
  delta=y-x
  mono.append({'alpha':a,'comparison':f'{f2} vs {f1}','mean_ASR_change':delta.mean(),'seeds_ASR_reduced':int((delta<0).sum()),'seeds_ASR_unchanged':int((delta==0).sum()),'seeds_ASR_increased':int((delta>0).sum())})
pd.DataFrame(mono).to_csv(base/'f_sensitivity_monotonicity.csv',index=False)
# plot mean ASR with 95% t CIs
fig,axs=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
for ax,a in zip(axs,[.5,1.0]):
 q=s[(s.alpha==a)&(s.defense=='Top-f')].sort_values('assumed_f')
 ax.errorbar(q.assumed_f,q.final_ASR_mean,yerr=[q.final_ASR_mean-q.final_ASR_ci95_low,q.final_ASR_ci95_high-q.final_ASR_mean],marker='o',capsize=4,label='Top-f')
 m=s[(s.alpha==a)&(s.defense=='MAD')].iloc[0]
 ax.axhline(m.final_ASR_mean,linestyle='--',label=f'MAD mean = {m.final_ASR_mean:.3f}')
 ax.set_title(f'Non-IID severity α={a:g}'); ax.set_xlabel('Assumed number of malicious clients (f)'); ax.set_xticks([2,3,4,5,6]); ax.grid(alpha=.25); ax.legend()
axs[0].set_ylabel('Final attack success rate (ASR)')
fig.suptitle('Sensitivity of rank-based filtering to misspecified malicious-client count (20 seeds)')
fig.tight_layout(); fig.savefig(base/'f_sensitivity_asr_plot.png',dpi=200); plt.close(fig)
# plain-text report
lines=['ProbeGuard f-sensitivity experiment — reproducibility report','',
'Question: How sensitive is rank-based Top-f filtering to the assumed malicious-client count?',
'Design: 20 seeds (100–119), 20 clients, exactly 4 malicious clients (20%), 30 rounds, fixed trigger, sklearn digits, MLP 64-32-10, clean root set=100. Dirichlet α in {0.5, 1.0}. MAD uses k=2.0 and delta=0.1. Top-f removes the f clients with highest ProbeGuard score each round, with f assumed in {2,3,4,5,6}. Each seed/configuration rerun under the same seed across methods.',
'Total runs: 240 (2 alpha values × 20 seeds × [MAD + five Top-f settings]).',
'Primary outcome: final attack success rate (ASR); secondary: clean accuracy and cumulative client-level TPR/FPR. Confidence intervals are two-sided 95% Student-t intervals across seeds. Paired tests compare each Top-f setting against MAD for the same seed using Wilcoxon signed-rank tests. These are exploratory paired tests; no multiplicity correction applied.',
'', 'Mean results by alpha and rule:']
for _,r in s.iterrows():
 f='MAD' if r['defense']=='MAD' else f"Top-f (assumed f={int(r['assumed_f'])})"
 lines.append(f"alpha={r['alpha']:.1f} | {f}: ASR={r['final_ASR_mean']:.3f} (SD={r['final_ASR_sd']:.3f}, 95% CI [{r['final_ASR_ci95_low']:.3f}, {r['final_ASR_ci95_high']:.3f}]); clean acc={r['clean_accuracy_mean']:.3f}; TPR={r['cumulative_TPR_mean']:.3f}; FPR={r['cumulative_FPR_mean']:.3f}")
lines+=['','Paired Top-f minus MAD ASR comparisons:']
for _,r in pair.iterrows(): lines.append(f"alpha={r['alpha']:.1f}, assumed f={int(r['assumed_f'])}: ΔASR={r['mean_ASR_difference_Topf_minus_MAD']:.3f}, 95% CI [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}], Wilcoxon p={r['wilcoxon_p']:.4g}; Top-f lower ASR on {int(r['topf_lower_ASR_seed_count'])}/20 paired seeds; TPR={r['mean_TPR']:.3f}, FPR={r['mean_FPR']:.3f}")
lines+=['','Interpretation guardrails: The f=4 row matches the actual malicious-client count and is the correctly specified rank-rule condition. f<4 under-removes malicious clients; f>4 removes benign clients too, raising false-positive cost. More aggressive removal often lowers ASR but is not a free win: clean accuracy and FPR must be considered. The benchmark is a small sklearn-digits simulation, not a real-world FL deployment. This experiment supports sensitivity analysis, not a claim that Top-f works when f is unknown.']
(base/'f_sensitivity_report.txt').write_text('\n'.join(lines)+'\n')
# package artifacts and runner source, code copy and readme
files=['f_sensitivity_raw.csv','f_sensitivity_summary.csv','f_sensitivity_paired_tests.csv','f_sensitivity_monotonicity.csv','f_sensitivity_asr_plot.png','f_sensitivity_report.txt']
with zipfile.ZipFile(base/'probeguard_f_sensitivity_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in files: z.write(base/f,arcname=f)
 z.write(Path(__file__).resolve().parent/'run_f_sensitivity.py',arcname='run_f_sensitivity.py')
 z.write(Path(__file__).resolve().parent/'fl_instrumented.py',arcname='fl_instrumented.py')
print('RAW rows',len(raw),'summary rows',len(s),'paired rows',len(pair))
print(s[['alpha','defense','assumed_f','n_seeds','final_ASR_mean','final_ASR_sd','final_ASR_ci95_low','final_ASR_ci95_high','clean_accuracy_mean','cumulative_TPR_mean','cumulative_FPR_mean']].to_string(index=False))
print('\nPAIRED\n',pair.to_string(index=False))
