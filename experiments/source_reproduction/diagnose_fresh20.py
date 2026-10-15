import pandas as pd, numpy as np
from scipy.stats import spearmanr, wilcoxon
import matplotlib.pyplot as plt
from pathlib import Path
base=Path(__file__).resolve().parents[1]/'results_rerun'
d=pd.read_csv(base/'fresh20_seed_level.csv')
pgk=d[d.defense.eq('probeguard_k')].copy()
# Detailed diagnostic rows at each alpha, ordered by ASR
pgk['asr_pct']=100*pgk.asr; pgk['acc_pct']=100*pgk.acc; pgk['tpr_pct']=100*pgk.tpr; pgk['fpr_pct']=100*pgk.fpr
pgk.sort_values(['alpha','asr'],ascending=[True,False]).to_csv(base/'fresh20_probeguardk_seed_diagnostics.csv',index=False)
# correlations and descriptive summaries
summ=[]
for a,g in pgk.groupby('alpha'):
    for metric in ['tpr','fpr','acc']:
        r=spearmanr(g.asr,g[metric],nan_policy='omit')
        summ.append({'alpha':a,'analysis':f'Spearman ASR vs {metric.upper()}','rho':r.statistic,'p_value':r.pvalue,'n':len(g)})
    for metric in ['asr','acc','tpr','fpr']:
        vals=g[metric].dropna()*100
        summ.append({'alpha':a,'analysis':f'{metric.upper()} mean','rho':vals.mean(),'p_value':np.nan,'n':len(vals)})
pd.DataFrame(summ).to_csv(base/'fresh20_mechanism_diagnostics.csv',index=False)
# seed-paired method comparison table
wide=d.pivot(index=['alpha','seed'],columns='defense',values='asr').reset_index()
wide['pgk_minus_mad_pp']=100*(wide.probeguard_k-wide.probeguard)
wide['pgk_minus_fedavg_pp']=100*(wide.probeguard_k-wide.fedavg)
wide.to_csv(base/'fresh20_paired_seed_differences.csv',index=False)
# plot ASR vs TPR and FPR per alpha
fig,axs=plt.subplots(2,2,figsize=(10,8))
for ax,a in zip(axs.flat,[0.1,0.5,1.0,10.0]):
 g=pgk[pgk.alpha.eq(a)]
 ax.scatter(g.tpr_pct,g.asr_pct,label='ASR vs TPR')
 for _,r in g.iterrows():
  if r.asr_pct >= g.asr_pct.quantile(.8) or r.asr_pct <= g.asr_pct.quantile(.2): ax.annotate(str(int(r.seed)),(r.tpr_pct,r.asr_pct),fontsize=7,xytext=(3,2),textcoords='offset points')
 ax.set_title(f'α = {a:g}')
 ax.set_xlabel('TPR (%)'); ax.set_ylabel('ASR (%)'); ax.grid(True,alpha=.25)
fig.suptitle('ProbeGuard-K fresh-seed diagnostics: ASR versus detection TPR')
fig.tight_layout(); fig.savefig(base/'fresh20_asr_vs_tpr.png',dpi=180); plt.close(fig)
# plot paired differences at alpha .5 and 1
fig,axs=plt.subplots(1,2,figsize=(10,4.5))
for ax,a in zip(axs,[.5,1.0]):
 q=wide[wide.alpha.eq(a)].sort_values('pgk_minus_mad_pp')
 ax.bar(q.seed.astype(str),q.pgk_minus_mad_pp)
 ax.axhline(0,linewidth=1)
 ax.set_title(f'α = {a:g}: ProbeGuard-K minus MAD')
 ax.set_xlabel('Seed'); ax.set_ylabel('ASR difference (percentage points)'); ax.tick_params(axis='x',rotation=60); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(base/'fresh20_paired_seed_differences.png',dpi=180); plt.close(fig)
# text report
lines=['ProbeGuard-K fresh-seed diagnosis','Seeds: 100–119; 20 per alpha; fixed trigger; 20% malicious; top-f rule assumes true f=4.','TPR/FPR are aggregated over client-round decisions in the implementation, not one-round values.','']
for a,g in pgk.groupby('alpha'):
 lines.append(f'alpha={a:g}: ASR mean={g.asr.mean()*100:.2f}%, median={g.asr.median()*100:.2f}%, range={g.asr.min()*100:.2f}–{g.asr.max()*100:.2f}%; clean accuracy={g.acc.mean()*100:.2f}%; TPR={g.tpr.mean()*100:.2f}%; FPR={g.fpr.mean()*100:.2f}%.')
 for metric in ['tpr','fpr','acc']:
  rr=spearmanr(g.asr,g[metric],nan_policy='omit')
  lines.append(f'  Spearman ASR vs {metric.upper()}: rho={rr.statistic:.3f}, p={rr.pvalue:.3g}')
 if a in [.5,1.0]:
  lines.append('  Highest-ASR seeds: '+', '.join(f'{int(r.seed)} ({r.asr*100:.1f}% ASR, TPR {r.tpr*100:.1f}%, FPR {r.fpr*100:.1f}%)' for _,r in g.nlargest(5,'asr').iterrows()))
 lines.append('')
(base/'fresh20_diagnostic_report.txt').write_text('\n'.join(lines))
print('\n'.join(lines))
