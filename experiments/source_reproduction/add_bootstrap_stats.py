from pathlib import Path
import pandas as pd, numpy as np, zipfile
from scipy import stats
base=Path(__file__).resolve().parents[1]/'results_rerun'; d=pd.read_csv(base/'f_sensitivity_raw.csv'); rng=np.random.default_rng(20261012); B=20000
# percentile bootstrap means by seed
rows=[]
for (a,defense,f),g in d.groupby(['alpha','defense','f_assumed'],dropna=False):
 x=g.asr.to_numpy(); means=np.mean(rng.choice(x,(B,len(x)),replace=True),axis=1)
 rows.append({'alpha':a,'defense':'MAD' if defense=='probeguard' else 'Top-f','assumed_f':None if pd.isna(f) else int(f),'n':len(x),'mean_ASR':x.mean(),'bootstrap_ci95_low':np.quantile(means,.025),'bootstrap_ci95_high':np.quantile(means,.975)})
pd.DataFrame(rows).to_csv(base/'f_sensitivity_bootstrap_ci.csv',index=False)
# paired bootstrap diffs and Wilcoxon p with Holm correction across 10 comparisons
ps=[]; pairs=[]
for a in [.5,1.0]:
 mad=d[(d.alpha==a)&(d.defense=='probeguard')].set_index('seed')
 for f in [2,3,4,5,6]:
  top=d[(d.alpha==a)&(d.defense=='probeguard_k')&(d.f_assumed==f)].set_index('seed')
  diff=(top.asr-mad.asr).sort_index().to_numpy(); boot=np.mean(rng.choice(diff,(B,len(diff)),replace=True),axis=1)
  test=stats.wilcoxon(diff); ps.append(test.pvalue)
  pairs.append({'alpha':a,'assumed_f':f,'n':len(diff),'mean_paired_ASR_difference':diff.mean(),'bootstrap_ci95_low':np.quantile(boot,.025),'bootstrap_ci95_high':np.quantile(boot,.975),'wilcoxon_p':test.pvalue})
# Holm adjustment
order=np.argsort(ps); adj=np.empty(len(ps)); running=0
for rank,idx in enumerate(order):
 val=min(1.0,(len(ps)-rank)*ps[idx]); running=max(running,val); adj[idx]=running
for i,r in enumerate(pairs): r['holm_adjusted_p']=adj[i]
pd.DataFrame(pairs).to_csv(base/'f_sensitivity_paired_bootstrap.csv',index=False)
# regenerate report append bootstrap note and tables
p=base/'f_sensitivity_report.txt'; txt=p.read_text()
txt += '\nBootstrap robustness check: 20,000 seed-resampling replicates with fixed RNG seed 20261012 were used for percentile 95% CIs for means and paired ASR differences. Paired Wilcoxon p-values across the 10 alpha × assumed-f comparisons are Holm-adjusted in f_sensitivity_paired_bootstrap.csv. Bootstrap intervals respect the bounded [0,1] ASR outcome, unlike unconstrained t intervals that can extend below zero.\n'
p.write_text(txt)
files=['f_sensitivity_raw.csv','f_sensitivity_summary.csv','f_sensitivity_paired_tests.csv','f_sensitivity_monotonicity.csv','f_sensitivity_bootstrap_ci.csv','f_sensitivity_paired_bootstrap.csv','f_sensitivity_asr_plot.png','f_sensitivity_report.txt']
with zipfile.ZipFile(base/'probeguard_f_sensitivity_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(base/f,arcname=f)
 z.write(Path(__file__).resolve().parent/'run_f_sensitivity.py',arcname='run_f_sensitivity.py')
 z.write(Path(__file__).resolve().parent/'fl_instrumented.py',arcname='fl_instrumented.py')
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(pairs).to_string(index=False))
