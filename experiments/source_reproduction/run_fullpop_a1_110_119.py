import sys,time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t, spearmanr, pearsonr
src=Path(__file__).resolve().parent/'paper'/'fl.py'; mod=Path(__file__).resolve().parent/'fl_instrumented.py'
s=src.read_text()
s=s.replace('    traj = []\n    for r in range(ROUNDS):','    traj = []\n    diagnostics = []\n    for r in range(ROUNDS):')
s=s.replace('            else:\n                fl = mad_flag(s, k, delta)\n            if fl.all(): fl[:] = False\n            upd = U[~fl].mean(0)','            else:\n                fl = mad_flag(s, k, delta)\n            if fl.all(): fl[:] = False\n            diagnostics.append({"round": r+1, "scores": s.copy(), "flagged": fl.copy(), "malicious": ft.copy()})\n            upd = U[~fl].mean(0)')
s=s.replace('    if trajectory:\n        out["traj"] = traj','    if trajectory:\n        out["traj"] = traj\n    if defense in ("probeguard", "probeguard+", "probeguard_k", "pg_abs", "pg_local", "pg_knn", "pg_pct90", "pg_pct95", "pg_pct975", "pg_oracle"):\n        out["diagnostics"] = diagnostics')
mod.write_text(s)
sys.path.insert(0,str(mod.parent)); import importlib.util
spec=importlib.util.spec_from_file_location('fl_instrumented',mod); fl=importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
out=Path(__file__).resolve().parents[1]/'results_rerun'; seeds=range(110,120); alphas=[1.0]; methods=['probeguard','probeguard_k']
client_rows=[]; round_rows=[]; run_rows=[]; out.parent.mkdir(parents=True, exist_ok=True)
t0=time.time(); total=len(seeds)*len(alphas)*len(methods); count=0
for a in alphas:
 for seed in seeds:
  for defense in methods:
   kw={'k':2.0,'delta':0.1} if defense=='probeguard' else {}
   r=fl.run(a,.2,'fixed',defense,seed,trajectory=True,**kw); count+=1
   run_rows.append({'alpha':a,'seed':seed,'defense':defense,'acc':r['acc'],'asr':r['asr'],'tpr':r.get('tpr'),'fpr':r.get('fpr')})
   for d in r['diagnostics']:
    scores=d['scores']; flags=d['flagged']; mal=d['malicious']; order=np.argsort(-scores); rank=np.empty(len(scores),int); rank[order]=np.arange(1,len(scores)+1)
    auc=float(np.mean([(scores[i]>scores[j])+.5*(scores[i]==scores[j]) for i in np.where(mal)[0] for j in np.where(~mal)[0]]))
    round_rows.append({'alpha':a,'seed':seed,'defense':defense,'round':d['round'],'auc':auc,'tpr':float(flags[mal].mean()),'fpr':float(flags[~mal].mean()),'malicious_top4_fraction':float((rank[mal]<=4).mean()),'malicious_mean_rank':float(rank[mal].mean()),'benign_mean_rank':float(rank[~mal].mean()),'malicious_score_mean':float(scores[mal].mean()),'benign_score_mean':float(scores[~mal].mean()),'asr':float(r['traj'][d['round']-1][2]),'acc':float(r['traj'][d['round']-1][1])})
   print(f'{count}/{total}: alpha={a} seed={seed} {defense} ASR={r["asr"]:.3f} elapsed={time.time()-t0:.1f}s',flush=True)
Rnew=pd.DataFrame(run_rows); Qnew=pd.DataFrame(round_rows); Rold=pd.read_csv(out/'fullpop_mechanism_run_level.csv') if (out/'fullpop_mechanism_run_level.csv').exists() else pd.DataFrame(); Qold=pd.read_csv(out/'fullpop_mechanism_round_level.csv') if (out/'fullpop_mechanism_round_level.csv').exists() else pd.DataFrame(); R=pd.concat([Rold,Rnew],ignore_index=True).drop_duplicates(['alpha','seed','defense'],keep='last'); Q=pd.concat([Qold,Qnew],ignore_index=True).drop_duplicates(['alpha','seed','defense','round'],keep='last'); R.to_csv(out/'fullpop_mechanism_run_level.csv',index=False); Q.to_csv(out/'fullpop_mechanism_round_level.csv',index=False)
# summarize by alpha, defense, phase with seed-level uncertainty
phases=[('early',1,5),('middle',6,15),('late',16,30)]
S=[]
for (a,d),g in Q.groupby(['alpha','defense']):
 for phase,lo,hi in phases:
  x=g[g['round'].between(lo,hi)].groupby('seed')[['auc','tpr','fpr','malicious_top4_fraction','asr','acc']].mean()
  for metric in x.columns:
   vals=x[metric].to_numpy(); n=len(vals); m=vals.mean(); half=t.ppf(.975,n-1)*vals.std(ddof=1)/np.sqrt(n)
   S.append({'alpha':a,'defense':d,'phase':phase,'metric':metric,'n_seeds':n,'mean':m,'sd':vals.std(ddof=1),'ci95_low':m-half,'ci95_high':m+half})
pd.DataFrame(S).to_csv(out/'fullpop_mechanism_phase_summary.csv',index=False)
# seed-level paired comparisons and association of late AUC/TPR with final ASR
stats=[]
for a in alphas:
 wide=R[R.alpha.eq(a)].pivot(index='seed',columns='defense',values='asr')
 diff=wide['probeguard_k']-wide['probeguard']
 n=len(diff); half=t.ppf(.975,n-1)*diff.std(ddof=1)/np.sqrt(n)
 from scipy.stats import wilcoxon
 stats.append({'alpha':a,'analysis':'paired_ASR_K_minus_MAD','n':n,'mean':diff.mean(),'ci95_low':diff.mean()-half,'ci95_high':diff.mean()+half,'wilcoxon_p':wilcoxon(diff).pvalue})
 for d in methods:
  sub=Q[(Q.alpha==a)&(Q.defense==d)]
  late=sub[sub['round'].between(16,30)].groupby('seed')[['auc','tpr','fpr','asr']].mean()
  for x in ['auc','tpr','fpr']:
   rho,p=spearmanr(late[x],R[(R.alpha==a)&(R.defense==d)].set_index('seed').loc[late.index,'asr'])
   stats.append({'alpha':a,'analysis':f'{d}_late_{x}_vs_final_asr_spearman','n':len(late),'mean':rho,'ci95_low':np.nan,'ci95_high':np.nan,'wilcoxon_p':p})
pd.DataFrame(stats).to_csv(out/'fullpop_mechanism_statistics.csv',index=False)
# readable report
with open(out/'fullpop_mechanism_report.txt','w') as f:
 f.write('Full-population mechanism analysis, seeds 100-119, alpha .5 and 1.0, fixed trigger, 20% malicious.\n')
 f.write('Each seed/configuration was run with both MAD (k=2, delta=.1) and top-f; n=20 paired seeds per alpha.\n\n')
 f.write('Final ASR mean by alpha/defense:\n'+R.groupby(['alpha','defense']).asr.agg(['mean','std']).to_string()+'\n\n')
 f.write('Phase means and seed-level 95% t intervals are in fullpop_mechanism_phase_summary.csv.\n')
 f.write('Paired tests and seed-level associations are in fullpop_mechanism_statistics.csv.\n')
print('\nFINAL ASR\n',R.groupby(['alpha','defense']).asr.agg(['mean','std']).to_string())
print('\nPHASE SUMMARY\n',pd.DataFrame(S).query("metric in ['auc','tpr','fpr','malicious_top4_fraction','asr']").to_string(index=False))
print('Wrote full-population mechanism analysis; elapsed',round(time.time()-t0,1),'s')
