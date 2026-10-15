import sys,time,csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'paper'))
import fl
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
base=Path(__file__).resolve().parents[1]/'results_rerun'
# Representative failure and success seeds from the fresh-seed run.
conditions=[(.5,109),(.5,117),(.5,108),(.5,103),(.5,101),(.5,116),(1.0,100),(1.0,119)]
rows=[]; t0=time.time()
for a,seed in conditions:
 for defense in ['probeguard','probeguard_k','fedavg','clean_oracle']:
  kw={'k':2.0,'delta':0.1} if defense=='probeguard' else {}
  r=fl.run(a,0.2,'fixed',defense,seed,trajectory=True,**kw)
  for rnd,acc,asr in r['traj']:
   rows.append({'alpha':a,'seed':seed,'defense':defense,'round':rnd+1,'acc':acc,'asr':asr})
  print(f'finished alpha={a}, seed={seed}, defense={defense}; elapsed={time.time()-t0:.1f}s',flush=True)
df=pd.DataFrame(rows); df.to_csv(base/'trajectory_diagnostic.csv',index=False)
# plots each alpha, facet by defense, bad/good seeds
for a in [.5,1.0]:
 sub=df[df.alpha.eq(a)]
 fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
 defenses=['probeguard','probeguard_k','fedavg','clean_oracle']
 labels={'probeguard':'ProbeGuard (MAD)','probeguard_k':'ProbeGuard-K','fedavg':'FedAvg','clean_oracle':'Benign-only oracle'}
 for ax,defense in zip(axs.flat,defenses):
  q=sub[sub.defense.eq(defense)]
  for (seed,g) in q.groupby('seed'):
   tag='failure' if (a==.5 and seed in [109,117,108]) or (a==1 and seed==100) else 'better'
   ax.plot(g['round'],100*g.asr,marker='.',markersize=2,linewidth=1,label=f'{seed} ({tag})')
  ax.set_title(labels[defense]); ax.set_xlabel('Communication round'); ax.set_ylabel('ASR (%)'); ax.grid(True,alpha=.25)
  ax.set_ylim(-2,102)
  if defense=='probeguard_k': ax.legend(fontsize=7,ncol=2)
 fig.suptitle(f'Per-round attack success trajectories, α={a:g}; selected fresh seeds')
 fig.tight_layout(); fig.savefig(base/f'trajectory_alpha_{str(a).replace(".","p")}.png',dpi=180); plt.close(fig)
# early ASR summaries rounds 1-5, 6-15, 16-30
summ=[]
for (a,seed,defense),g in df.groupby(['alpha','seed','defense']):
 for phase,lo,hi in [('early_1_5',1,5),('middle_6_15',6,15),('late_16_30',16,30)]:
  h=g[g['round'].between(lo,hi)]
  summ.append({'alpha':a,'seed':seed,'defense':defense,'phase':phase,'mean_asr_pct':100*h.asr.mean(),'last_round_asr_pct':100*h.asr.iloc[-1]})
pd.DataFrame(summ).to_csv(base/'trajectory_phase_summary.csv',index=False)
print('Saved trajectory diagnostic CSV and plots.')
