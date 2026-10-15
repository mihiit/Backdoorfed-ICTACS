import sys, time, csv, shutil
from pathlib import Path
import numpy as np
import pandas as pd
src=Path(__file__).resolve().parent/'paper'/'fl.py'
mod=Path(__file__).resolve().parent/'fl_instrumented.py'
s=src.read_text()
s=s.replace('    traj = []\n    for r in range(ROUNDS):','    traj = []\n    diagnostics = []\n    for r in range(ROUNDS):')
s=s.replace('            else:\n                fl = mad_flag(s, k, delta)\n            if fl.all(): fl[:] = False\n            upd = U[~fl].mean(0)','            else:\n                fl = mad_flag(s, k, delta)\n            if fl.all(): fl[:] = False\n            diagnostics.append({"round": r+1, "scores": s.copy(), "flagged": fl.copy(), "malicious": ft.copy()})\n            upd = U[~fl].mean(0)')
s=s.replace('    if trajectory:\n        out["traj"] = traj','    if trajectory:\n        out["traj"] = traj\n    if defense in ("probeguard", "probeguard+", "probeguard_k", "pg_abs", "pg_local", "pg_knn", "pg_pct90", "pg_pct95", "pg_pct975", "pg_oracle"):\n        out["diagnostics"] = diagnostics')
mod.write_text(s)
sys.path.insert(0,str(mod.parent))
import importlib.util
spec=importlib.util.spec_from_file_location('fl_instrumented',mod)
fl=importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
base=Path(__file__).resolve().parents[1]/'results_rerun'
conditions=[(.5,109),(.5,117),(.5,108),(.5,103),(.5,101),(.5,116),(1.0,100),(1.0,119)]
rows=[]; summary=[]; t0=time.time()
for a,seed in conditions:
  for defense in ['probeguard','probeguard_k']:
    kw={'k':2.0,'delta':0.1} if defense=='probeguard' else {}
    r=fl.run(a,0.2,'fixed',defense,seed,trajectory=True,**kw)
    for d in r['diagnostics']:
      scores=d['scores']; flags=d['flagged']; mal=d['malicious']
      order=np.argsort(-scores)
      rank=np.empty(len(scores),dtype=int); rank[order]=np.arange(1,len(scores)+1)
      for i in range(len(scores)):
        rows.append({'alpha':a,'seed':seed,'defense':defense,'round':d['round'],'client':i,'score':float(scores[i]),'malicious':int(mal[i]),'flagged':int(flags[i]),'score_rank_desc':int(rank[i])})
      malicious_ranks=rank[mal]
      benign_ranks=rank[~mal]
      summary.append({'alpha':a,'seed':seed,'defense':defense,'round':d['round'],
        'malicious_mean_rank':float(malicious_ranks.mean()),'benign_mean_rank':float(benign_ranks.mean()),
        'malicious_top4_fraction':float(np.mean(malicious_ranks<=4)),
        'tpr':float(np.mean(flags[mal])),'fpr':float(np.mean(flags[~mal])),
        'malicious_score_mean':float(scores[mal].mean()),'benign_score_mean':float(scores[~mal].mean()),
        'score_auc':float(np.mean([ (scores[m] > scores[b]) + .5*(scores[m]==scores[b]) for m in np.where(mal)[0] for b in np.where(~mal)[0] ]))})
    print(f'completed alpha={a} seed={seed} defense={defense} ASR={r["asr"]:.3f} TPR={r.get("tpr")} FPR={r.get("fpr")} elapsed={time.time()-t0:.1f}s',flush=True)
pd.DataFrame(rows).to_csv(base/'client_rank_round_level.csv',index=False)
df=pd.DataFrame(summary); df.to_csv(base/'client_rank_round_summary.csv',index=False)
# aggregate by alpha/defense and phase
phase=[]
for (a,defense),g in df.groupby(['alpha','defense']):
 for name,lo,hi in [('early_1_5',1,5),('middle_6_15',6,15),('late_16_30',16,30)]:
  h=g[g['round'].between(lo,hi)]
  phase.append({'alpha':a,'defense':defense,'phase':name,'malicious_top4_fraction_mean':h.malicious_top4_fraction.mean(),'TPR_mean':h.tpr.mean(),'FPR_mean':h.fpr.mean(),'score_AUC_mean':h.score_auc.mean(),'malicious_mean_rank':h.malicious_mean_rank.mean(),'benign_mean_rank':h.benign_mean_rank.mean()})
pd.DataFrame(phase).to_csv(base/'client_rank_phase_summary.csv',index=False)
# plot trajectories ranking auc, tpr, top-4
import matplotlib.pyplot as plt
for a in [.5,1.0]:
 fig,axs=plt.subplots(1,3,figsize=(14,4))
 sub=df[df.alpha.eq(a)]
 for defense,label in [('probeguard','MAD'),('probeguard_k','Top-f')]:
  q=sub[sub.defense.eq(defense)].groupby('round').mean(numeric_only=True)
  axs[0].plot(q.index,q.score_auc,label=label)
  axs[1].plot(q.index,q.tpr,label=label)
  axs[2].plot(q.index,q.malicious_top4_fraction,label=label)
 axs[0].set_title('Per-round score AUROC'); axs[1].set_title('Detection TPR'); axs[2].set_title('Malicious clients in top 4 scores')
 for ax in axs: ax.set_xlabel('Round'); ax.grid(alpha=.25); ax.legend()
 axs[0].set_ylim(0,1); axs[1].set_ylim(0,1); axs[2].set_ylim(0,1)
 fig.suptitle(f'Client ranking diagnostics, α={a:g}; selected seeds')
 fig.tight_layout(); fig.savefig(base/f'client_rank_diagnostics_alpha_{str(a).replace(".","p")}.png',dpi=180); plt.close(fig)
print('Saved diagnostic logs, phase summary and plots.')
