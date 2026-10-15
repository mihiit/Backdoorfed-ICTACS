import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pathlib; R=str(pathlib.Path(__file__).resolve().parents[1])+'/'
A=R+'experiments/completed/'
P=R+'paper/'
plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':8,'legend.fontsize':7,
 'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':0.6,
 'lines.linewidth':1.1,'lines.markersize':3.5,'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.4})
rng=np.random.default_rng(20261012)
def boot(x,n=10000):
    x=np.asarray(x); m=rng.choice(x,(n,len(x))).mean(1); return np.percentile(m,[2.5,97.5])
d=pd.read_csv(A+'alpha_sweep_raw.csv')
col={'fedavg':'#555555','probeguard':'#D55E00','probeguard_k':'#CC3399'}
lab={'fedavg':'FedAvg','probeguard':'MAD rule','probeguard_k':'Top-$f$ rule'}
mk={'fedavg':'s','probeguard':'o','probeguard_k':'^'}
ls={'fedavg':'-','probeguard':'-','probeguard_k':'--'}
al=sorted(d.alpha.unique())
# summary table with bootstrap CIs
rows=[]
for a in al:
    for m in ['fedavg','probeguard','probeguard_k']:
        x=d[(d.alpha==a)&(d.defense==m)]
        lo,hi=boot(x.asr)
        rows.append(dict(alpha=a,defense=m,asr=x.asr.mean()*100,lo=lo*100,hi=hi*100,acc=x.clean_accuracy.mean()*100,
            tpr=x.cumulative_tpr.mean(),fpr=x.cumulative_fpr.mean()))
S=pd.DataFrame(rows); S.to_csv(R+'paper_ieee/tableI_20seed_summary.csv',index=False)
print(S.round(1).to_string())
# Fig 2
fig,ax=plt.subplots(1,2,figsize=(7.1,2.3))
for m in ['fedavg','probeguard','probeguard_k']:
    s=S[S.defense==m]
    ax[0].plot(s.alpha,s.asr,color=col[m],marker=mk[m],ls=ls[m],label=lab[m])
    ax[0].fill_between(s.alpha,s.lo,s.hi,color=col[m],alpha=0.13,lw=0)
ax[0].set_xscale('log'); ax[0].set_xticks(al); ax[0].set_xticklabels([str(a).rstrip('0').rstrip('.') for a in al])
ax[0].minorticks_off(); ax[0].set_xlabel(r'Dirichlet concentration $\alpha$ (larger = more IID)'); ax[0].set_ylabel('Attack success rate (%)')
ax[0].set_title('(a) Final-round ASR, 20 seeds',fontsize=8); ax[0].legend(loc='lower left',frameon=False); ax[0].set_ylim(-3,100)
for m in ['probeguard','probeguard_k']:
    s=S[S.defense==m]
    ax[1].plot(s.alpha,s.tpr,color=col[m],marker=mk[m],ls='-',label=lab[m]+' TPR')
    ax[1].plot(s.alpha,s.fpr,color=col[m],marker=mk[m],ls=':',mfc='white',label=lab[m]+' FPR')
ax[1].set_xscale('log'); ax[1].set_xticks(al); ax[1].set_xticklabels([str(a).rstrip('0').rstrip('.') for a in al]); ax[1].minorticks_off()
ax[1].set_xlabel(r'Dirichlet concentration $\alpha$ (larger = more IID)'); ax[1].set_ylabel('Cumulative client-level rate')
ax[1].set_title('(b) Detection rates of the two rules',fontsize=8); ax[1].legend(loc='center right',frameon=False,ncol=1,fontsize=6.5); ax[1].set_ylim(-0.02,1)
plt.tight_layout(); plt.savefig(R+'paper_ieee/figs/fig_alpha.pdf'); plt.close()
# Fig 3
dB=json.load(open(P+'deltaB_extended.json'))
fr=pd.read_csv(A+'fullpop_mechanism_round_level.csv')
late=fr[(fr.defense=='probeguard')&(fr['round']>=16)].groupby(['alpha','seed']).auc.mean().rename('auc')
fin=fr[(fr.defense=='probeguard')&(fr['round']==fr['round'].max())].groupby(['alpha','seed']).asr.mean().rename('asr')
J=pd.concat([late,fin],axis=1).reset_index(); J.to_csv(R+'paper_ieee/fig3b_points.csv',index=False)
_raw=pd.read_csv(A+'f_sensitivity_raw.csv')
_raw['defense']=_raw.defense.map({'probeguard':'MAD','probeguard_k':'Top-f'}); _raw=_raw.rename(columns={'f_assumed':'assumed_f'})
fs=_raw.groupby(['alpha','defense','assumed_f'],dropna=False).asr.mean().rename('final_ASR_mean').reset_index()
fig,ax=plt.subplots(1,3,figsize=(7.1,2.25))
sp=[dB['spread'][str(a)] for a in dB['alphas']]; tp=[dB['tpr'][str(a)] for a in dB['alphas']]
ax[0].scatter(sp,tp,s=14,color='#D55E00',zorder=3)
for a,x,y in zip(dB['alphas'],sp,tp):
    if a in (0.1,0.5,1.0,2.0,10.0): ax[0].annotate(r'$\alpha$='+str(a).rstrip('0').rstrip('.'),(x,y),xytext=(4,3),textcoords='offset points',fontsize=6)
ax[0].set_ylim(-0.03,1.08); ax[0].set_xlabel("Mean benign score spread (SD)"); ax[0].set_ylabel('MAD-rule TPR'); ax[0].set_title(r'(a) Spread vs. TPR ($r{=}{-}0.81$, $n{=}9$)',fontsize=8)
for a,c,m in [(0.5,'#0072B2','o'),(1.0,'#009E73','s')]:
    j=J[J.alpha==a]; rho,p=spearmanr(j.auc,j.asr); print('spearman',a,rho,p,len(j))
    ax[1].scatter(j.auc,j.asr*100,s=14,color=c,marker=m,label=fr'$\alpha$={a} ($\rho$={rho:.2f})',zorder=3)
ax[1].set_xlabel('Late-round score AUROC (rounds 16--30)'); ax[1].set_ylabel('Final ASR, MAD rule (%)'); ax[1].legend(frameon=False,loc='lower left'); ax[1].set_title('(b) Per-seed, closed loop',fontsize=8)
for a,c,m in [(0.5,'#0072B2','o'),(1.0,'#009E73','s')]:
    t=fs[(fs.alpha==a)&(fs.defense=='Top-f')].sort_values('assumed_f')
    ax[2].plot(t.assumed_f,t.final_ASR_mean*100,color=c,marker=m,label=fr'Top-$f$, $\alpha$={a}')
    mad=fs[(fs.alpha==a)&(fs.defense=='MAD')].final_ASR_mean.iloc[0]*100
    ax[2].axhline(mad,color=c,ls='--',lw=0.8)
ax[2].axvline(4,color='k',ls=':',lw=0.8); ax[2].text(4.08,92,'true $f$=4',fontsize=6)
ax[2].set_xlabel(r'Assumed $f$'); ax[2].set_ylabel('Final ASR (%)'); ax[2].set_ylim(0,100); ax[2].set_title('(c) Misspecified $f$ (dashed: MAD)',fontsize=8); ax[2].legend(frameon=False,loc='center right',fontsize=6.5)
plt.tight_layout(); plt.savefig(R+'paper_ieee/figs/fig_mech.pdf'); plt.close()
