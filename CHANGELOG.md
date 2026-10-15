# Changelog

## v6 (reviewer-response experiments, 6 pages)
- Adaptive attacker re-run on 20 seeds (was 5): top-f ASR at alpha=1 rises 7.3% -> 12.0% under beta=0.5 (+4.8 pts, p=0.0015); Table IV replaced. Old 5-seed Multi-Krum/MAD values were noisy and are superseded.
- New `run_k_sweep_all_alpha.py`: MAD k in {1,1.5,2} vs top-f on all alphas, 20 seeds. k=1 matches top-f from alpha=2 upward without knowing f; gaps +9.4 (alpha=0.5) and +3.5 (alpha=1) points remain. The earlier 10-seed k=1 figure at alpha=0.5 (34.5%) was optimistic; 20-seed value is 43.6%.
- Baselines (Table I) and decision rules (Table III) re-run on the same 20 seeds, removing the Study A vs Study B 8-vs-20-seed discrepancy (top-f at alpha=0.5 is now 34.2% everywhere).


## v5 (final ICTACS 2026 revision, 6 pages)
Paper (`paper_ieee/body.tex`):
- Study A vs Study B: explained that Study A is an earlier 8-seed campaign (runner scripts not preserved); all MAD/top-f claims now rest on Study B. Added a variance argument for the 15.0% vs 34.2% top-f gap at alpha=0.5 (~1.5 standard errors, assuming equal SD) while stating protocol differences cannot be excluded.
- Scope: explicit statements that no claim is made about realistic FL workloads or robustness to adaptive adversaries.
- Overhead: stated linear cost scaling, that the NumPy code is unoptimised, and added an Overhead limitation.
- Maths: Delta_B written out per cell; Proposition 1 proof pointers to the lemma bounds, "round by round" and "Delta_B is proxied by benign-score SD" caveats; Proposition 2 stated per round with ties excluded.
- Reproducibility line now points to this repository; acknowledgment added. Text tightened to hold exactly 6 pages. No numerical result was changed.
Repo: tests re-run (12 passed); CHANGELOG added.
