# Scale-swap permutation campaign

The follow-up expands the LR-adjusted scale-swap crossover from one complementary assignment pair to 15 pairs, or 30 assignment configurations in total.
Pair 1 reuses completed job 18985, while pairs 2--15 run through `scripts/run_scale_swap_permutation_pair.sbatch`.
Every configuration assigns four of the eight datasets to $b=10$ and four to $b=1$, and its complement reverses all eight assignments.
Across the full schedule, every dataset appears at each scale 15 times and every dataset pair appears together in the high-scale group either six or seven times.
This balances both the marginal scale assignments and the identities of the other datasets sharing the high-scale group.

Each Slurm job receives a pair index as its only positional argument and runs both complementary configurations sequentially under the LR-adjusted original-space loss.
The jobs target Rhea's `legacy` partition with `qos_legacy` and write stable outputs under `outputs/scale_swap_permutations/pair_XX_{a,b}`.
Existing outputs cause the launcher to fail rather than overwrite a previous run.

After all pairs finish, `scripts/aggregate_scale_swap_permutations.py` averages seeds within each configuration, computes ordinary nMSE AUC over the complete trajectory, and forms 15 paired $\mathrm{AUC}(b=1)-\mathrm{AUC}(b=10)$ differences per dataset.
It reports a one-sided exact Wilcoxon signed-rank test for each dataset with Holm correction across the eight tests.
It also produces aggregate and per-dataset linear-nMSE trajectories after averaging across all assignments at each scale.
The CPU launcher `scripts/aggregate_scale_swap_permutations.sbatch` is submitted with an `afterok` dependency on every permutation-pair job, so analysis starts only after all 14 new pairs complete successfully.
The completed inference is reproduced in `notebooks/scale_swap_wilcoxon.ipynb` using SciPy's exact one-sided Wilcoxon signed-rank test, Holm correction across the eight datasets, and an exact sign-test sensitivity analysis.

The normalized-space control uses the same complementary assignment schedule and the same base learning rate. Pair 1 is already available in jobs 18948, while pairs 2--15 use `scripts/run_scale_swap_permutation_normalized_pair.sbatch` and write to `outputs/scale_swap_permutations_normalized`. The aggregator can overlay the normalized control with the learning-rate-adjusted original-space trajectories in the eight-panel appendix figure after those jobs finish.
