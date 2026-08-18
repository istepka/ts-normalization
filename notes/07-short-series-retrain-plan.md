# Variable-context pretraining with M-series train splits

Status: plan, not started. Written 2026-08-17. Work begins once the Moirai
2.0 natural-scale runs (job 32462) and their eval chain finish.

Depends on the motivation and measurements in
[06-variable-context-windows.md](06-variable-context-windows.md).

## Why the two changes are one piece of work

They are ordered, not parallel. At context 512 plus horizon 128 a window
needs 640 valid points, and M1 tops out at 132, M3 at 126, Tourism at 309.
**Adding the M-series train splits to the corpus before the window geometry
changes adds exactly zero windows.** The variable-context change has to land
first or the second change is invisible.

## Split convention

Taken from the supervised worktree at
`/zfsauton/scratch/istepka/tmp/worktrees/m-series-supervised`, branch
`feat/m-series-supervised`, so both experiments cut every series the same way.
`src/supervised/data.py` defines it:

- **test** is the final `2H - 1` observations, where `H` is that series' own
  official competition horizon.
- **validation** is the `validation_size` observations before the test tail,
  which in a pooled run is the frequency's common model horizon.
- **train** is everything before validation, that is `values[:validation_start]`.

Pretraining takes the `train` region only. Reserving validation as well costs
little and keeps the TSFM and the supervised models honest against the same
held-out region, so neither has seen what the other is scored on.

The eval harness cuts only the final official `H`, so a train region ending
at `len - (2H - 1) - validation_size` sits strictly inside what the harness
already treats as conditioning history. No evaluated horizon is touched.

## Todo

### Phase 1, variable-context windows

1. **Size the corpus at candidate geometries.** A CPU job that walks all 152
   dataset directories once and reports series and window counts at 512+128,
   256+64, 128+32, and 64+8.
   *Verify: the report accounts for all 152 directories and reproduces the
   known 40,952,582 windows over 53 datasets at 512+128.*

2. **Decide the horizon rule.** Fixed 8 for short windows, or proportional to
   context. A 64-point context with a 128-point horizon is not learnable, and
   TimesFM's `prediction_length` must stay 128 to match every TimesFM config's
   `horizon_len`, while Moirai 2.0's binding native horizon is already 64.
   *Verify: the rule is written down here and each of the three adapters can
   express it, or is explicitly scoped out.*

3. **Emit per-window geometry from the index builder.** `WindowIndex` carries
   context and horizon length per row rather than one fixed pair.
   *Verify: an index built at a single fixed geometry is row-identical to the
   current one.*

4. **Ragged batching.** Pad to the batch maximum and mask, which the adapters
   already accept, or bucket by length. Pick one and record why.
   *Verify: a batch mixing 64 and 512-point windows produces the same
   per-example loss as those windows run separately.*

5. **Sampling rule.** Short windows will vastly outnumber long ones once the
   99 currently absent datasets appear. Left alone they dominate every batch.
   *Verify: the realized length distribution over a full epoch is recorded in
   `summary.json` for the run.*

### Phase 2, M-series train splits in the corpus

6. **Extend the supervised loader to Favorita.** `load_series` currently takes
   m1, m3, m4, and tourism only, and Favorita is wanted too.
   *Verify: Favorita loads at 83,207 eligible series with its 16-step horizon,
   matching the eval harness count.*

7. **Emit the train regions as a corpus dataset.** One directory per suite in
   the same Arrow layout `discover_dataset_dirs` reads, so the window builder
   needs no special case.
   *Verify: every emitted series is a strict prefix of the canonical series
   and ends at or before `len - (2H - 1) - validation_size`.*

8. **Remove those suites from `corpus.exclude`, for this run only.**
   `resolve_window_index` raises when a cached index holds an excluded
   dataset, so the exclusion list and the index must move together.
   *Verify: a leakage audit shows no emitted series overlaps any evaluated
   horizon, run the same way as the 2026-08-14 fingerprint audit.*

### Phase 3, retrain and re-evaluate

9. **Rebuild the index** at the chosen geometry with the M-series included.
   *Verify: window count and dataset count recorded, and the count of
   datasets that changed `_dataset_scale_group` noted, since that split at
   `len // 2` moves when the dataset set changes.*

10. **Retrain Chronos-2 and Moirai-2.0**, natural scale, 4 seeds, both
    conditions. Same launcher, new index.
    *Verify: 8 runs per model reach the step target and write
    `checkpoint_step<STEPS>.pt` and `summary.json`.*

11. **Re-run the eval chain and the reference baselines.**
    *Verify: the reports regenerate, and the short-series suites are compared
    against the current numbers below.*

## What this is expected to change

The current Chronos-2 result, native MASE, 4 seeds:

| suite | SIT | RevIN | seasonal naive |
|---|---|---|---|
| favorita | **0.838** | 0.889 | 1.079 |
| gifteval | **1.603** | 2.588 | 1.824 |
| m1 | 2.304 | 3.063 | **2.117** |
| m3 | 1.859 | 5.463 | **1.764** |
| m4 | 2.186 | 4.463 | **2.057** |
| tourism | 3.260 | 4.052 | **2.412** |

The pretrained model beats seasonal naive only where series are long. The
prediction is that variable context closes most of the M1, M3, M4, and
Tourism gap, and that the M-series train splits close the rest. If phase 1
alone closes it, phase 2 was not the cause and that is worth knowing.

## What this costs

**These runs are not zero-shot on M1, M3, M4, Tourism, or Favorita.** Once
their train regions are in the corpus, those suites are in-domain. The
existing clean-corpus runs (jobs 32459 and 32462) stay as the zero-shot table
that `notes/PLAN.md` asks for. Two sets of pretrained checkpoints therefore
have to coexist and be labelled, and neither the eval reports nor the
artifact currently carry a field distinguishing them.

Controlled-scale results cannot be pooled across the index rebuild, since
`_dataset_scale_group` reassigns datasets when the corpus set changes. Only
natural-scale runs are in scope here, so this is recorded rather than blocking.

## Code duplication

The supervised worktree will land as its own PR. Duplicating the split
convention between `src/supervised/data.py` and whatever emits the corpus
regions is acceptable for now, by explicit decision. The shared definition is
the `2H - 1` test tail and the validation region before it, and if the two
ever disagree the comparison between the supervised and pretrained tables
stops meaning anything, so the duplication is worth a test that pins them
together.
