# Natural-scale TSFM continuation to 60k updates

## Purpose

Extend the four-model natural-scale loss-space comparison from 30,000 to 60,000 optimizer attempts. The experiment tests whether the normalized-space accuracy advantage persists, narrows, or reverses with longer training.

## Runs

The launcher continues 32 completed runs. These are four models, two loss spaces, and four seeds. It does not continue the controlled-scale conditions.

Each task loads `checkpoint_step30000.pt` from the completed run. The checkpoint restores model weights and AdamW state. The source `history.json`, exposure counts, and optimization counts are copied into the new run record. Training begins at step 30,001 and ends at step 60,000. New outputs are written under a separate dated directory.

All runs use batch size 512. Evaluation runs every 250 steps. Checkpoints are saved every 6,000 steps. The Slurm array permits at most eight concurrent GPU jobs.

## Continuation limitation

The 30k checkpoints do not contain PyTorch random-number-generator state. The batch schedule generator also precomputes a requested schedule length. Therefore, the continuation is not bitwise equivalent to one uninterrupted 60k process.

For each seed, all matched loss conditions and all four models use the same continuation schedule. This preserves the comparison needed by the experiment. The saved model and optimizer states preserve the learned state at 30k.

## Launch

Validate inputs without submission.

```bash
scripts/submit_tsfm_natural_continue60k.sh --check-only
```

Submit through Rhea.

```bash
scripts/submit_tsfm_natural_continue60k.sh
```

The wrapper uses SSH only for `sbatch`. It does not use Slurm `--export`.
