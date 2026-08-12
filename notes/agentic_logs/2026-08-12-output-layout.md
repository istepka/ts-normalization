# Output layout

The output tree had two competing conventions. Older runs used job-ID-prefixed
directories, while Hydra created separate date and time directories containing
metadata and logs. New producers use

```text
outputs/YYYY-MM-DD/<category>/<experiment>/<run>/
```

The categories are `experiments`, `analysis`, `visualizations`, `data`, and
`diagnostics`. Training artifacts stay grouped by experiment and run. Derived
reports have a separate analysis path. Replots and figures retain the source
experiment or run name so they are traceable.

The shared shell helper is `scripts/output_paths.sh`. The migration utility is
`scripts/organize_outputs.py`. It is dry-run by default, refuses destination
collisions, and records applied moves in
`outputs/organization_manifest.json`.

Live paths can be preserved with repeated `--exclude NAME` arguments. The
canonical window index was excluded from the first migration preview because
the active batch scripts still read its old location.

The migration must exclude output directories that are still being written by a
Slurm job. On 2026-08-12, Chronos-2 array `29436` was running and Moirai-2 array
`29437` was pending, so current-date outputs were left untouched during the
initial migration pass.
