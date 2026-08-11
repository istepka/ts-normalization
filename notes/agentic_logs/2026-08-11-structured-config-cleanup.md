# Structured configuration cleanup

The Hydra YAML files remain the experiment record, while structured OmegaConf
schemas now validate the toy and TSFM runtime configurations at their entrypoints.
The four TSFM configurations share corpus, window-index, training, and W&B
defaults through `conf/tsfm_base.yaml`.

The toy configuration no longer contains `seed`, which was always overwritten by
the `seeds` loop, or `modes`, which no runtime code reads. Model architecture
blocks remain in their model-specific YAML files because they are reproducibility
inputs and are converted into adapter dataclasses by the training code.
