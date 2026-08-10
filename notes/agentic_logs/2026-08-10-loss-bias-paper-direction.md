# Loss-bias paper direction

## Assessment

The loss-space bias work could become a strong standalone ICLR paper if the
empirical evidence becomes broader and more causal. The current results are
interesting but preliminary.

The elementary observation that original-space MSE scales with the square of
the normalization scale is not enough for a paper by itself. The stronger claim
is that loss space acts as an implicit dataset-weighting rule during
multi-dataset TSFM pretraining. This weighting can change optimization, final
per-dataset accuracy, and the distribution of errors across datasets. Aggregate
benchmark metrics can conceal these effects.

## Relationship to the causal-normalization work

Splitting the two stories would likely improve the paper's coherence.

The loss-bias story concerns multi-dataset optimization and differences in how
well individual datasets are learned. The causal-normalization story concerns
temporal information, the estimation of normalization statistics, and
inference-time distribution shift. The topics share normalization as a theme,
but they ask different scientific questions.

A split is worthwhile only if the loss-bias paper receives enough evidence to
stand independently. Two incomplete papers would be weaker than one complete
paper.

## Central claim

A useful framing is the following.

> Loss space determines an implicit weighting over datasets during TSFM
> pretraining. Original-space loss makes that weighting depend on dataset
> scale, which can produce unequal convergence and unequal final fit.
> Per-dataset evaluation is needed because aggregate error can hide these
> effects.

The central question is, "Which datasets does a TSFM learn, and how does the
choice of loss space decide?"

## Evidence needed

### Replication across models

The central result should be reproduced across at least two substantially
different TSFM architectures and objectives. TimesFM forecasting and MOMENT
reconstruction provide a starting point. The main comparisons should use four
or five seeds.

### Stronger causal scale tests

The complementary scale assignment is a useful causal design because each
dataset is evaluated at both imposed scales. It should be extended beyond
$b\in\{1,10\}$ to several scale levels. A dose-response pattern would show how
the final dataset preference changes with scale.

An explicit weighting control would be especially informative. Normalized-space
loss with weights derived from the imposed scale should reproduce the behavior
of original-space loss. This would connect the mathematical weighting identity
to the observed optimization and forecasting outcomes.

### Official benchmark evaluation

The models should be evaluated on official GIFT-Eval. The report should include
pooled MASE, per-dataset MASE, the Gini coefficient across datasets, the number
of datasets improved, worst-quartile performance, and seed-level confidence
intervals.

MASE and Gini must be interpreted jointly. A lower Gini is not beneficial when
it results from uniformly worse forecasts.

### A principled intervention

Normalized-space loss cannot yet be recommended universally because it performs
poorly with first-patch normalization. Whole-context normalization is currently
the only tested TimesFM setting that improves average accuracy while lowering
the Gini point estimate.

The next experiments should determine whether normalized-space loss depends on
the target being expressed using representative context statistics. A strong
paper should explain this interaction and provide a practical intervention,
not only document the original-space failure.

## Main reviewer risks

Reviewers may argue that the scale dependence is algebraically obvious, that
the artificial scaling creates the result by construction, or that Gini is only
a descriptive statistic. They may also question whether reduced pretraining
transfers to released TSFMs and whether natural dataset scale can be separated
from difficulty, frequency, sparsity, and noise.

The paper should acknowledge that the gradient identity is expected. Its
contribution is to establish the consequences for realistic multi-dataset TSFM
training, measure the resulting differences in final dataset-level fit, and
provide an intervention that improves accuracy without merely making failure
more uniform.

The observed numerical instability at extreme scales is a secondary practical
consequence rather than a separate contribution. It should remain a brief
observation or appendix result unless it is studied systematically across
datasets and models.
