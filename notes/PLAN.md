# Planning for the experiments that will be the core of the paper 

## Table format: 
- Rows = datasets, columns=models with sub columns ={norm, deform}
- zeroshot table: Chronos2, TimesFM, Moirai 2
- supervised table: NHITS, NBEATS, PatchTST | all from neuralforecast

## Tables:

- Table on supervision: M-comp, tourism. Train cross dataset on the same frequencies
- Table on zero-shot: M-comp, tourism. Train TSFMs on GiftEvalPretrain (clean)
- Statsforecast baselines (ETS, ARIMA) as additional columns for reference (clearly separated)


## Experiments so far:

1. Synthetic data, different learning rates
2. Multiplier per dataset, different learning rates
3. TSFM pretrain on gift-eval (minus M-comp, tourism, and favorita)
4. M-series supervision.
    - Test size = 2H-1, val size = H, train size = all prior time steps preceding validation
    - Train cross dataset (per frequency) on M-comp and Tourism 
    - Heteroscadsity test. Eval in buckets (later, for now just keep scores in a way that will allow this later)



## Later:

1. Hyperparameter tuning
2. Train cross frequency for M-competition
3. MQForecaster for M-series supervision experiments
4. Can we automate selection based on trend/no trend metrics?
5. Cool temporal scaler —> support different lag sizes for statistic computation. Lag size as hyper parameter. Return during hyperparameter tuning?
