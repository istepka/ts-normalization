"""NeuralForecast model construction for supervised conditions."""

from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NBEATS, NHITS, PatchTST
from pytorch_lightning import Callback

MODEL_CLASSES = {"nhits": NHITS, "nbeats": NBEATS, "patchtst": PatchTST}
CONDITIONS = ("sit", "revin")
NORMALIZATION_MODES = ("standard", "causal")


class BestValidationWeights(Callback):
    """Restores the lowest-validation-loss weights when training finishes."""

    def __init__(self):
        self.best_loss = float("inf")
        self.best_state = None

    def on_validation_end(self, trainer, pl_module) -> None:
        loss = float(trainer.callback_metrics["ptl/val_loss"])
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_state = {
                name: value.detach().cpu().clone()
                for name, value in pl_module.state_dict().items()
            }

    def on_train_end(self, trainer, pl_module) -> None:
        if self.best_state is None:
            raise RuntimeError("training finished without a validation checkpoint")
        pl_module.load_state_dict(self.best_state)


class OriginalSpaceMAE(MAE):
    """MAE that inverts NeuralForecast's window scaler before training loss."""

    def __init__(self, scaler):
        super().__init__()
        object.__setattr__(self, "_scaler", scaler)

    def __call__(self, y, y_hat, mask=None, y_insample=None):
        y = self._scaler.inverse_transform(y)
        y_hat = self._scaler.inverse_transform(y_hat)
        if y_insample is not None:
            y_insample = self._scaler.inverse_transform(y_insample)
        return super().__call__(y, y_hat, mask=mask, y_insample=y_insample)


def build_model(
    name: str,
    condition: str,
    horizon: int,
    input_size: int,
    max_steps: int,
    batch_size: int,
    windows_batch_size: int,
    learning_rate: float,
    val_check_steps: int,
    early_stop_patience_steps: int,
    num_lr_decays: int,
    seed: int,
    device: str,
    normalization_mode: str = "standard",
):
    """Builds one NeuralForecast model with a fixed condition."""
    if name not in MODEL_CLASSES:
        raise ValueError(f"model must be one of {tuple(MODEL_CLASSES)}, got {name!r}")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(
            "normalization_mode must be one of "
            f"{NORMALIZATION_MODES}, got {normalization_mode!r}"
        )
    model_kwargs = {"revin": False} if name == "patchtst" else {}
    model = MODEL_CLASSES[name](
        h=horizon,
        input_size=input_size,
        loss=MAE(),
        valid_loss=MAE(),
        max_steps=max_steps,
        batch_size=batch_size,
        windows_batch_size=windows_batch_size,
        inference_windows_batch_size=windows_batch_size,
        start_padding_enabled=False,
        learning_rate=learning_rate,
        val_check_steps=val_check_steps,
        early_stop_patience_steps=early_stop_patience_steps,
        num_lr_decays=num_lr_decays,
        scaler_type=(
            "identity"
            if normalization_mode == "causal"
            else "standard"
            if condition == "sit"
            else "revin"
        ),
        random_seed=seed,
        alias=name,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        enable_checkpointing=False,
        callbacks=[BestValidationWeights()],
        logger=False,
        **model_kwargs,
    )
    if condition == "revin" and normalization_mode == "standard":
        model.loss = OriginalSpaceMAE(model.scaler)
        model.valid_loss = MAE()
    return model
