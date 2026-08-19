"""All-history causal normalization for NeuralForecast backbones."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from neuralforecast.losses.pytorch import MAE
from torch import nn

from src.models.normalization import (
    SIT,
    PopulationStdScheme,
    RevIN,
    TransformStats,
)
from src.supervised.data import SupervisedSeries, SupervisedSplit


@dataclass(frozen=True)
class CausalWindowSet:
    """Training windows with statistics from history through each context."""

    contexts: torch.Tensor
    targets: torch.Tensor
    locations: torch.Tensor
    scales: torch.Tensor
    series_offsets: torch.Tensor


def _prefix_statistics(
    values: np.ndarray, ends: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Returns population mean and std for each prefix ending in ``ends``."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    ends = np.asarray(ends, dtype=np.int64)
    if np.any(ends < 1) or np.any(ends > len(values)):
        raise ValueError("prefix ends must be between one and the series length")
    prefix_sum = np.concatenate(([0.0], np.cumsum(values)))
    prefix_square_sum = np.concatenate(([0.0], np.cumsum(values * values)))
    count = ends.astype(np.float64)
    locations = prefix_sum[ends] / count
    variance = prefix_square_sum[ends] / count - locations * locations
    scales = np.sqrt(np.maximum(variance, 0.0)) + eps
    return locations.astype(np.float32), scales.astype(np.float32)


def causal_training_windows(
    splits: list[SupervisedSplit], input_size: int, horizon: int
) -> CausalWindowSet:
    """Builds windows whose statistics use all observed values through context."""
    if input_size < 1 or horizon < 1:
        raise ValueError("input_size and horizon must be positive")

    contexts = []
    targets = []
    locations = []
    scales = []
    series_offsets = [0]
    for split in splits:
        values = split.train
        ends = np.arange(input_size, len(values) - horizon + 1)
        if len(ends) == 0:
            raise ValueError(f"{split.item.unique_id} has no complete causal windows")
        item_locations, item_scales = _prefix_statistics(values, ends)
        for end, location, scale in zip(ends, item_locations, item_scales):
            contexts.append(values[end - input_size : end])
            targets.append(values[end : end + horizon])
            locations.append(location)
            scales.append(scale)
        series_offsets.append(len(contexts))
    if not contexts:
        raise ValueError(
            "the supervised training split has no complete causal windows; "
            "reduce input_size or horizon"
        )
    return CausalWindowSet(
        contexts=torch.tensor(np.stack(contexts), dtype=torch.float32),
        targets=torch.tensor(np.stack(targets), dtype=torch.float32),
        locations=torch.tensor(locations, dtype=torch.float32),
        scales=torch.tensor(scales, dtype=torch.float32),
        series_offsets=torch.tensor(series_offsets, dtype=torch.int64),
    )


def _stats(
    locations: torch.Tensor,
    scales: torch.Tensor,
    scheme: PopulationStdScheme,
) -> TransformStats:
    return TransformStats(
        loc=locations,
        scale=scales,
        degenerate=scales <= scheme.scale_floor * (1.0 + 1e-6),
        scheme=scheme,
    )


def _module_for_condition(condition: str, scheme: PopulationStdScheme):
    if condition == "sit":
        return SIT(scheme)
    if condition == "revin":
        return RevIN(scheme)
    raise ValueError(f"condition must be one of ('sit', 'revin'), got {condition!r}")


@dataclass
class CausalForecaster:
    """An identity-scaled NeuralForecast backbone with causal preprocessing."""

    model: nn.Module
    condition: str
    input_size: int
    horizon: int
    device: str
    scheme: PopulationStdScheme

    def predict(
        self,
        items: list[SupervisedSeries],
        histories: list[np.ndarray],
        freq: str,
        horizon: int,
    ) -> dict[str, np.ndarray]:
        """Forecasts each series using all values available at its origin."""
        del freq
        if len(items) != len(histories):
            raise ValueError(f"{len(items)} items for {len(histories)} histories")
        if horizon != self.horizon:
            raise ValueError(f"expected horizon {self.horizon}, got {horizon}")
        if any(len(history) < self.input_size for history in histories):
            raise ValueError("every prediction history must contain input_size values")

        device = torch.device(self.device)
        contexts = torch.tensor(
            np.stack([history[-self.input_size :] for history in histories]),
            dtype=torch.float32,
            device=device,
        )
        locations = []
        scales = []
        for history in histories:
            location, scale = _prefix_statistics(
                np.asarray(history), np.array([len(history)])
            )
            locations.append(location[0])
            scales.append(scale[0])
        stats = _stats(
            torch.tensor(locations, dtype=torch.float32, device=device),
            torch.tensor(scales, dtype=torch.float32, device=device),
            self.scheme,
        )
        normalized = stats.forward(contexts.unsqueeze(-1))
        batch = {
            "insample_y": normalized,
            "insample_mask": torch.ones_like(normalized),
            "futr_exog": None,
            "hist_exog": None,
            "stat_exog": None,
        }
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch)
            predictions = stats.inverse(output).squeeze(-1).cpu().numpy()
        return {
            item.unique_id: prediction
            for item, prediction in zip(items, predictions, strict=True)
        }

    def save(self, path: Path) -> None:
        """Saves model weights and the causal preprocessing metadata."""
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "condition": self.condition,
                "input_size": self.input_size,
                "horizon": self.horizon,
                "normalization": "causal",
                "scaler_type": "identity",
            },
            path,
        )


def train_causal(
    model: nn.Module,
    splits: list[SupervisedSplit],
    input_size: int,
    horizon: int,
    condition: str,
    max_steps: int,
    batch_size: int,
    windows_batch_size: int,
    learning_rate: float,
    val_check_steps: int,
    early_stop_patience_steps: int,
    num_lr_decays: int,
    seed: int,
    device: str,
) -> CausalForecaster:
    """Trains an identity-scaled NeuralForecast model on causal windows."""
    if (
        max_steps < 1
        or batch_size < 1
        or windows_batch_size < 1
        or learning_rate <= 0
        or val_check_steps < 1
        or early_stop_patience_steps == 0
        or num_lr_decays < 0
    ):
        raise ValueError("causal training configuration is invalid")
    windows = causal_training_windows(splits, input_size, horizon)
    scheme = PopulationStdScheme(eps=1e-6)
    normalizer = _module_for_condition(condition, scheme)
    target_device = torch.device(device)
    model.to(target_device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_decay_steps = max(max_steps // num_lr_decays, 1) if num_lr_decays > 0 else 10**8
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_steps, gamma=0.5
    )
    loss_fn = MAE()
    generator = torch.Generator().manual_seed(seed)
    n_series = len(splits)
    best_validation = float("inf")
    best_state = None
    checks_without_improvement = 0
    validation_contexts = torch.tensor(
        np.stack([split.train[-input_size:] for split in splits]),
        dtype=torch.float32,
    ).unsqueeze(-1)
    validation_targets = torch.tensor(
        np.stack([split.validation[:horizon] for split in splits]),
        dtype=torch.float32,
    ).unsqueeze(-1)
    validation_locations = []
    validation_scales = []
    for split in splits:
        location, scale = _prefix_statistics(split.train, np.array([len(split.train)]))
        validation_locations.append(location[0])
        validation_scales.append(scale[0])
    validation_locations = torch.tensor(validation_locations, dtype=torch.float32)
    validation_scales = torch.tensor(validation_scales, dtype=torch.float32)

    for step in range(max_steps):
        if n_series <= batch_size:
            series_indexes = torch.arange(n_series)
        else:
            series_indexes = torch.randperm(n_series, generator=generator)[:batch_size]
        candidates = torch.cat(
            [
                torch.arange(
                    windows.series_offsets[index],
                    windows.series_offsets[index + 1],
                )
                for index in series_indexes
            ]
        )
        if len(candidates) < windows_batch_size:
            indexes = candidates[
                torch.randint(
                    len(candidates),
                    (windows_batch_size,),
                    generator=generator,
                )
            ]
        else:
            indexes = candidates[
                torch.randperm(len(candidates), generator=generator)[
                    :windows_batch_size
                ]
            ]
        contexts = windows.contexts[indexes].to(target_device).unsqueeze(-1)
        targets = windows.targets[indexes].to(target_device).unsqueeze(-1)
        stats = _stats(
            windows.locations[indexes].to(target_device),
            windows.scales[indexes].to(target_device),
            scheme,
        )
        normalized_context = stats.forward(contexts)
        batch = {
            "insample_y": normalized_context,
            "insample_mask": torch.ones_like(normalized_context),
            "futr_exog": None,
            "hist_exog": None,
            "stat_exog": None,
        }
        output = model(batch)
        output, target = normalizer.transform_target_and_output(output, targets, stats)
        loss = loss_fn(y=target, y_hat=output, mask=torch.ones_like(target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.train_trajectories.append((step + 1, loss.detach().item()))

        if (step + 1) % min(val_check_steps, max_steps) == 0:
            model.eval()
            validation_loss_sum = 0.0
            with torch.no_grad():
                for start in range(0, n_series, windows_batch_size):
                    stop = min(start + windows_batch_size, n_series)
                    batch_contexts = validation_contexts[start:stop].to(target_device)
                    batch_targets = validation_targets[start:stop].to(target_device)
                    validation_stats = _stats(
                        validation_locations[start:stop].to(target_device),
                        validation_scales[start:stop].to(target_device),
                        scheme,
                    )
                    normalized_validation = validation_stats.forward(batch_contexts)
                    validation_batch = {
                        "insample_y": normalized_validation,
                        "insample_mask": torch.ones_like(normalized_validation),
                        "futr_exog": None,
                        "hist_exog": None,
                        "stat_exog": None,
                    }
                    validation_output = validation_stats.inverse(
                        model(validation_batch)
                    )
                    batch_loss = loss_fn(
                        y=batch_targets,
                        y_hat=validation_output,
                        mask=torch.ones_like(batch_targets),
                    ).item()
                    validation_loss_sum += batch_loss * (stop - start)
            validation_loss = validation_loss_sum / n_series
            model.valid_trajectories.append((step + 1, validation_loss))
            model.train()
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            if (
                early_stop_patience_steps > 0
                and checks_without_improvement >= early_stop_patience_steps
            ):
                break

    if best_state is None:
        raise RuntimeError("causal training finished without a validation checkpoint")
    model.load_state_dict(best_state)

    return CausalForecaster(
        model=model,
        condition=condition,
        input_size=input_size,
        horizon=horizon,
        device=device,
        scheme=scheme,
    )
