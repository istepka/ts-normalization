"""Trainer for a single loss-space run.

Every run is constructed from the same seed (identical weight init) and consumes the
same shared batch schedule, so the only difference between a normalized-space and an
original-space run is the loss space. Two controls are supported:

- equal_variance: handled upstream in the dataset (all categories at scale 1).
- grad_norm_match: rescale the full parameter gradient to unit global norm each step,
  removing the global learning-rate inflation so only per-category disparity remains.
"""

import numpy as np
import torch
from omegaconf import DictConfig

from src.data import SyntheticTSDataset
from src.loss import compute_loss, normalize_target, per_sample_nmse
from src.model import PatchTransformer


class Trainer:
    def __init__(
        self, cfg: DictConfig, dataset: SyntheticTSDataset, mode: str, wandb_run
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.mode = mode
        self.wandb_run = wandb_run
        self.device = torch.device(cfg.device)
        self.grad_norm_match = cfg.train.grad_norm_match
        self.num_categories = dataset.num_categories
        self.category_names = dataset.category_names

        self.eval_steps = self._segment_steps(cfg.train.eval_schedule, cfg.train.steps)
        self.snapshot_steps = self._segment_steps(
            cfg.train.forecast_schedule, cfg.train.steps
        )
        self.snapshot_steps.update(
            min(int(c), cfg.train.steps - 1) for c in cfg.train.forecast_columns
        )
        self.probe_context, self.probe_target = self._build_probe(dataset)
        self.val_context = dataset.val_context.to(self.device)
        self.val_target = dataset.val_target.to(self.device)
        self.val_category = dataset.val_category.to(self.device)

        torch.manual_seed(cfg.seed)  # identical init across runs
        self.model = PatchTransformer(cfg).to(self.device)
        optimizers = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
        self.optimizer = optimizers[cfg.train.optimizer](
            self.model.parameters(), lr=cfg.train.lr
        )

    @staticmethod
    def _segment_steps(schedule, total: int) -> set[int]:
        """Step indices from a list of {until, every} segments (dense-early schedule),
        always including the final step."""
        steps, start = set(), 0
        for segment in schedule:
            until = min(segment.until, total)
            steps.update(range(start, until, segment.every))
            start = until
        steps.add(total - 1)
        return steps

    def _build_probe(self, dataset: SyntheticTSDataset):
        """One fixed window per category, evaluated at each snapshot step."""
        ctx, tgt = [], []
        for windows in dataset.windows:
            window = windows[0]
            ctx.append(window[: dataset.context_length])
            tgt.append(window[dataset.context_length :])
        return torch.stack(ctx).to(self.device), torch.stack(tgt)

    def run(self) -> dict:
        history = {
            "step": [],
            "train_loss": [],
            "nmse": [],  # list of [num_categories]
            "global_nmse": [],  # list of scalars (over all categories)
            "grad_mag": [],  # list of [num_categories]
            "forecast_steps": [],  # snapshot steps for qualitative forecasts
            "forecast_pred": [],  # list of [num_categories, horizon], original space
        }
        for step, batch in enumerate(self.dataset.batch_schedule):
            if step in self.snapshot_steps:  # forecast after exactly `step` updates
                history["forecast_steps"].append(step)
                history["forecast_pred"].append(self._probe_forecast())

            context = batch.context.to(self.device)
            target = batch.target.to(self.device)
            category = batch.category.to(self.device)

            z_pred, a, b = self.model(context)
            z_pred.retain_grad()  # to read d loss / d z_pred per category
            z_target = normalize_target(target, a, b)
            loss = compute_loss(self.mode, z_pred, z_target, b)
            if step in self.eval_steps:
                self._require_finite(loss, "training loss", step)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_norm_match:
                self._rescale_to_unit_norm()
            self.optimizer.step()

            if step in self.eval_steps:
                nmse, global_nmse = self._eval_val()
                self._require_finite(nmse, "validation nMSE", step)
                self._require_finite(global_nmse, "global validation nMSE", step)
                grad_per_row = z_pred.grad.detach().pow(2).sum(dim=1).sqrt()
                grad_mag = self._per_category(grad_per_row, category)
                self._require_finite(grad_mag, "gradient magnitude", step)
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["nmse"].append(nmse)
                history["global_nmse"].append(global_nmse)
                history["grad_mag"].append(grad_mag)
                self._log(step, loss.item(), nmse, global_nmse, grad_mag)
        history["probe_context"] = self.probe_context.cpu().numpy()
        history["probe_target"] = self.probe_target.numpy()
        return history

    @staticmethod
    def _require_finite(value, name: str, step: int):
        if isinstance(value, torch.Tensor):
            finite = torch.isfinite(value).all().item()
        else:
            finite = np.isfinite(value).all()
        if not finite:
            raise FloatingPointError(f"non-finite {name} at step {step}")

    def _eval_val(self) -> tuple[np.ndarray, float]:
        """Per-category and global nMSE (normalized space) on the fixed held-out
        validation windows -- a large, low-variance estimate, unlike the 8-window
        training minibatch. Always measured in normalized space so the two loss
        spaces are comparable."""
        self.model.eval()
        with torch.no_grad():
            z_pred, a, b = self.model(self.val_context)
            z_target = normalize_target(self.val_target, a, b)
            sample_nmse = per_sample_nmse(z_pred, z_target)
        self.model.train()
        return self._per_category(
            sample_nmse, self.val_category
        ), sample_nmse.mean().item()

    def _probe_forecast(self) -> np.ndarray:
        """Denormalized horizon prediction on the fixed probe, [num_categories, H]."""
        with torch.no_grad():
            z_pred, a, b = self.model(self.probe_context)
            return (b * z_pred + a).cpu().numpy()

    def _log(self, step, train_loss, nmse, global_nmse, grad_mag):
        metrics = {"train_loss": train_loss, "nmse/global": global_nmse}
        for c, name in enumerate(self.category_names):
            metrics[f"nmse/{name}"] = nmse[c]
            metrics[f"grad_mag/{name}"] = grad_mag[c]
        self.wandb_run.log(metrics, step=step)

    def _per_category(self, values: torch.Tensor, category: torch.Tensor) -> np.ndarray:
        values = values.detach()
        out = np.empty(self.num_categories, dtype=np.float64)
        for c in range(self.num_categories):
            out[c] = values[category == c].mean().item()
        return out

    def _rescale_to_unit_norm(self):
        total = torch.zeros((), device=self.device)
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.pow(2).sum()
        total = total.sqrt()
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.div_(total + 1e-12)
