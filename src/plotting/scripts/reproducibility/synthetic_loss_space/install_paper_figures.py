"""Validate and install the SGD and Adam forecast-evolution paper figures.

Paper: installs Fig. loss_space_comp:synthetic:forecasts (default source runs
are jobs 23555 SGD / 23556 Adam) into
overleaf/figures/loss_space/synthetic_loss_space/{SGD,ADAM}.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

REQUESTED_SNAPSHOT_STEPS = {0, 10, 50, 100, 500, 29999}


def install_figures(sgd_dir: Path, adam_dir: Path, figure_root: Path) -> None:
    runs = {"SGD": sgd_dir, "ADAM": adam_dir}
    for optimizer, run_dir in runs.items():
        for label in ("normalized", "original"):
            data_path = run_dir / f"forecast_data_{label}.npz"
            captured_steps = set(np.load(data_path)["steps"].tolist())
            missing_steps = REQUESTED_SNAPSHOT_STEPS - captured_steps
            if missing_steps:
                raise ValueError(
                    f"{data_path} is missing steps {sorted(missing_steps)}"
                )

            source = run_dir.parent / f"{run_dir.name}_paper"
            source /= f"forecast_evolution_{label}.pdf"
            destination = figure_root / optimizer / source.name
            shutil.copy2(source, destination)
            print(f"installed {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgd-dir", type=Path, required=True)
    parser.add_argument("--adam-dir", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    args = parser.parse_args()
    install_figures(args.sgd_dir, args.adam_dir, args.figure_root)


if __name__ == "__main__":
    main()
