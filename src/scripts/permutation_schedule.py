"""Balanced complementary assignments for the scale-swap permutation campaign.

Paper: generates the 15 complementary assignment pairs behind
apd_loss_space_scale_swap.tex "Balanced assignment".
"""

import argparse
import itertools

import numpy as np

LOW_SCALE = 1.0
HIGH_SCALE = 10.0
NUM_DATASETS = 8
HIGH_GROUPS = [
    (4, 5, 6, 7),
    (0, 1, 2, 5),
    (0, 1, 3, 7),
    (0, 1, 4, 5),
    (0, 1, 4, 6),
    (0, 1, 6, 7),
    (0, 2, 3, 4),
    (0, 2, 4, 7),
    (0, 2, 5, 6),
    (0, 2, 5, 7),
    (0, 2, 6, 7),
    (0, 3, 4, 6),
    (0, 3, 5, 6),
    (0, 3, 5, 7),
    (0, 4, 5, 7),
]


def assignment_pair(pair_index: int) -> tuple[np.ndarray, np.ndarray]:
    high_group = set(HIGH_GROUPS[pair_index - 1])
    assignment_a = np.array(
        [
            HIGH_SCALE if dataset in high_group else LOW_SCALE
            for dataset in range(NUM_DATASETS)
        ]
    )
    assignment_b = np.where(assignment_a == HIGH_SCALE, LOW_SCALE, HIGH_SCALE)
    return assignment_a, assignment_b


def validate_schedule():
    if len(HIGH_GROUPS) != 15 or len(set(HIGH_GROUPS)) != 15:
        raise ValueError("schedule must contain 15 unique complementary pairs")
    if HIGH_GROUPS[0] != (4, 5, 6, 7):
        raise ValueError("pair 1 must match completed LR-adjusted job 18985")

    assignments = []
    for pair_index, high_group in enumerate(HIGH_GROUPS, start=1):
        if len(high_group) != 4 or len(set(high_group)) != 4:
            raise ValueError(f"pair {pair_index} must contain four unique datasets")
        assignment_a, assignment_b = assignment_pair(pair_index)
        assignments.extend((assignment_a, assignment_b))

    stacked = np.stack(assignments)
    high_counts = (stacked == HIGH_SCALE).sum(axis=0)
    if not np.array_equal(high_counts, np.full(NUM_DATASETS, 15)):
        raise ValueError(f"each dataset must be high 15 times, got {high_counts}")

    coassignment_counts = []
    for first, second in itertools.combinations(range(NUM_DATASETS), 2):
        coassignment_counts.append(
            np.logical_and(
                stacked[:, first] == HIGH_SCALE,
                stacked[:, second] == HIGH_SCALE,
            ).sum()
        )
    if set(coassignment_counts) != {6, 7}:
        raise ValueError(
            "dataset pairs must be assigned high together either six or seven times"
        )


def format_assignment(values: np.ndarray) -> str:
    return "[" + ",".join(f"{value:.1f}" for value in values) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-index", type=int, choices=range(1, 16), required=True)
    parser.add_argument("--arm", choices=("a", "b"), required=True)
    args = parser.parse_args()

    validate_schedule()
    assignment_a, assignment_b = assignment_pair(args.pair_index)
    assignment = assignment_a if args.arm == "a" else assignment_b
    print(format_assignment(assignment))


if __name__ == "__main__":
    main()
