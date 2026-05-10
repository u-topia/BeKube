import json
import math
from pathlib import Path


def _percentile(values, q):
    if not values:
        raise ValueError("Cannot compute percentile of empty values.")
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1 - weight) + ordered[right] * weight


class SVDDModel:
    """A pure-Python hypersphere detector inspired by SVDD."""

    def __init__(self, nu=0.05, eps=1e-9):
        if not 0 < nu < 1:
            raise ValueError("nu must be in (0, 1).")
        self.name = "svdd"
        self.nu = nu
        self.eps = eps
        self.feature_names = []
        self.mean_ = []
        self.std_ = []
        self.center_ = []
        self.radius_ = 0.0
        self.train_distances_ = []

    def _fit_scaler(self, rows):
        feature_count = len(rows[0])
        means = []
        stds = []

        for feature_idx in range(feature_count):
            column = [row[feature_idx] for row in rows]
            mean_value = sum(column) / len(column)
            variance = sum((value - mean_value) ** 2 for value in column) / len(column)
            std_value = math.sqrt(variance)
            means.append(mean_value)
            stds.append(std_value if std_value > self.eps else 1.0)

        self.mean_ = means
        self.std_ = stds

    def _transform(self, rows):
        return [
            [
                (row[feature_idx] - self.mean_[feature_idx]) / self.std_[feature_idx]
                for feature_idx in range(len(row))
            ]
            for row in rows
        ]

    @staticmethod
    def _center(rows):
        feature_count = len(rows[0])
        return [
            sum(row[feature_idx] for row in rows) / len(rows)
            for feature_idx in range(feature_count)
        ]

    @staticmethod
    def _distance(row, center):
        return math.sqrt(
            sum((row[feature_idx] - center[feature_idx]) ** 2 for feature_idx in range(len(row)))
        )

    def fit(self, rows, feature_names=None):
        if not rows:
            raise ValueError("Training data is empty.")

        self.feature_names = list(feature_names or [])
        self._fit_scaler(rows)
        scaled_rows = self._transform(rows)
        self.center_ = self._center(scaled_rows)
        self.train_distances_ = [self._distance(row, self.center_) for row in scaled_rows]
        self.radius_ = _percentile(self.train_distances_, 1.0 - self.nu)
        return self

    def decision_function(self, rows):
        scaled_rows = self._transform(rows)
        distances = [self._distance(row, self.center_) for row in scaled_rows]
        return [self.radius_ - distance for distance in distances]

    def predict(self, rows):
        scores = self.decision_function(rows)
        return [1 if score < 0 else 0 for score in scores]

    def to_dict(self):
        return {
            "detector": self.name,
            "nu": self.nu,
            "feature_names": self.feature_names,
            "mean": self.mean_,
            "std": self.std_,
            "center": self.center_,
            "radius": self.radius_,
            "train_distance_min": min(self.train_distances_) if self.train_distances_ else 0.0,
            "train_distance_max": max(self.train_distances_) if self.train_distances_ else 0.0,
        }

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
