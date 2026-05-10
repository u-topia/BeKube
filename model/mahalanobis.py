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


def _identity_matrix(size):
    return [[1.0 if row_idx == col_idx else 0.0 for col_idx in range(size)] for row_idx in range(size)]


def _gauss_jordan_inverse(matrix):
    size = len(matrix)
    augmented = [list(matrix[row_idx]) + _identity_matrix(size)[row_idx] for row_idx in range(size)]

    for pivot_idx in range(size):
        best_row = max(range(pivot_idx, size), key=lambda row_idx: abs(augmented[row_idx][pivot_idx]))
        pivot_value = augmented[best_row][pivot_idx]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Covariance matrix is singular and cannot be inverted.")
        if best_row != pivot_idx:
            augmented[pivot_idx], augmented[best_row] = augmented[best_row], augmented[pivot_idx]

        pivot_value = augmented[pivot_idx][pivot_idx]
        augmented[pivot_idx] = [value / pivot_value for value in augmented[pivot_idx]]

        for row_idx in range(size):
            if row_idx == pivot_idx:
                continue
            factor = augmented[row_idx][pivot_idx]
            if factor == 0:
                continue
            augmented[row_idx] = [
                current - factor * pivot
                for current, pivot in zip(augmented[row_idx], augmented[pivot_idx])
            ]

    return [row[size:] for row in augmented]


class MahalanobisModel:
    """A pure-Python one-class detector based on Mahalanobis distance."""

    def __init__(self, nu=0.05, regularization=1e-6, eps=1e-9):
        if not 0 < nu < 1:
            raise ValueError("nu must be in (0, 1).")
        if regularization <= 0:
            raise ValueError("regularization must be positive.")
        self.name = "mahalanobis"
        self.nu = nu
        self.regularization = regularization
        self.eps = eps
        self.feature_names = []
        self.mean_ = []
        self.std_ = []
        self.center_ = []
        self.covariance_ = []
        self.inverse_covariance_ = []
        self.threshold_ = 0.0
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

    def _covariance(self, rows, center):
        feature_count = len(rows[0])
        covariance = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
        denominator = max(1, len(rows) - 1)

        for row in rows:
            centered = [row[feature_idx] - center[feature_idx] for feature_idx in range(feature_count)]
            for left_idx in range(feature_count):
                for right_idx in range(feature_count):
                    covariance[left_idx][right_idx] += centered[left_idx] * centered[right_idx]

        for left_idx in range(feature_count):
            for right_idx in range(feature_count):
                covariance[left_idx][right_idx] /= denominator
            covariance[left_idx][left_idx] += self.regularization

        return covariance

    @staticmethod
    def _quadratic_form(vector, matrix):
        projected = [sum(matrix[row_idx][col_idx] * vector[col_idx] for col_idx in range(len(vector))) for row_idx in range(len(vector))]
        return sum(vector[row_idx] * projected[row_idx] for row_idx in range(len(vector)))

    def _distance(self, row):
        centered = [row[feature_idx] - self.center_[feature_idx] for feature_idx in range(len(row))]
        distance_sq = max(0.0, self._quadratic_form(centered, self.inverse_covariance_))
        return math.sqrt(distance_sq)

    def fit(self, rows, feature_names=None):
        if not rows:
            raise ValueError("Training data is empty.")

        self.feature_names = list(feature_names or [])
        self._fit_scaler(rows)
        scaled_rows = self._transform(rows)
        self.center_ = self._center(scaled_rows)
        self.covariance_ = self._covariance(scaled_rows, self.center_)
        self.inverse_covariance_ = _gauss_jordan_inverse(self.covariance_)
        self.train_distances_ = [self._distance(row) for row in scaled_rows]
        self.threshold_ = _percentile(self.train_distances_, 1.0 - self.nu)
        return self

    def decision_function(self, rows):
        scaled_rows = self._transform(rows)
        distances = [self._distance(row) for row in scaled_rows]
        return [self.threshold_ - distance for distance in distances]

    def predict(self, rows):
        scores = self.decision_function(rows)
        return [1 if score < 0 else 0 for score in scores]

    def to_dict(self):
        return {
            "detector": self.name,
            "nu": self.nu,
            "regularization": self.regularization,
            "feature_names": self.feature_names,
            "mean": self.mean_,
            "std": self.std_,
            "center": self.center_,
            "covariance": self.covariance_,
            "inverse_covariance": self.inverse_covariance_,
            "threshold": self.threshold_,
            "train_distance_min": min(self.train_distances_) if self.train_distances_ else 0.0,
            "train_distance_max": max(self.train_distances_) if self.train_distances_ else 0.0,
        }

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
