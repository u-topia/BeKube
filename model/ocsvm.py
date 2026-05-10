import json
import math
from pathlib import Path


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _dot(left, right):
    return sum(left[idx] * right[idx] for idx in range(len(left)))


def _squared_distance(left, right):
    return sum((left[idx] - right[idx]) ** 2 for idx in range(len(left)))


def _project_capped_simplex(values, cap, target_sum=1.0, iterations=80):
    lower = min(values) - cap
    upper = max(values)
    for _ in range(iterations):
        middle = (lower + upper) / 2.0
        projected_sum = sum(_clamp(value - middle, 0.0, cap) for value in values)
        if projected_sum > target_sum:
            lower = middle
        else:
            upper = middle
    theta = (lower + upper) / 2.0
    return [_clamp(value - theta, 0.0, cap) for value in values]


class OneClassSVMModel:
    """A pure-Python RBF one-class SVM solved by projected gradient descent."""

    def __init__(self, nu=0.05, gamma=0.0, max_iter=200, tol=1e-6, eps=1e-9):
        if not 0 < nu < 1:
            raise ValueError("nu must be in (0, 1).")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")
        self.name = "ocsvm"
        self.nu = nu
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.feature_names = []
        self.mean_ = []
        self.std_ = []
        self.support_vectors_ = []
        self.alphas_ = []
        self.rho_ = 0.0
        self.effective_gamma_ = 0.0
        self.objective_ = 0.0

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

    def _resolve_gamma(self, rows):
        if self.gamma > 0:
            self.effective_gamma_ = self.gamma
            return
        feature_count = len(rows[0])
        variance = sum(_dot(row, row) for row in rows) / max(1, len(rows) * feature_count)
        scale = feature_count * variance if variance > self.eps else feature_count
        self.effective_gamma_ = 1.0 / max(scale, self.eps)

    def _rbf_kernel(self, left, right):
        return math.exp(-self.effective_gamma_ * _squared_distance(left, right))

    def _kernel_matrix(self, rows):
        size = len(rows)
        kernel = [[0.0 for _ in range(size)] for _ in range(size)]
        for left_idx in range(size):
            kernel[left_idx][left_idx] = 1.0
            for right_idx in range(left_idx + 1, size):
                value = self._rbf_kernel(rows[left_idx], rows[right_idx])
                kernel[left_idx][right_idx] = value
                kernel[right_idx][left_idx] = value
        return kernel

    @staticmethod
    def _matrix_vector(matrix, vector):
        return [sum(row[col_idx] * vector[col_idx] for col_idx in range(len(vector))) for row in matrix]

    @staticmethod
    def _objective(kernel_matrix, alphas):
        gradient = OneClassSVMModel._matrix_vector(kernel_matrix, alphas)
        return 0.5 * _dot(alphas, gradient)

    def _fit_dual(self, kernel_matrix):
        sample_count = len(kernel_matrix)
        cap = 1.0 / (self.nu * sample_count)
        alphas = [1.0 / sample_count for _ in range(sample_count)]
        lipschitz = max(sum(abs(value) for value in row) for row in kernel_matrix)
        step_size = 1.0 / max(lipschitz, 1.0)

        for _ in range(self.max_iter):
            gradient = self._matrix_vector(kernel_matrix, alphas)
            candidate = [alphas[idx] - step_size * gradient[idx] for idx in range(sample_count)]
            projected = _project_capped_simplex(candidate, cap=cap)
            change = math.sqrt(sum((projected[idx] - alphas[idx]) ** 2 for idx in range(sample_count)))
            alphas = projected
            if change < self.tol:
                break

        self.objective_ = self._objective(kernel_matrix, alphas)
        return alphas, cap

    def fit(self, rows, feature_names=None):
        if not rows:
            raise ValueError("Training data is empty.")

        self.feature_names = list(feature_names or [])
        self._fit_scaler(rows)
        scaled_rows = self._transform(rows)
        self._resolve_gamma(scaled_rows)
        kernel_matrix = self._kernel_matrix(scaled_rows)
        alphas, cap = self._fit_dual(kernel_matrix)

        support_indices = [idx for idx, alpha in enumerate(alphas) if alpha > self.eps]
        self.support_vectors_ = [scaled_rows[idx] for idx in support_indices]
        self.alphas_ = [alphas[idx] for idx in support_indices]

        margin_indices = [idx for idx, alpha in enumerate(alphas) if self.eps < alpha < cap - self.eps]
        reference_indices = margin_indices or support_indices or list(range(len(scaled_rows)))
        rho_values = []
        for ref_idx in reference_indices:
            rho_values.append(sum(alphas[col_idx] * kernel_matrix[col_idx][ref_idx] for col_idx in range(len(alphas))))
        self.rho_ = sum(rho_values) / len(rho_values)
        return self

    def decision_function(self, rows):
        scaled_rows = self._transform(rows)
        scores = []
        for row in scaled_rows:
            kernel_sum = sum(alpha * self._rbf_kernel(row, support_vector) for alpha, support_vector in zip(self.alphas_, self.support_vectors_))
            scores.append(kernel_sum - self.rho_)
        return scores

    def predict(self, rows):
        scores = self.decision_function(rows)
        return [1 if score < 0 else 0 for score in scores]

    def to_dict(self):
        return {
            "detector": self.name,
            "nu": self.nu,
            "gamma": self.gamma,
            "effective_gamma": self.effective_gamma_,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "feature_names": self.feature_names,
            "mean": self.mean_,
            "std": self.std_,
            "rho": self.rho_,
            "objective": self.objective_,
            "support_vector_count": len(self.support_vectors_),
            "alphas": self.alphas_,
            "support_vectors": self.support_vectors_,
        }

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
