import math
import numpy as np
from typing import Dict, List, Tuple, Union
from .matrix import Matrix


class Statistics:
    @staticmethod
    def mean(data: Union[List[float], np.ndarray]) -> float:
        return float(np.mean(data))

    @staticmethod
    def median(data: Union[List[float], np.ndarray]) -> float:
        return float(np.median(data))

    @staticmethod
    def mode(data: Union[List[float], np.ndarray]) -> List[float]:
        values, counts = np.unique(data, return_counts=True)
        max_count = counts.max()
        return [float(v) for v in values[counts == max_count]]

    @staticmethod
    def variance(data: Union[List[float], np.ndarray], ddof: int = 1) -> float:
        return float(np.var(data, ddof=ddof))

    @staticmethod
    def std(data: Union[List[float], np.ndarray], ddof: int = 1) -> float:
        return float(np.std(data, ddof=ddof))

    @staticmethod
    def percentile(data: Union[List[float], np.ndarray], p: float) -> float:
        return float(np.percentile(data, p))

    @staticmethod
    def quartiles(data: Union[List[float], np.ndarray]) -> Dict[str, float]:
        q1, q2, q3 = np.percentile(data, [25, 50, 75])
        return {"Q1": float(q1), "Q2": float(q2), "Q3": float(q3)}

    @staticmethod
    def iqr(data: Union[List[float], np.ndarray]) -> float:
        return float(np.percentile(data, 75) - np.percentile(data, 25))

    @staticmethod
    def covariance_matrix(data: Matrix) -> Matrix:
        return Matrix(np.cov(data._data, rowvar=False))

    @staticmethod
    def correlation_coefficient(x: Union[List[float], np.ndarray],
                                 y: Union[List[float], np.ndarray]) -> float:
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def linear_regression(x: Union[List[float], np.ndarray],
                          y: Union[List[float], np.ndarray]) -> Dict[str, float]:
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        n = len(x_arr)
        sx, sy = x_arr.sum(), y_arr.sum()
        sxx = (x_arr ** 2).sum()
        sxy = (x_arr * y_arr).sum()
        slope = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
        intercept = (sy - slope * sx) / n
        y_pred = slope * x_arr + intercept
        ss_res = ((y_arr - y_pred) ** 2).sum()
        ss_tot = ((y_arr - sy / n) ** 2).sum()
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
        }

    @staticmethod
    def normal_pdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
        return float(np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi)))

    @staticmethod
    def normal_cdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
        return float(0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2)))))

    @staticmethod
    def random_sample(dist: str = "uniform", **kwargs) -> float:
        if dist == "uniform":
            low = kwargs.get("low", 0.0)
            high = kwargs.get("high", 1.0)
            return float(np.random.uniform(low, high))
        if dist == "normal":
            loc = kwargs.get("mean", 0.0)
            scale = kwargs.get("std", 1.0)
            return float(np.random.normal(loc, scale))
        raise ValueError(f"Unknown distribution: {dist}")
