import logging
import math
from typing import Union

import numpy as np

from .matrix import Matrix

logger = logging.getLogger(__name__)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a,b) via continued fraction."""
    if x < 0.0 or x > 1.0:
        return float(x)
    if x == 0.0 or x == 1.0:
        return x
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a

    def d_even(m: int) -> float:
        return m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))

    def d_odd(m: int) -> float:
        return -(a + m - 1) * (a + b + m - 1) * x / ((a + 2 * m - 2) * (a + 2 * m - 1))

    cf = 1.0
    for m in range(200, 0, -1):
        cf = 1.0 + d_even(m) / cf
        cf = 1.0 + d_odd(m) / cf
    cf = 1.0 / cf

    return front * cf


def _gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a,x) via series representation."""
    if x < 0.0 or a <= 0.0:
        return 0.0
    if x == 0.0:
        return 0.0

    ln_gamma_a = math.lgamma(a)
    ap = a
    summ = 1.0 / a
    delt = summ
    for _ in range(201):
        ap += 1.0
        delt *= x / ap
        summ += delt
        if abs(delt) < abs(summ) * 1e-14:
            break
    return summ * math.exp(-x + a * math.log(x) - ln_gamma_a)


def _t_cdf(t: float, df: int) -> float:
    """CDF of Student's t-distribution."""
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - p if t >= 0 else p


def _chi2_cdf(x: float, df: int) -> float:
    """CDF of the chi-square distribution."""
    return _gammainc(df / 2.0, x / 2.0)


def _f_cdf(f: float, df1: int, df2: int) -> float:
    """CDF of the F-distribution."""
    x = df1 * f / (df1 * f + df2)
    return _betainc(df1 / 2.0, df2 / 2.0, x)


class Statistics:
    """Statistical functions including descriptive statistics and distributions."""

    @staticmethod
    def mean(data: Union[list[float], np.ndarray]) -> float:
        """Compute the arithmetic mean of the data.

        Args:
            data (Union[List[float], np.ndarray]): Input data.

        Returns:
            float: The arithmetic mean.
        """
        return float(np.mean(data))

    @staticmethod
    def median(data: Union[list[float], np.ndarray]) -> float:
        """Compute the median of the data.

        Args:
            data (Union[List[float], np.ndarray]): Input data.

        Returns:
            float: The median value.
        """
        return float(np.median(data))

    @staticmethod
    def mode(data: Union[list[float], np.ndarray]) -> list[float]:
        """Compute the mode(s) of the data.

        Args:
            data (Union[List[float], np.ndarray]): Input data.

        Returns:
            List[float]: A list of the most frequent value(s).
        """
        values, counts = np.unique(data, return_counts=True)
        max_count = counts.max()
        return [float(v) for v in values[counts == max_count]]

    @staticmethod
    def variance(data: Union[list[float], np.ndarray], ddof: int = 1) -> float:
        """Compute the sample variance of the data.

        Args:
            data (Union[List[float], np.ndarray]): Input data.
            ddof (int): Delta degrees of freedom (default 1 for sample variance).

        Returns:
            float: The variance.
        """
        return float(np.var(data, ddof=ddof))

    @staticmethod
    def std(data: Union[list[float], np.ndarray], ddof: int = 1) -> float:
        """Compute the sample standard deviation of the data.

        Args:
            data (Union[List[float], np.ndarray]): Input data.
            ddof (int): Delta degrees of freedom (default 1 for sample std).

        Returns:
            float: The standard deviation.
        """
        return float(np.std(data, ddof=ddof))

    @staticmethod
    def percentile(data: Union[list[float], np.ndarray], p: float) -> float:
        """Compute the p-th percentile of the data.

        Args:
            data (Union[List[float], np.ndarray]): Input data.
            p (float): The percentile to compute (0-100).

        Returns:
            float: The p-th percentile value.
        """
        return float(np.percentile(data, p))

    @staticmethod
    def quartiles(data: Union[list[float], np.ndarray]) -> dict[str, float]:
        """Compute the first, second, and third quartiles.

        Args:
            data (Union[List[float], np.ndarray]): Input data.

        Returns:
            Dict[str, float]: Dictionary with keys Q1, Q2, Q3.
        """
        q1, q2, q3 = np.percentile(data, [25, 50, 75])
        return {"Q1": float(q1), "Q2": float(q2), "Q3": float(q3)}

    @staticmethod
    def iqr(data: Union[list[float], np.ndarray]) -> float:
        """Compute the interquartile range (Q3 - Q1).

        Args:
            data (Union[List[float], np.ndarray]): Input data.

        Returns:
            float: The interquartile range.
        """
        return float(np.percentile(data, 75) - np.percentile(data, 25))

    @staticmethod
    def covariance_matrix(data: Union[Matrix, list[list[float]], np.ndarray],
                          ddof: int = 1) -> Matrix:
        """Compute the covariance matrix of the data.

        Args:
            data (Matrix | List[List[float]] | np.ndarray): Data where rows are observations
                and columns are variables.
            ddof (int): Delta degrees of freedom (default 1).

        Returns:
            Matrix: The covariance matrix.
        """
        if isinstance(data, Matrix):
            arr = data._data
        else:
            arr = np.array(data, dtype=float)
        return Matrix(np.cov(arr, rowvar=False, ddof=ddof))

    @staticmethod
    def zscore(data: Union[list[float], np.ndarray],
               ddof: int = 0) -> list[float]:
        """Compute the z-score of each data point.

        Args:
            data (Union[List[float], np.ndarray]): Input data.
            ddof (int): Delta degrees of freedom for std (default 0 for population).

        Returns:
            List[float]: Z-scores.
        """
        arr = np.array(data, dtype=float)
        return ((arr - arr.mean()) / arr.std(ddof=ddof)).tolist()

    @staticmethod
    def mad(data: Union[list[float], np.ndarray]) -> float:
        """Compute the median absolute deviation.

        Args:
            data (Union[List[float], np.ndarray]): Input data.

        Returns:
            float: The median absolute deviation.
        """
        arr = np.array(data, dtype=float)
        med = np.median(arr)
        return float(np.median(np.abs(arr - med)))

    @staticmethod
    def ttest_1samp(data: Union[list[float], np.ndarray],
                     popmean: float = 0.0) -> dict[str, float]:
        """Perform a one-sample t-test.

        Args:
            data (Union[List[float], np.ndarray]): Sample data.
            popmean (float): Population mean under the null hypothesis (default 0.0).

        Returns:
            Dict[str, float]: Dictionary with keys ``t_statistic`` and ``p_value``.
        """
        arr = np.array(data, dtype=float)
        n = len(arr)
        mean = arr.mean()
        std = arr.std(ddof=1)
        se = std / math.sqrt(n)
        t = (mean - popmean) / se if se != 0 else 0.0
        df = n - 1
        p = 2 * (1 - _t_cdf(abs(t), df))
        return {"t_statistic": float(t), "p_value": float(p)}

    @staticmethod
    def ttest_ind(a: Union[list[float], np.ndarray],
                  b: Union[list[float], np.ndarray]) -> dict[str, float]:
        """Perform an independent two-sample t-test (Welch's).

        Args:
            a (Union[List[float], np.ndarray]): First sample.
            b (Union[List[float], np.ndarray]): Second sample.

        Returns:
            Dict[str, float]: Dictionary with keys ``t_statistic`` and ``p_value``.
        """
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        na, nb = len(a_arr), len(b_arr)
        mean_a, mean_b = a_arr.mean(), b_arr.mean()
        var_a, var_b = a_arr.var(ddof=1), b_arr.var(ddof=1)
        se = math.sqrt(var_a / na + var_b / nb)
        t = (mean_a - mean_b) / se if se != 0 else 0.0
        df_num = (var_a / na + var_b / nb) ** 2
        df_den = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
        df = df_num / df_den if df_den != 0 else 1.0
        p = 2 * (1 - _t_cdf(abs(t), round(df)))
        return {"t_statistic": float(t), "p_value": float(p)}

    @staticmethod
    def ttest_paired(a: Union[list[float], np.ndarray],
                     b: Union[list[float], np.ndarray]) -> dict[str, float]:
        """Perform a paired two-sample t-test.

        Args:
            a (Union[List[float], np.ndarray]): First sample.
            b (Union[List[float], np.ndarray]): Second sample.

        Returns:
            Dict[str, float]: Dictionary with keys ``t_statistic`` and ``p_value``.
        """
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        d = a_arr - b_arr
        n = len(d)
        mean_d = d.mean()
        std_d = d.std(ddof=1)
        se = std_d / math.sqrt(n)
        t = mean_d / se if se != 0 else 0.0
        df = n - 1
        p = 2 * (1 - _t_cdf(abs(t), df))
        return {"t_statistic": float(t), "p_value": float(p)}

    @staticmethod
    def chisquare(observed: Union[list[float], np.ndarray],
                   expected: Union[list[float], np.ndarray, None] = None) -> dict[str, float]:
        """Perform a chi-square goodness-of-fit test.

        Args:
            observed (Union[List[float], np.ndarray]): Observed frequencies.
            expected (Union[List[float], np.ndarray], optional): Expected frequencies.
                If None, uniform expectation is used.

        Returns:
            Dict[str, float]: Dictionary with keys ``chi2_statistic`` and ``p_value``.
        """
        obs = np.array(observed, dtype=float)
        if expected is None:
            exp = np.full_like(obs, obs.mean())
        else:
            exp = np.array(expected, dtype=float)
        chi2 = float(((obs - exp) ** 2 / exp).sum())
        df = len(obs) - 1
        p = 1 - _chi2_cdf(chi2, df)
        return {"chi2_statistic": chi2, "p_value": float(p)}

    @staticmethod
    def f_oneway(*samples) -> dict[str, float]:
        """Perform a one-way ANOVA (F-test).

        Args:
            *samples: Two or more sample arrays.

        Returns:
            Dict[str, float]: Dictionary with keys ``f_statistic`` and ``p_value``.
        """
        arrays = [np.array(s, dtype=float) for s in samples]
        all_data = np.concatenate(arrays)
        grand_mean = all_data.mean()
        ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
        ss_within = sum(((a - a.mean()) ** 2).sum() for a in arrays)
        k = len(arrays)
        n = len(all_data)
        df_between = k - 1
        df_within = n - k
        ms_between = ss_between / df_between if df_between > 0 else 0.0
        ms_within = ss_within / df_within if df_within > 0 else 0.0
        f = ms_between / ms_within if ms_within > 0 else 0.0
        p = 1 - _f_cdf(f, df_between, df_within)
        return {"f_statistic": float(f), "p_value": float(p)}

    @staticmethod
    def uniform_pdf(x: float, low: float = 0.0, high: float = 1.0) -> float:
        """Evaluate the uniform probability density function at x.

        Args:
            x (float): The point at which to evaluate.
            low (float): Lower bound (default 0.0).
            high (float): Upper bound (default 1.0).

        Returns:
            float: The PDF value at x.
        """
        if low >= high:
            raise ValueError("low must be less than high")
        return 1.0 / (high - low) if low <= x <= high else 0.0

    @staticmethod
    def uniform_cdf(x: float, low: float = 0.0, high: float = 1.0) -> float:
        """Evaluate the uniform cumulative distribution function at x.

        Args:
            x (float): The point at which to evaluate.
            low (float): Lower bound (default 0.0).
            high (float): Upper bound (default 1.0).

        Returns:
            float: The CDF value at x.
        """
        if low >= high:
            raise ValueError("low must be less than high")
        if x < low:
            return 0.0
        if x > high:
            return 1.0
        return (x - low) / (high - low)

    @staticmethod
    def exponential_pdf(x: float, rate: float = 1.0) -> float:
        """Evaluate the exponential probability density function at x.

        Args:
            x (float): The point at which to evaluate.
            rate (float): The rate parameter (lambda, default 1.0).

        Returns:
            float: The PDF value at x.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        return rate * math.exp(-rate * x) if x >= 0 else 0.0

    @staticmethod
    def binomial_pmf(k: int, n: int, p: float) -> float:
        """Evaluate the binomial probability mass function at k.

        Args:
            k (int): Number of successes.
            n (int): Number of trials.
            p (float): Probability of success per trial.

        Returns:
            float: The PMF value at k.
        """
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
        if k < 0 or k > n:
            return 0.0
        return float(math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)))

    @staticmethod
    def correlation_coefficient(x: Union[list[float], np.ndarray],
                                 y: Union[list[float], np.ndarray]) -> float:
        """Compute the Pearson correlation coefficient between x and y.

        Args:
            x (Union[List[float], np.ndarray]): First variable.
            y (Union[List[float], np.ndarray]): Second variable.

        Returns:
            float: The correlation coefficient.
        """
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def linear_regression(x: Union[list[float], np.ndarray],
                          y: Union[list[float], np.ndarray]) -> dict[str, float]:
        """Fit a simple linear regression model y = slope * x + intercept.

        Args:
            x (Union[List[float], np.ndarray]): Independent variable.
            y (Union[List[float], np.ndarray]): Dependent variable.

        Returns:
            Dict[str, float]: Dictionary with keys slope, intercept, r_squared.
        """
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
        """Evaluate the normal probability density function at x.

        Args:
            x (float): The point at which to evaluate.
            mean (float): The mean of the distribution (default 0.0).
            std (float): The standard deviation (default 1.0).

        Returns:
            float: The PDF value at x.
        """
        return float(np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi)))

    @staticmethod
    def normal_cdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
        """Evaluate the normal cumulative distribution function at x.

        Args:
            x (float): The point at which to evaluate.
            mean (float): The mean of the distribution (default 0.0).
            std (float): The standard deviation (default 1.0).

        Returns:
            float: The CDF value at x.
        """
        return float(0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2)))))

    @staticmethod
    def random_sample(dist: str = "uniform", **kwargs) -> float:
        """Generate a random sample from the specified distribution.

        Args:
            dist (str): The distribution name ("uniform" or "normal").
            **kwargs: Distribution parameters (low, high for uniform; mean, std for normal).

        Returns:
            float: A random sample.

        Raises:
            ValueError: If the distribution name is unknown.
        """
        if dist == "uniform":
            low = kwargs.get("low", 0.0)
            high = kwargs.get("high", 1.0)
            return float(np.random.uniform(low, high))
        if dist == "normal":
            loc = kwargs.get("mean", 0.0)
            scale = kwargs.get("std", 1.0)
            return float(np.random.normal(loc, scale))
        raise ValueError(f"Unknown distribution: {dist}")
