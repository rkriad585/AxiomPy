import math
import random
from typing import Callable, Optional

# ---- helper distributions (used internally) --------------------------------

def _beta_pdf(x: float, a: float, b: float) -> float:
    if x <= 0 or x >= 1:
        return 0.0
    from .special import beta as beta_fn
    return x ** (a - 1) * (1 - x) ** (b - 1) / beta_fn(a, b)


def _gamma_pdf(x: float, shape: float, rate: float) -> float:
    if x <= 0:
        return 0.0
    from .special import gamma as gamma_fn
    return rate ** shape * x ** (shape - 1) * math.exp(-rate * x) / gamma_fn(shape)


# ---- Conjugate families ----------------------------------------------------

class BetaBinomial:
    """Beta-Binomial conjugate family for Bernoulli/binomial likelihood.

    Posterior: Beta(alpha + k, beta + n - k)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def posterior(self, k: int, n: int) -> 'BetaBinomial':
        """Return a new BetaBinomial with updated parameters given k successes in n trials.

        Args:
            k: Number of successes.
            n: Total number of trials.

        Returns:
            BetaBinomial: Posterior distribution.
        """
        return BetaBinomial(self.alpha + k, self.beta + n - k)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1))

    def credible_interval(self, mass: float = 0.95) -> tuple[float, float]:
        """Approximate credible interval via normal approximation to Beta."""
        m = self.mean()
        v = self.variance()
        sigma = math.sqrt(v)
        z = 1.96  # ~95% for large mass, close enough
        lo = max(0.0, m - z * sigma)
        hi = min(1.0, m + z * sigma)
        return (lo, hi)


class NormalNormal:
    """Normal-Normal conjugate family for Gaussian likelihood with known variance.

    Prior: N(mu0, sigma0^2).  Likelihood variance ``sigma2`` is assumed known.
    """

    def __init__(self, mu0: float = 0.0, sigma0: float = 1.0):
        self.mu = mu0
        self.sigma = sigma0

    def posterior(self, data: list[float], sigma2: float) -> 'NormalNormal':
        """Return a new NormalNormal with updated parameters.

        Args:
            data: Observed data points.
            sigma2: Known likelihood variance.

        Returns:
            NormalNormal: Posterior distribution.
        """
        n = len(data)
        xbar = sum(data) / n
        inv_var_prior = 1.0 / (self.sigma ** 2)
        inv_var_lik = n / sigma2
        post_var = 1.0 / (inv_var_prior + inv_var_lik)
        post_mu = post_var * (inv_var_prior * self.mu + inv_var_lik * xbar)
        return NormalNormal(post_mu, math.sqrt(post_var))

    def mean(self) -> float:
        return self.mu

    def variance(self) -> float:
        return self.sigma ** 2

    def credible_interval(self, mass: float = 0.95) -> tuple[float, float]:
        z = 1.96
        return (self.mu - z * self.sigma, self.mu + z * self.sigma)


class PoissonGamma:
    """Poisson-Gamma conjugate family for Poisson likelihood.

    Prior: Gamma(shape, rate).  Posterior: Gamma(shape + sum(data), rate + n).
    """

    def __init__(self, shape: float = 1.0, rate: float = 1.0):
        self.shape = shape
        self.rate = rate

    def posterior(self, data: list[float]) -> 'PoissonGamma':
        """Return a new PoissonGamma with updated parameters.

        Args:
            data: Observed count data.

        Returns:
            PoissonGamma: Posterior distribution.
        """
        return PoissonGamma(self.shape + sum(data), self.rate + len(data))

    def mean(self) -> float:
        return self.shape / self.rate

    def variance(self) -> float:
        return self.shape / (self.rate ** 2)

    def credible_interval(self, mass: float = 0.95) -> tuple[float, float]:
        # Gamma normal approximation
        m = self.mean()
        s = math.sqrt(self.variance())
        z = 1.96
        return (max(0.0, m - z * s), m + z * s)


# ---- Generic posterior -----------------------------------------------------

def posterior(prior, likelihood: str, data) -> object:
    """Generic conjugate update.

    Args:
        prior: A conjugate-distribution instance (BetaBinomial, NormalNormal, PoissonGamma).
        likelihood:  One of ``'binomial'``, ``'normal'``, ``'poisson'``.
        data: Data appropriate for the likelihood type.

    Returns:
        Posterior distribution instance (same type as ``prior``).
    """
    if isinstance(prior, BetaBinomial) and likelihood == 'binomial':
        k, n = data
        return prior.posterior(k, n)
    if isinstance(prior, NormalNormal) and likelihood == 'normal':
        return prior.posterior(data[0], data[1])
    if isinstance(prior, PoissonGamma) and likelihood == 'poisson':
        return prior.posterior(data)
    raise TypeError(f"No conjugate update known for {type(prior).__name__} with likelihood '{likelihood}'")


# ---- MCMC ------------------------------------------------------------------

def mcmc_metropolis(log_pdf: Callable[[list[float]], float],
                    start: list[float],
                    steps: int = 1000,
                    proposal_std: float = 0.1,
                    seed: Optional[int] = None) -> list[list[float]]:
    """Metropolis MCMC sampler.

    Args:
        log_pdf: Log of the (unnormalized) target density.
        start: Starting point.
        steps: Number of MCMC steps.
        proposal_std: Standard deviation of the Gaussian proposal.
        seed: Random seed for reproducibility.

    Returns:
        List of samples (each sample is a list of floats).
    """
    if seed is not None:
        random.seed(seed)
    dim = len(start)
    current = start[:]
    log_curr = log_pdf(current)
    samples = [current[:]]

    for _ in range(steps - 1):
        proposal = [current[i] + random.gauss(0, proposal_std) for i in range(dim)]
        log_prop = log_pdf(proposal)
        log_alpha = log_prop - log_curr
        if log_alpha >= 0 or random.random() < math.exp(log_alpha):
            current = proposal
            log_curr = log_prop
        samples.append(current[:])

    return samples


def credible_interval(samples: list[list[float]], mass: float = 0.95
                      ) -> list[tuple[float, float]]:
    """Compute an equal-tailed credible interval from MCMC samples.

    Args:
        samples: MCMC samples (each element is a parameter vector).
        mass: Probability mass to cover (default 0.95).

    Returns:
        List of ``(lower, upper)`` tuples, one per parameter dimension.
    """
    dim = len(samples[0])
    alpha = (1.0 - mass) / 2
    intervals = []
    for j in range(dim):
        col = sorted(samples[i][j] for i in range(len(samples)))
        lo = col[int(alpha * len(col))]
        hi = col[int((1 - alpha) * len(col))]
        intervals.append((lo, hi))
    return intervals
