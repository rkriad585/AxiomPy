import pytest

from axiompy import Axiom
from axiompy.bayesian import BetaBinomial, NormalNormal, PoissonGamma


class TestBetaBinomial:
    def test_prior_default(self):
        bb = BetaBinomial()
        assert bb.alpha == 1.0
        assert bb.beta == 1.0

    def test_posterior(self):
        bb = BetaBinomial(2, 2)
        post = bb.posterior(3, 5)
        assert post.alpha == 5
        assert post.beta == 4

    def test_mean(self):
        bb = BetaBinomial(2, 2)
        assert bb.mean() == 0.5

    def test_credible_interval(self):
        lo, hi = BetaBinomial(10, 10).credible_interval()
        assert lo < hi


class TestNormalNormal:
    def test_posterior(self):
        nn = NormalNormal(0, 10)
        post = nn.posterior([1, 2, 3], sigma2=1.0)
        assert abs(post.mu - 2.0) < 0.1
        assert post.sigma < 10

    def test_credible_interval(self):
        lo, hi = NormalNormal(0, 1).credible_interval()
        assert lo < 0 < hi


class TestPoissonGamma:
    def test_posterior(self):
        pg = PoissonGamma(1, 1)
        post = pg.posterior([2, 3, 5])
        assert post.shape == 11
        assert post.rate == 4

    def test_mean(self):
        pg = PoissonGamma(5, 2)
        assert pg.mean() == 2.5


class TestPosterior:
    def test_binomial(self):
        prior = BetaBinomial(1, 1)
        post = Axiom.posterior(prior, 'binomial', (3, 10))
        assert post.alpha == 4

    def test_normal(self):
        prior = NormalNormal(0, 10)
        post = Axiom.posterior(prior, 'normal', ([1, 2, 3], 1.0))
        assert abs(post.mu - 2.0) < 0.1

    def test_poisson(self):
        prior = PoissonGamma(1, 1)
        post = Axiom.posterior(prior, 'poisson', [1, 2, 3])
        assert post.shape == 7

    def test_invalid(self):
        with pytest.raises(TypeError):
            Axiom.posterior('foo', 'bar', None)


class TestMCMC:
    def test_metropolis_1d(self):
        def log_pdf(x):
            return -0.5 * x[0] ** 2
        samples = Axiom.mcmc_metropolis(log_pdf, [0.0], steps=500, proposal_std=1.0, seed=42)
        assert len(samples) == 500
        mean = sum(s[0] for s in samples) / 500
        assert abs(mean) < 0.3

    def test_credible_interval(self):
        samples = [[x] for x in [1, 2, 2, 3, 3, 3, 4, 4, 5]]
        intervals = Axiom.credible_interval(samples, mass=0.5)
        lo, hi = intervals[0]
        assert lo <= hi

    def test_facade(self):
        bb = Axiom.BetaBinomial(1, 1)
        assert bb.alpha == 1.0
