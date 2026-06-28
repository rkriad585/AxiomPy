import pytest

from axiompy import Axiom


class TestStatistics:
    def test_mean(self):
        assert Axiom.stats.mean([1, 2, 3, 4, 5]) == pytest.approx(3.0)

    def test_median_odd(self):
        assert Axiom.stats.median([1, 3, 2, 5, 4]) == pytest.approx(3.0)

    def test_median_even(self):
        assert Axiom.stats.median([1, 2, 3, 4]) == pytest.approx(2.5)

    def test_mode_single(self):
        assert Axiom.stats.mode([1, 2, 2, 3]) == [2.0]

    def test_mode_multi(self):
        assert sorted(Axiom.stats.mode([1, 1, 2, 2])) == [1.0, 2.0]

    def test_variance(self):
        assert Axiom.stats.variance([1, 2, 3, 4, 5]) == pytest.approx(2.5)

    def test_std(self):
        assert Axiom.stats.std([1, 2, 3, 4, 5]) == pytest.approx(1.581138, rel=1e-5)

    def test_percentile(self):
        assert Axiom.stats.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_quartiles(self):
        q = Axiom.stats.quartiles([1, 2, 3, 4, 5])
        assert q["Q1"] == pytest.approx(2.0)
        assert q["Q2"] == pytest.approx(3.0)
        assert q["Q3"] == pytest.approx(4.0)

    def test_iqr(self):
        assert Axiom.stats.iqr([1, 2, 3, 4, 5]) == pytest.approx(2.0)

    def test_correlation_coefficient(self):
        r = Axiom.stats.correlation_coefficient([1, 2, 3], [2, 4, 6])
        assert r == pytest.approx(1.0)

    def test_linear_regression(self):
        res = Axiom.stats.linear_regression([1, 2, 3], [2, 4, 6])
        assert res["slope"] == pytest.approx(2.0)
        assert res["intercept"] == pytest.approx(0.0)
        assert res["r_squared"] == pytest.approx(1.0)

    def test_normal_pdf(self):
        assert Axiom.stats.normal_pdf(0, 0, 1) == pytest.approx(0.398942, rel=1e-5)

    def test_normal_cdf(self):
        assert Axiom.stats.normal_cdf(0) == pytest.approx(0.5)

    def test_covariance_matrix(self):
        mat = Axiom.Matrix([[1, 2], [3, 4], [5, 6]])
        cov = Axiom.stats.covariance_matrix(mat)
        assert cov.shape == (2, 2)

    def test_covariance_matrix_raw(self):
        cov = Axiom.stats.covariance_matrix([[1, 2], [3, 4], [5, 6]])
        assert cov.shape == (2, 2)

    def test_zscore(self):
        z = Axiom.stats.zscore([1, 2, 3, 4, 5])
        assert z[2] == pytest.approx(0.0, abs=1e-10)

    def test_mad(self):
        m = Axiom.stats.mad([1, 2, 3, 4, 5])
        assert m == pytest.approx(1.0)

    def test_ttest_1samp(self):
        res = Axiom.stats.ttest_1samp([1, 2, 3, 4, 5], popmean=3)
        assert abs(res["t_statistic"]) < 1e-10
        assert res["p_value"] == pytest.approx(1.0, abs=1e-6)

    def test_ttest_ind(self):
        res = Axiom.stats.ttest_ind([1, 2, 3], [4, 5, 6])
        assert res["t_statistic"] == pytest.approx(-3.67423, rel=1e-4)
        assert res["p_value"] == pytest.approx(0.0213, rel=1e-3)

    def test_ttest_paired(self):
        before = [1, 2, 3]
        after = [2, 4, 5]
        res = Axiom.stats.ttest_paired(before, after)
        assert res["t_statistic"] == pytest.approx(-5.0, rel=1e-4)

    def test_chisquare(self):
        res = Axiom.stats.chisquare([10, 10, 10])
        assert res["chi2_statistic"] == pytest.approx(0.0, abs=1e-10)
        assert res["p_value"] == pytest.approx(1.0, abs=1e-6)

    def test_f_oneway(self):
        res = Axiom.stats.f_oneway([1, 2, 3], [4, 5, 6])
        assert res["f_statistic"] == pytest.approx(13.5, rel=1e-4)
        assert res["p_value"] == pytest.approx(0.0213, rel=1e-3)

    def test_uniform_pdf(self):
        assert Axiom.stats.uniform_pdf(0.5, 0, 1) == pytest.approx(1.0)
        assert Axiom.stats.uniform_pdf(-0.5, 0, 1) == pytest.approx(0.0)

    def test_uniform_cdf(self):
        assert Axiom.stats.uniform_cdf(0.5, 0, 1) == pytest.approx(0.5)
        assert Axiom.stats.uniform_cdf(-1, 0, 1) == pytest.approx(0.0)
        assert Axiom.stats.uniform_cdf(2, 0, 1) == pytest.approx(1.0)

    def test_exponential_pdf(self):
        assert Axiom.stats.exponential_pdf(0, rate=1) == pytest.approx(1.0)
        assert Axiom.stats.exponential_pdf(1, rate=2) == pytest.approx(0.270670, rel=1e-5)

    def test_binomial_pmf(self):
        assert Axiom.stats.binomial_pmf(2, 5, 0.5) == pytest.approx(0.3125, rel=1e-5)
        assert Axiom.stats.binomial_pmf(6, 5, 0.5) == pytest.approx(0.0)
