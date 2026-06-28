import math

from axiompy import Axiom

# ---- Beta-Binomial: coin bias estimation ----
beta_prior = Axiom.BetaBinomial(2, 2)         # weak prior favoring 0.5
post = beta_prior.posterior(k=7, n=10)         # 7 heads in 10 flips
print(f"Beta-Binomial posterior: alpha={post.alpha}, beta={post.beta}")
print(f"  posterior mean = {post.mean():.4f}")
print(f"  95% CI = {post.credible_interval()}")

# ---- Normal-Normal: estimating a mean ----
nn_prior = Axiom.NormalNormal(mu0=0, sigma0=10)  # vague prior
data = [1.2, 1.5, 0.8, 1.1, 1.3]
nn_post = nn_prior.posterior(data, sigma2=0.25)   # known likelihood variance
print(f"\nNormal-Normal posterior: mu={nn_post.mu:.4f}, sigma={nn_post.sigma:.4f}")
print(f"  95% CI = {nn_post.credible_interval()}")

# ---- Poisson-Gamma: count data ----
pg_prior = Axiom.PoissonGamma(shape=1, rate=1)
pg_post = pg_prior.posterior([2, 0, 3, 1, 2, 4, 1])
print(f"\nPoisson-Gamma posterior: shape={pg_post.shape}, rate={pg_post.rate}")
print(f"  posterior mean = {pg_post.mean():.4f}")

# ---- Generic posterior dispatch ----
post = Axiom.posterior(Axiom.BetaBinomial(1, 1), 'binomial', (3, 10))
print(f"\nGeneric posterior: alpha={post.alpha}, beta={post.beta}")

# ---- MCMC: sampling from N(0, 1) ----


def log_pdf(x):
    return -0.5 * x[0] ** 2 - math.log(math.sqrt(2 * math.pi))

samples = Axiom.mcmc_metropolis(log_pdf, [0.0], steps=2000, proposal_std=1.0, seed=42)
mean = sum(s[0] for s in samples) / len(samples)
print(f"\nMCMC posterior mean = {mean:.4f}  (expected ~0)")

intervals = Axiom.credible_interval(samples, mass=0.95)
print(f"  95% CI = {intervals}")
