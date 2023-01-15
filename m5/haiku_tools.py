import jax


def negative_binomial_log(y, eta, phi):
    log_phi = jax.numpy.log(phi)
    log_mu_phi = jax.numpy.logaddexp(eta, log_phi)
    log_binomial = jax.scipy.special.gammaln(y + phi) - jax.scipy.special.gammaln(y + 1) - jax.scipy.special.gammaln(phi)
    log_term1 = y * (eta - log_mu_phi)
    log_term2 = phi * (log_phi - log_mu_phi)
    return log_binomial + log_term1 + log_term2