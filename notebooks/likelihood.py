# Third-party packages
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.special import logsumexp


def z_to_a(zj):
    K = len(zj) + 1
    a = np.zeros(K)
    a[0] = 1 - zj[0]
    for k in range(1, K-1):
        a[k] = np.prod(zj[:k]) * (1 - zj[k])
    a[K-1] = np.prod(zj)
    return a


def a_to_z(ak):
    K = len(ak)
    z = np.zeros(K - 1)
    z[0] = 1 - ak[0]
    for i in range(1, K-1):
        z[i] = 1 - ak[i] / np.prod(z[:i])
    return z


def ln_normal(x, mu, var):
    return -0.5*np.log(2*np.pi) - 0.5*np.log(var) - 0.5 * (x-mu)**2 / var


class Model:

    def __init__(self, density_model, h, l=None, frozen=None):
        if frozen is None:
            frozen = dict()
        self.frozen = dict(frozen)

        if density_model.K is None:
            raise ValueError('Density model not initialized')
        self.density_model = density_model
        self._X = density_model.X
        self._K = self.density_model.K

        self.h = h
        if l is None: # width of prior on m
            l = self.h
        self.l = l

        self._params = dict()
        self._params['ln_z'] = (self.density_model.K-1, )
        self._params['ln_s'] = (self.density_model.K, )
        self._params['m'] = (self.density_model.K, )

    def pack_pars(self, **kwargs):
        vals = []
        for k in sorted(list(self._params.keys())):
            frozen_val = self.frozen.get(k, None)
            val = kwargs.get(k, frozen_val)
            if val is None:
                raise ValueError("No value passed in for parameter {0}, but "
                                 "it isn't frozen either!".format(k))
            vals.append(val)
        return np.concatenate(vals)

    def unpack_pars(self, p):
        key_vals = []

        j = 0
        for name in sorted(list(self._params.keys())):
            shape = self._params[name]
            size = np.prod(shape)
            if name in self.frozen:
                key_vals.append((name, self.frozen[name]))
            else:
                key_vals.append((name, p[j:j+size]))
                j += size

        return dict(key_vals)

    def get_a(self, p):
        zj = self.get_z(p)
        return z_to_a(zj)

    def get_s(self, p):
        return np.exp(p['ln_s'])

    def get_z(self, p):
        return np.exp(p['ln_z'])

    def get_mu(self, p):
        mu = self.density_model.nodes.copy()
        mu[:, 1] = mu[:, 1] + p['m']
        return mu

    # =========================================================================
    # Probability functions:
    #

    def ln_prior(self, p):
        lp = 0.

        if np.any(p['ln_z'] > 0):
            return -np.inf

        # this is like L2 regularization, but I should try Lasso at some point
        lp += ln_normal(p['m'], 0, self.l).sum()

        return lp

    def ln_density(self, p, X, sum=True):
        s = self.get_s(p)
        a = self.get_a(p)
        mu = self.get_mu(p)
        return self.density_model.ln_density(X, a, s, self.h,
                                             nodes=mu, sum=sum)

    def ln_likelihood(self, p, sum=True):
        return self.ln_density(p, self._X, sum=sum)

    def ln_posterior(self, p):
        # unpack parameter vector, p
        kw_pars = self.unpack_pars(p)

        lnp = self.ln_prior(kw_pars)
        if not np.isfinite(lnp):
            return -np.inf

        lnl = self.ln_likelihood(kw_pars)
        if not np.isfinite(lnl).all():
            return -np.inf

        return lnp + lnl.sum()

    def __call__(self, p):
        return self.ln_posterior(p)

    def derivs_ln_post(self, p):
        """Derivatives of the ln_posterior, not ln_likelihood!"""

        R = self.density_model._R_k
        K = self.density_model.K
        X = self.density_model.X
        h = self.h
        (N, D) = X.shape

        a = self.get_a(p)
        s = self.get_s(p)
        z = self.get_z(p)
        nodes = self.get_mu(p)

        # We need this for all of the derivatives
        ln_N_nk = np.zeros((N, K))
        C = np.zeros((K, D, D))
        Cinv = np.zeros((K, D, D))
        for k in range(K):
            C[k] = self.density_model.get_Ck(k, s[k], h)
            Cinv[k] = self.density_model.get_Ck(k, s[k], h, inv=True)
            ln_N_nk[:, k] = mvn.logpdf(X, nodes[k], C[k],
                                       allow_singular=True)
        ln_aN_nk = np.log(a) + ln_N_nk
        ln_denom = logsumexp(ln_aN_nk, axis=1)

        # ---
        ln_d_lnL_d_mk = np.zeros((N, K))
        signs = np.zeros((N, K))
        for k in range(K):
            # the [:,1] below picks off only the y term, because we only allow
            # the means to move in y
            Cinv_dx = np.einsum('ij,nj->ni', Cinv[k], X - nodes[k:k+1])[:, 1]
            signs[:, k] = np.sign(Cinv_dx)
            log_Cinv_dx = np.log(np.abs(Cinv_dx))
            ln_d_lnL_d_mk[:, k] = ln_aN_nk[:, k] + log_Cinv_dx - ln_denom

        ln_d_lnL_d_m, sign = logsumexp(ln_d_lnL_d_mk, b=signs, axis=0,
                                       return_sign=True)
        # the last term is because of the prior we put on m
        d_lnL_d_m = sign * np.exp(ln_d_lnL_d_m) - p['m'] / self.l**2

        # ---
        ln_d_lnL_d_a = logsumexp(ln_N_nk - ln_denom[:, None], axis=0)
        d_lnL_d_a = np.exp(ln_d_lnL_d_a)

        d_lnL_d_z = np.zeros(K-1)
        for k in range(K-1):
            term1 = a[k] / (z[k] - 1) * d_lnL_d_a[k]

            term2 = 0.
            for j in range(k+1, K):
                term2 += a[j] / z[k] * d_lnL_d_a[j]
            d_lnL_d_z[k] = term1 + term2

        # ---
        ln_d_lnL_d_sk = np.zeros((N, K))
        signs = np.zeros((N, K))
        for k in range(K):
            b = X - nodes[k:k+1]

            # what APW derived:
            term = ((b[:, 1]*R[k,0,0] + b[:, 0]*R[k,0,1])**2/s[k]**2 - 1) / s[k]

            signs[:, k] = np.sign(term)
            log_term = np.log(np.abs(term))

            ln_d_lnL_d_sk[:, k] = ln_aN_nk[:, k] + log_term - ln_denom

        ln_d_lnL_d_s, sign = logsumexp(ln_d_lnL_d_sk, b=signs, axis=0,
                                       return_sign=True)
        d_lnL_d_s = sign * np.exp(ln_d_lnL_d_s)
        return np.concatenate((d_lnL_d_s * s, d_lnL_d_z * z, d_lnL_d_m))
