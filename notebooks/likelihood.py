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


class StreamDensityModel:

    def __init__(self, X, density_model, h, bg_ln_likelihood,
                 m_prior_sigma=None, frozen=None):

        if frozen is None:
            frozen = dict()
        self.frozen = dict(frozen)

        self.X = np.array(X)
        self.N = len(self.X)

        if density_model.K is None:
            raise ValueError('Density model not initialized')
        self.density_model = density_model
        self._K = self.density_model.K
        self._nodes = self.density_model.nodes

        self.h = h

        self.bg_ln_likelihood = bg_ln_likelihood

        if m_prior_sigma is None: # width of prior on m
            m_prior_sigma = self.h
        self.m_prior_sigma = m_prior_sigma

        self._params = dict()
        self._params['ln_z'] = (self.density_model.K-1, )
        self._params['ln_s'] = (self.density_model.K, )
        self._params['m'] = (self.density_model.K, )
        self._params['f'] = (1, )
        # TODO: how to generalize the background model? for now, assume uniform
        # - I could take functions that evaluate the background likelihood
        #   and any derivatives?

        self._params_sorted = sorted(list(self._params.keys()))

    # =========================================================================
    # Parameter packing and unpacking:
    #

    def pack_pars(self, **kwargs):
        vals = []
        for k in self._params_sorted:
            frozen_val = self.frozen.get(k, None)
            val = kwargs.get(k, frozen_val)
            if val is None:
                raise ValueError("No value passed in for parameter {0}, but "
                                 "it isn't frozen either!".format(k))
            vals.append(np.array(val).reshape(self._params[k]))
        return np.concatenate(vals)

    def unpack_pars(self, p):
        key_vals = []

        j = 0
        for name in self._params_sorted:
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
        mu = self._nodes.copy()
        mu[:, 1] = mu[:, 1] + p['m']
        return mu

    # =========================================================================
    # Probability functions:
    #

    def ln_prior(self, p):
        lp = 0.

        if 'ln_z' not in self.frozen:
            if np.any(p['ln_z'] > 0):
                return -np.inf

        if 'm' not in self.frozen:
            # this is like L2 regularization -- I should try Lasso
            lp += ln_normal(p['m'], 0, self.m_prior_sigma).sum()

        return lp

    def ln_density(self, p, X):
        s = self.get_s(p)
        a = self.get_a(p)
        mu = self.get_mu(p)
        return self.density_model.ln_density(X, a, s, self.h, mu_k=mu)

    def ln_likelihood(self, p):
        ln_fg = self.ln_density(p, self.X)
        ln_bg = self.bg_ln_likelihood(p, self.X)
        return np.logaddexp(ln_fg + np.log(p['f']),
                            ln_bg + np.log(1 - p['f']))

    def ln_posterior(self, x):
        # unpack parameter vector, p
        kw_pars = self.unpack_pars(x)

        lnp = self.ln_prior(kw_pars)
        if not np.isfinite(lnp):
            return -np.inf

        lnl = self.ln_likelihood(kw_pars)
        if not np.isfinite(lnl).all():
            return -np.inf

        return lnp + lnl.sum()

    def __call__(self, x):
        return self.ln_posterior(x)

    # =========================================================================
    # Derivatives:
    #

    def ln_d_likelihood_dp(self, p):
        """Log of the derivatives of the likelihood (the linear likelihood, not
        the log-likelihood!) with respect to all enabled parameters.
        """

        R = self.density_model._R_k
        K = self.density_model.K
        X = self.X
        h = self.h
        N = self.N
        D = 2

        a = self.get_a(p)
        s = self.get_s(p)
        z = self.get_z(p)
        mu = self.get_mu(p)

        # We need this for all of the derivatives
        ln_N_nk = np.zeros((N, K))
        C = np.zeros((K, D, D))
        Cinv = np.zeros((K, D, D))
        for k in range(K):
            C[k] = self.density_model.get_Ck(k, s[k], h)
            Cinv[k] = self.density_model.get_Ck(k, s[k], h, inv=True)
            ln_N_nk[:, k] = mvn.logpdf(X, mu[k], C[k],
                                       allow_singular=True)
        ln_aN_nk = np.log(a) + ln_N_nk

        ln_derivs = dict()
        signs = dict()

        # ----
        if 'ln_z' not in self.frozen:
            ln_d_L_d_a = ln_N_nk
            ln_d_L_dznk = np.zeros((N, K-1))
            z_sign = np.ones((N, K-1))
            for k in range(K-1):
                # note the change: z-1 to 1-z and minus sign out front
                ln_t1 = np.log(a[k]) - np.log(1 - z[k]) + ln_d_L_d_a[:, k]

                terms = []
                for j in range(k+1, K):
                    terms.append(np.log(a[j]) - np.log(z[k]) + ln_d_L_d_a[:, j])
                ln_t2 = logsumexp(terms, axis=0)

                ln_d_L_dznk[:, k], z_sign[:, k] = logsumexp(
                    [ln_t1, ln_t2], b=[np.ones(N), -np.ones(N)],
                    axis=0, return_sign=True)

            ln_derivs['ln_z'] = ln_d_L_dznk + np.log(z)
            signs['ln_z'] = -z_sign # change of sign

        # ----
        if 'm' not in self.frozen:
            ln_d_L_d_mnk = np.zeros((N, K))
            m_sign = np.zeros((N, K))
            for k in range(K):
                # the [:,1] below picks off only the y term, because we only
                # allow the means to move in y
                Cinv_dx = np.einsum('ij,nj->ni', Cinv[k], X - mu[k:k+1])[:, 1]
                m_sign[:, k] = np.sign(Cinv_dx)
                log_Cinv_dx = np.log(np.abs(Cinv_dx))
                ln_d_L_d_mnk[:, k] = ln_aN_nk[:, k] + log_Cinv_dx

            ln_derivs['m'] = ln_d_L_d_mnk
            signs['m'] = m_sign

        # ----
        if 'ln_s' not in self.frozen:
            ln_d_L_d_snk = np.zeros((N, K))
            s_sign = np.zeros((N, K))
            for k in range(K):
                b = X - mu[k:k+1]

                term = ((b[:, 1]*R[k,0,0] + b[:, 0]*R[k,0,1])**2 / s[k]**2 - 1) / s[k]

                s_sign[:, k] = np.sign(term)
                log_term = np.log(np.abs(term))

                ln_d_L_d_snk[:, k] = ln_aN_nk[:, k] + log_term

            ln_derivs['ln_s'] = ln_d_L_d_snk + np.log(s)
            signs['ln_s'] = s_sign

        # ----
        if 'f' not in self.frozen:
            ln_fg = self.ln_density(p, X)
            ln_bg = self.bg_ln_likelihood(p, X)
            ln_numer, f_sign = logsumexp([ln_fg, ln_bg],
                                         b=np.stack((np.ones(N), -np.ones(N))),
                                         return_sign=True, axis=0)
            ln_derivs['f'] = ln_numer
            signs['f'] = f_sign

        for k in ln_derivs:
            ln_derivs[k] = ln_derivs[k].reshape((N, ) + self._params[k])
            signs[k] = signs[k].reshape((N, ) + self._params[k])

        return ln_derivs, signs

    def d_ln_likelihood_dp(self, p):
        ln_denom = self.ln_likelihood(p)
        derivs, signs = self.ln_d_likelihood_dp(p)

        full_derivs = dict()
        for name in self._params_sorted:
            if name == 'f':
                f_fac = 0.
            else:
                f_fac = np.log(p['f'])
            full_derivs[name] = np.sum(signs[name] *
                                       np.exp(f_fac + derivs[name] - ln_denom[:, None]),
                                       axis=0)

        return full_derivs
