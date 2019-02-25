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

    def __init__(self, X, density_model, h,
                 bg_ln_likelihood=None, d_bg_ln_likelihood_dp=None,
                 bg_params=None, bg_kwargs=None,
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

        if m_prior_sigma is None: # width of prior on m
            m_prior_sigma = self.h
        self.m_prior_sigma = m_prior_sigma

        self._params = dict()
        self._params['ln_z'] = (self.density_model.K-1, )
        self._params['ln_s'] = (self.density_model.K, )
        self._params['m'] = (self.density_model.K, )

        # Background model:
        if bg_ln_likelihood is not None:
            self._params['f'] = (1, )
        else:
            self.frozen['f'] = 1.

        if bg_kwargs is None:
            bg_kwargs = dict()
        self.bg_kwargs = bg_kwargs
        self.bg_ln_likelihood = bg_ln_likelihood
        self.d_bg_ln_likelihood_dp = d_bg_ln_likelihood_dp

        if bg_params is None:
            bg_params = dict()
        self._bg_params = bg_params
        for name, shape in bg_params.items():
            self._params[name] = shape

        # Packing and unpacking is done by sorted names:
        self._params_sorted = sorted(list(self._params.keys()))
        self._params_sorted_unfrozen = sorted([k for k in self._params.keys()
                                               if k not in self.frozen])

    # =========================================================================
    # Parameter packing and unpacking:
    #

    def pack_pars(self, p, fill_frozen=True):
        vals = []
        for k in self._params_sorted:
            if k in self.frozen:
                val = self.frozen.get(k, None)
                if not fill_frozen:
                    continue

            else:
                val = p.get(k, None)

            if val is None:
                raise ValueError("No value passed in for parameter {0}, but "
                                 "it isn't frozen either!".format(k))
            vals.append(np.array(val).reshape(self._params[k]))
        return np.concatenate(vals)

    def unpack_pars(self, x):
        key_vals = []

        j = 0
        for name in self._params_sorted:
            shape = self._params[name]
            size = np.prod(shape)
            if name in self.frozen:
                key_vals.append((name, self.frozen[name]))
            else:
                key_vals.append((name, x[j:j+size]))
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

    def ln_likelihood(self, p, X=None):
        if X is None:
            X = self.X
        ln_fg = self.ln_density(p, X)
        ln_bg = self.bg_ln_likelihood(p, X, **self.bg_kwargs)
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
            ln_bg = self.bg_ln_likelihood(p, X, **self.bg_kwargs)
            ln_numer, f_sign = logsumexp([ln_fg, ln_bg],
                                         b=np.stack((np.ones(N), -np.ones(N))),
                                         return_sign=True, axis=0)
            ln_derivs['f'] = ln_numer
            signs['f'] = f_sign

        for name in self._bg_params:
            ln_bg_derivs, bg_signs = self.d_bg_ln_likelihood_dp(p, X,
                                                                **self.bg_kwargs)
            if name not in self.frozen:
                ln_derivs[name] = ln_bg_derivs[name]
                signs[name] = bg_signs[name]

        for k in ln_derivs:
            ln_derivs[k] = ln_derivs[k].reshape((N, ) + self._params[k])
            signs[k] = signs[k].reshape((N, ) + self._params[k])

        return ln_derivs, signs

    def d_ln_likelihood_dp(self, p):
        ln_denom = self.ln_likelihood(p)
        derivs, signs = self.ln_d_likelihood_dp(p)

        full_derivs = dict()
        for name in self._params_sorted:
            if name in self.frozen:
                continue

            if name == 'f':
                f_fac = 0.
            elif name.startswith('bg'):
                f_fac = np.log(1 - p['f'])
            else:
                f_fac = np.log(p['f'])

            full_derivs[name] = np.sum(signs[name] *
                                       np.exp(f_fac + derivs[name] - ln_denom[:, None]),
                                       axis=0)

        return full_derivs


# Background models:

def ln_bg_quadratic_uniform(p, X, window_bounds):
    a, b = window_bounds[0]
    x1, x2 = X.T
    N = len(x1)

    # x1 direction:
    c1 = np.exp(p['ln_bg_c1'])
    c2 = np.exp(p['ln_bg_c2'])
    c3 = np.exp(p['ln_bg_c3'])
    x0 = p['bg_x0']

    lnA = np.log(6) - np.log((b - a)*(2*a**2*c1 + 2*a*b*c1 + 2*b**2*c1 + 3*a*c2 + 3*b*c2 + 6*c3 - 6*a*c1*x0 - 6*b*c1*x0 - 6*c2*x0 + 6*c1*x0**2))
    ln_px1 = lnA + logsumexp([np.log(c1) + np.log((x1-x0)**2),
                              np.log(c2) + np.log(np.abs(x1-x0)),
                              np.full_like(x1, np.log(c3))],
                             b=[np.ones(N), np.sign(x1-x0), np.ones(N)],
                             axis=0)

    # x2 direction:
    ln_px2 = -np.log(window_bounds[1][1] - window_bounds[1][0])

    return ln_px1 + ln_px2


def ln_d_ln_bg_quadratic_uniform_dp(p, X, window_bounds):
    a, b = window_bounds[0]
    x, x2 = X.T
    N = len(x)

    c1 = np.exp(p['ln_bg_c1'])
    c2 = np.exp(p['ln_bg_c2'])
    c3 = np.exp(p['ln_bg_c3'])
    x0 = p['bg_x0']

    derivs = dict()
    signs = dict()

    derivs['ln_bg_c1'] = ((6*(2*a**2*(c3 + c2*(x - x0)) + 2*b**2*(c3 + c2*(x - x0)) -
                          6*x*(c3*(x - 2*x0) + c2*x0*(-x + x0)) - 3*b*(2*c3*x0 + c2*(x - x0)*(x + x0)) +
                          a*(-3*c2*x**2 - 6*c3*x0 + 3*c2*x0**2 + 2*b*(c3 + c2*x - c2*x0)))) /
                        ((a - b)*(2*a**2*c1 + 2*a*b*c1 + 2*b**2*c1 + 3*a*c2 + 3*b*c2 + 6*c3 -
                                  6*((a + b)*c1 + c2)*x0 + 6*c1*x0**2)**2))

    derivs['ln_bg_c2'] = ((6*(2*a**2*c1*(-x + x0) + 2*b**2*c1*(-x + x0) - 6*x*(c3 + c1*(x - x0)*x0) +
   3*b*(c3 + c1*(x - x0)*(x + x0)) + a*(3*c3 + c1*(x - x0)*(-2*b + 3*(x + x0)))))/
 ((a - b)*(2*a**2*c1 + 2*a*b*c1 + 2*b**2*c1 + 3*a*c2 + 3*b*c2 + 6*c3 -
    6*((a + b)*c1 + c2)*x0 + 6*c1*x0**2)**2))

    derivs['ln_bg_c3'] = ((-6*(2*a**2*c1 + 2*b**2*c1 + 3*b*c2 + a*(2*b*c1 + 3*c2) - 6*x*(c2 + c1*x)) +
  36*c1*(a + b - 2*x)*x0)/((a - b)*(2*a**2*c1 + 2*a*b*c1 + 2*b**2*c1 + 3*a*c2 + 3*b*c2 +
    6*c3 - 6*((a + b)*c1 + c2)*x0 + 6*c1*x0**2)**2))

    derivs['bg_x0'] = ((6*(6*(c2 + c1*(a + b - 2*x0))*(c3 + (c2 + c1*(x - x0))*(x - x0)) -
   (c2 + 2*c1*(x - x0))*(2*a**2*c1 + 2*a*b*c1 + 2*b**2*c1 + 3*a*c2 + 3*b*c2 + 6*c3 -
     6*((a + b)*c1 + c2)*x0 + 6*c1*x0**2)))/
 ((-a + b)*(2*a**2*c1 + 2*a*b*c1 + 2*b**2*c1 + 3*a*c2 + 3*b*c2 + 6*c3 -
    6*((a + b)*c1 + c2)*x0 + 6*c1*x0**2)**2))

    # because it's multiplied in the likelihood
    ln_px2 = -np.log(window_bounds[1][1] - window_bounds[1][0])

    for name in ['ln_bg_c1', 'ln_bg_c2', 'ln_bg_c3', 'bg_x0']:
        signs[name] = np.sign(derivs[name])
        derivs[name] = np.log(np.abs(derivs[name])) + ln_px2

        if 'bg_c' in name:
            derivs[name] = derivs[name] + p[name]

    return derivs, signs


def ln_bg_uniform(p, X, window_bounds):
    N = len(X)

    # x1 direction:
    ln_px1 = -np.log(window_bounds[0][1] - window_bounds[0][0])

    # x2 direction:
    ln_px2 = -np.log(window_bounds[1][1] - window_bounds[1][0])

    return np.full(N, ln_px1 + ln_px2)


def ln_d_ln_bg_uniform_dp(p, X, window_bounds):
    return dict(), dict()
