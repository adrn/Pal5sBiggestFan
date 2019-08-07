import os
from os import path
import pickle

from astropy.table import Table
import astropy.units as u
import numpy as np
from scipy.stats import truncnorm
from tqdm import tqdm
import emcee

from coordinates import pal5_lead_frame, pal5_trail_frame

##############################################################################
# Utilities


def get_phi2_mask(X):
    min_ = X[:, 1].min() + 0.15
    max_ = X[:, 1].max() - 0.15

    if (max_ - min_) < 1.:
        min_ -= 0.5
        max_ += 0.5

    return (X[:, 1] > min_) & (X[:, 1] < max_), (min_, max_)


def unpack_chain(chain, thin=8):
    # TODO: move this list elsewhere...
    param_names = ['ln_a', 'mu_s', 'ln_std_s', 'bg_p1', 'bg_p2']

    flatchain = np.vstack(chain[:, ::thin])
    if flatchain.shape[0] < 256:
        print("warning: flatchain has < 256 samples")

    res = {}
    for k, par in enumerate(param_names):
        if par.startswith('ln_'):
            name = par[3:]
            vals = np.exp(flatchain[..., k])
        else:
            name = par
            vals = flatchain[..., k]
        res[name] = vals

    return res


def run_sampler(X, phi2_lim, nwalkers, nburn, nsteps, pool=None):
    phi2_bins = np.arange(phi2_lim[0], phi2_lim[1], 0.1)  # MAGIC NUMBER
    H, _ = np.histogram(X[:, 1], bins=phi2_bins)
    phi2_bin_c = 0.5*(phi2_bins[:-1] + phi2_bins[1:])
    mu = phi2_bin_c[H.argmax()]
    if np.abs(mu) > 1.:
        mu = 0.

    # for 1-component gaussian
    # linear bg not quadratic:
    p0 = (np.log(0.2), mu, np.log(0.1)) + (0, 1.)

    p0s = emcee.utils.sample_ball(p0, [1e-3]*len(p0), nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, len(p0),
                                    log_prob_fn=lnprob,
                                    args=(X[:, 1], phi2_lim),
                                    pool=pool)

    pos, *_ = sampler.run_mcmc(p0s, nburn)
    pos = emcee.utils.sample_ball(np.median(pos, axis=0),
                                  [1e-3]*len(p0), nwalkers)
    sampler.reset()
    pos, *_ = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    _ = sampler.run_mcmc(pos, nsteps)

    return sampler


##############################################################################
# Likelihood and helpers


def ln_normal(x, mu, std):
    return -0.5 * (x-mu)**2 / std**2 - 0.5*np.log(2*np.pi) - np.log(std)


def ln_truncnorm(x, mu, sigma, clip_a, clip_b):
    a, b = (clip_a - mu) / sigma, (clip_b - mu) / sigma
    return truncnorm.logpdf(x, a, b, loc=mu, scale=sigma)


def lnprior(p):
    ln_a, mu_s, lnstd_s, *bg_p = p

    lp = 0

    if not -10 <= ln_a < 0:
        return -np.inf

    lp += ln_truncnorm(mu_s, 0, 1, -1, 1)
    # lp += ln_truncnorm(lnstd_s, -0.5, 1, -2.5, 1.5)

    for pp in bg_p:
        lp += ln_normal(pp, 0, 5)

    return lp


def lnlike(p, phi2, phi2_lim):
    ln_a, mu_s, lnstd_s, *bg_p = p

    a = np.exp(ln_a)
    a_bg = 1 - a

    stream = ln_normal(phi2, mu_s, np.exp(lnstd_s)) + ln_a

    # Background model:
    phi2_min, phi2_max = phi2_lim

    # for quadratic bg:
    # a, b, c = bg_p
    # for linear bg:
    b, c = bg_p
    a = 0.
    lnA = np.log(6) - np.log(-6*c*phi2_min - 3*b*phi2_min**2 - 2*a*phi2_min**3 +
                             6*c*phi2_max + 3*b*phi2_max**2 + 2*a*phi2_max**3)
    bg_ll = lnA + np.log(a*phi2**2 + b*phi2 + c)
    bg = bg_ll + np.log(a_bg)

    # for constant bg:
    # bg_ll = -np.log(phi2_max - phi2_min)
    # bg = np.full_like(stream1, bg_ll) + np.log(a_bg)

    # This is a wee bit faster than logsumexp
    return np.logaddexp(stream, bg)


def lnprob(p, phi2, phi2_lim):
    lp = lnprior(p)
    if not np.all(np.isfinite(lp)):
        return -np.inf

    ll = lnlike(p, phi2, phi2_lim).sum()
    if not np.all(np.isfinite(ll)):
        return -np.inf

    return ll + lp


##############################################################################
# The main worker that does it all


def run_it_all(c, name, h_phi1=0.75*u.deg,
               nwalkers=64, nburn=512, nsteps=1024,
               progress=True, overwrite=False, pool=None):

    c_pal5 = {'lead': c.transform_to(pal5_lead_frame),
              'trail': c.transform_to(pal5_trail_frame)}

    Xs = {}
    for k in c_pal5.keys():
        Xs[k] = np.stack((c_pal5[k].phi1.wrap_at(180*u.deg).degree,
                          c_pal5[k].phi2.degree)).T

    _h_phi1 = h_phi1.to_value(u.degree)
    all_phi1_bins = np.arange(0, 20.+1e-3, _h_phi1)

    caches = {}
    cache_path = 'cache'
    os.makedirs(cache_path, exist_ok=True)
    for k in Xs.keys():
        cache = dict()
        cache['N'] = []
        cache['phi1_c'] = []
        print("running measurements for '{}'".format(k))

        X = Xs[k]
        phi1_bins = all_phi1_bins[all_phi1_bins < X[:, 0].max()]

        for i, l, r in tqdm(zip(range(len(phi1_bins)), phi1_bins[:-1],
                                phi1_bins[1:]), total=len(phi1_bins[1:])):
            # Make overlapping bins by expanding the right boundary:
            phi1_mask = (X[:, 0] > l) & (X[:, 0] <= (r + _h_phi1))
            phi2_mask, phi2_lim = get_phi2_mask(X[phi1_mask])
            binX = X[phi1_mask][phi2_mask]
            if len(binX) < 1:
                print("skipping bin {} at phi1={}".format(i, 0.5*(l+r)))
                continue
            cache['N'].append(len(binX))
            cache['phi1_c'].append(0.5 * (l + r))

            sampler_file = path.join(cache_path,
                                     '{}_{}_{:02d}.pkl'.format(name, k, i))
            if not path.exists(sampler_file) or overwrite:
                sampler = run_sampler(binX, phi2_lim,
                                      nwalkers, nburn, nsteps,
                                      pool=pool)
                sampler.pool = None
                with open(sampler_file, 'wb') as f:
                    pickle.dump(sampler, f)

            with open(sampler_file, 'rb') as f:
                sampler = pickle.load(f)

            stuff = unpack_chain(sampler.chain)
            for par_name in stuff:
                if par_name not in cache:
                    cache[par_name] = []
                cache[par_name].append(stuff[par_name])

        cache = Table(cache)
        caches[k] = cache

    return caches
