import os
from os import path
import sys

# Third-party
import astropy.coordinates as coord
from astropy.table import Table, vstack
from astropy.io import fits, ascii
import astropy.units as u
import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import truncnorm
from schwimmbad import MultiPool, SerialPool

import gala.coordinates as gc
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic

# This project:
from potential import default_mw
from coordinates import (pal5_c, galcen_frame,
                         pal5_lead_frame, pal5_trail_frame)

from potential import get_bar_model
from coordinates import pal5_M

plot_path = path.abspath('../plots')
os.makedirs(plot_path, exist_ok=True)

def morphology(omega, m_b,
               release_every=1, n_particles=1,
               dt=-1, n_steps=6000):
    """
    Takes the pattern speed (with units, in km/s/kpc), and release_every, which controls how
    often to release particles, and the bar mass m_b. Creates mock streams,
    returns RA, dec to be used for track calculation
    """

    S = np.load('../data/Sn9l19m.npy') #expansion coeff.
    S = S[:2, :6, :6]

    w0 = gd.PhaseSpacePosition(pal5_c.transform_to(galcen_frame).cartesian)

    pot = gp.CCompositePotential()
    pot['disk'] = default_mw['disk']
    pot['halo'] = default_mw['halo']
    pot['bar'] = get_bar_model(Omega=omega, Snlm=S, m=m_b)

    frame = gp.ConstantRotatingFrame(Omega=[0,0,-1] * omega, units=galactic)
    H = gp.Hamiltonian(pot, frame) #frame such that we're "moving" with respect to bar
    df = gd.FardalStreamDF(random_state=np.random.RandomState(42))

    prog_pot = gp.PlummerPotential(pal5_M, 4*u.pc, units=galactic)
    gen = gd.MockStreamGenerator(df=df, hamiltonian=H,
                                 progenitor_potential=prog_pot)
    # gen = gd.MockStreamGenerator(df=df, hamiltonian=H)
    stream_data, _ = gen.run(w0, pal5_M,
                             dt=dt, n_steps=n_steps,
                             release_every=release_every,
                             n_particles=n_particles)

    stream_data.to_hdf5(path.join(plot_path, 'BarModels_RL{:d}_Mb{:.0e}_Om{:.1f}.hdf5'.format(release_every, m_b.value, omega.value)))
    sim_c = stream_data.to_coord_frame(coord.ICRS,
                                       galactocentric_frame=galcen_frame)
    return sim_c


def lnnormal(x, mu, std):
    return -0.5 * (x-mu)**2 / std**2 - 0.5*np.log(2*np.pi) - np.log(std)

def ln_truncnorm(x, mu, sigma, clip_a, clip_b):
    a, b = (clip_a - mu) / sigma, (clip_b - mu) / sigma
    return truncnorm.logpdf(x, a, b, loc=mu, scale=sigma)

# 1 gaussian:
param_names = ['mu_s', 'lnstd_s']

def lnprior(p):
    mu_s, lnstd_s = p

    lp = 0

    lp += ln_truncnorm(mu_s, 0, 1, -1, 1)
    lp += ln_truncnorm(lnstd_s, -0.5, 1, -2.5, 1.5)

    return lp

def lnlike(p, phi2):
    mu_s, lnstd_s = p
    return lnnormal(phi2, mu_s, np.exp(lnstd_s))

def lnprob(p, phi2):
    lp = lnprior(p)
    if not np.all(np.isfinite(lp)):
        return -np.inf

    ll = lnlike(p, phi2).sum()
    if not np.all(np.isfinite(ll)):
        return -np.inf

    return ll + lp

lead_tbl = Table.read('../data/pal5_lead_samples.fits')
bin_mask_le = np.median(lead_tbl['a'] * lead_tbl['N'][:, None], axis=1) > 50.

trail_tbl = Table.read('../data/pal5_trail_samples.fits')
bin_mask_tr = np.median(trail_tbl['a'] * trail_tbl['N'][:, None], axis=1) > 50.

def width_track(omega, m_b, release_every=1, n_particles=1, **kwargs):
    """Function takes in an array of unitless pattern speeds, an integer of released
    particles and unitless a bar_mass, then calls stream generating function, and
    calculates the stream width and track from the simulated streams.
    The function returns width, track and morhpology of streams in separate plots. """
    print(omega)
    c = morphology(omega, m_b,
                   release_every=release_every,
                   n_particles=n_particles,
                   **kwargs)

    print('done morphology')

    c_l = c.transform_to(pal5_lead_frame)
    c_t = c.transform_to(pal5_trail_frame)

    Xl = np.stack((c_l.phi1.wrap_at(180*u.deg).degree,c_l.phi2.degree)).T
    Xt = np.stack((c_t.phi1.wrap_at(180*u.deg).degree,c_t.phi2.degree)).T


    phi1_bins = np.arange(0, 18+1e-3, 0.75)
    phi2_bins = np.arange(-2, 2+1e-3, 0.1)

    nwalkers = 64
    nburn = 128
    nsteps = 256

    phi2_min = -2.
    phi2_max = 2.

    data = dict()
    for name, X, _phi1_bins in zip(['lead', 'trail'],
                                   [Xl, Xt],
                                   [phi1_bins[:14], phi1_bins[:23]]):
        phi2_mask = (X[:, 1] > phi2_min) & (X[:, 1] < phi2_max)

        all_samplers = []
        Ns = []
        for i, l, r in zip(range(len(phi1_bins)-1),
                           _phi1_bins[:-1],
                           _phi1_bins[1:]):

            phi1_mask = (X[:, 0] > l) & (X[:, 0] <= r)
            binX = X[phi1_mask & phi2_mask]
            Ns.append((phi1_mask & phi2_mask).sum())

            H, _ = np.histogram(binX[:, 1], bins=phi2_bins)
            phi2_bin_c = 0.5*(phi2_bins[:-1]+phi2_bins[1:])
            mu = phi2_bin_c[H.argmax()]
            if np.abs(mu) > 1.:
                mu = 0.

        # for 1-component gaussian
            p0 = (mu, np.log(0.1))

            p0s = emcee.utils.sample_ball(p0, [1e-3]*len(p0), nwalkers)

            sampler = emcee.EnsembleSampler(nwalkers, len(p0),
                                            log_prob_fn=lnprob,
                                            args=(binX[:, 1], ))

            pos,*_ = sampler.run_mcmc(p0s, nburn, progress=False)
            pos = emcee.utils.sample_ball(np.median(pos, axis=0),
                                          [1e-3]*len(p0), nwalkers)

            sampler.reset()
            pos,*_ = sampler.run_mcmc(pos, nburn, progress=False)
            sampler.reset()
            _ = sampler.run_mcmc(pos, nsteps, progress=False)

            all_samplers.append(sampler)

        data[name] = dict()
        data[name]['X'] = X
        data[name]['samplers'] = all_samplers
        data[name]['phi1_bins'] = _phi1_bins
        data[name]['N'] = np.array(Ns)

    flatchains = dict()
    for name in data:
        all_flatchains = []
        for sampler in data[name]['samplers']:
            all_flatchains.append(sampler.flatchain)

        all_flatchains = np.array(all_flatchains)

        flatchains[name] = Table()
        for k, par in enumerate(param_names):
            if par.startswith('ln_'):
                flatchains[name][par] = all_flatchains[..., k]
                flatchains[name][par[3:]] = np.exp(all_flatchains[..., k])
            elif par.startswith('ln'):
                flatchains[name][par] = all_flatchains[..., k]
                flatchains[name][par[2:]] = np.exp(all_flatchains[..., k])
            else:
                flatchains[name][par] = all_flatchains[..., k]

        phi1_bin_c = 0.5 * (data[name]['phi1_bins'][:-1] + data[name]['phi1_bins'][1:]) * u.deg
        flatchains[name]['phi1_bin_c'] = phi1_bin_c
        flatchains[name]['N'] = data[name]['N']

    med_le_w = np.median(lead_tbl['std_s'], axis=1)
    err1_le_w = med_le_w - np.percentile(lead_tbl['std_s'], 16, axis=1)
    err2_le_w = np.percentile(lead_tbl['std_s'], 84, axis=1) - med_le_w

    med_tr_w = np.median(trail_tbl['std_s'], axis=1)
    err1_tr_w = med_tr_w - np.percentile(trail_tbl['std_s'], 16, axis=1)
    err2_tr_w = np.percentile(trail_tbl['std_s'], 84, axis=1) - med_tr_w

    med_le_t = np.median(lead_tbl['mu_s'], axis=1)
    err1_le_t = med_le_t - np.percentile(lead_tbl['mu_s'], 16, axis=1)
    err2_le_t = np.percentile(lead_tbl['mu_s'], 84, axis=1) - med_le_t

    med_tr_t = np.median(trail_tbl['mu_s'], axis=1)
    err1_tr_t = med_tr_t - np.percentile(trail_tbl['mu_s'], 16, axis=1)
    err2_tr_t = np.percentile(trail_tbl['mu_s'], 84, axis=1) - med_tr_t

    #Now plotting everything
    mpl.rcParams.update({'font.size': 24})
    label_size = 20
    mpl.rcParams['xtick.labelsize'] = 18#label_size
    mpl.rcParams['ytick.labelsize'] = 18#label_size
    fig,axes = plt.subplots(1, 3,figsize=(21,7))#), sharex=True)

    #stream morphology
    axes[0].scatter(c.ra.degree, c.dec.degree,
                    s=0.01, color='black',
                    label='Mb = {:.0e} $\Omega_b = ${:.1f}'.format(m_b, omega),
                    marker='.', rasterized=True)
    axes[0].set_xlim(250,215)
    axes[0].set_ylabel('Dec [deg]')
    axes[0].set_yticks([-10,-5,0,5, 10])
    axes[0].set_ylim(-10,10)
    axes[0].set_aspect('equal')
    axes[0].legend(fontsize=12, loc='upper right')



    for name in data.keys():
        flatchain = flatchains[name]
        med = np.median(flatchain['std_s'], axis=1)
        axes[1].plot(flatchains[name]['phi1_bin_c'], med, linestyle='--', label='sim: '+name)

    axes[1].errorbar(lead_tbl['phi1_bin_c'][bin_mask_le],
                     med_le_w[bin_mask_le],
                     yerr=(err1_le_w[bin_mask_le], err2_le_w[bin_mask_le]),
                     marker='o', ls='none',ecolor='steelblue', color='steelblue',label='data: lead')

    axes[1].errorbar(trail_tbl['phi1_bin_c'][bin_mask_tr],
                     med_tr_w[bin_mask_tr],
                     yerr=(err1_tr_w[bin_mask_tr], err2_tr_w[bin_mask_tr]),
                     marker='o', ls='none', ecolor='orange', color='orange',label='data: trail')


    axes[1].set_xlim(0, 17)
    axes[1].set_ylim(0, 0.6)
    axes[1].set_xlabel(r'$\Delta \phi_1$ [deg]')
    axes[1].set_ylabel(r'$\sigma$ [deg]')
    axes[1].legend(loc='best', fontsize=15)


    for name in data.keys():
        flatchain = flatchains[name]
        med = np.median(flatchain['mu_s'], axis=1)
        axes[2].plot(flatchain['phi1_bin_c'], med, linestyle='--', label='sim: '+name)


    axes[2].errorbar(lead_tbl['phi1_bin_c'][bin_mask_le],
                     med_le_t[bin_mask_le],
                     yerr=(err1_le_t[bin_mask_le], err2_le_t[bin_mask_le]),
                     marker='o', ls='none',ecolor='steelblue', color='steelblue',label='data: lead')



    axes[2].errorbar(trail_tbl['phi1_bin_c'][bin_mask_tr],
                    med_tr_t[bin_mask_tr],
                    yerr=(err1_tr_t[bin_mask_tr], err2_tr_t[bin_mask_tr]),
                    marker='o', ls='none', ecolor='orange', color='orange',label='data: trail')


    axes[2].set_xlim(0, 17)
    axes[2].set_ylim(-1, 1)
    axes[2].set_xlabel(r'$\Delta \phi_1$ [deg]')
    axes[2].set_ylabel('$\Delta \phi_2$ [deg]')
    axes[2].legend(loc='best', fontsize=15)

    fig.tight_layout()
    fig.savefig(path.join(plot_path,
                          'BarModels_RL{:d}_Mb{:.0e}_Om{:.1f}.png'
                          .format(release_every, m_b.value, omega.value)))


def worker(task):
    omega, = task
    width_track(omega*u.km/u.s/u.kpc, m_b=1e10*u.Msun,
                release_every=1, n_steps=6000)


tasks = [(om, ) for om in np.arange(25, 60+1e-3, 0.5)]

with SerialPool() as pool:
#with MultiPool() as pool:
    print(pool.size)
    for r in pool.map(worker, tasks):
        pass
