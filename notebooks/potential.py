from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as u
from scipy.optimize import minimize

import gala.potential as gp
from gala.units import galactic


mw = gp.BovyMWPotential2014()
default_disk_bulge = gp.CCompositePotential()
default_disk_bulge['disk'] = mw['disk']
default_disk_bulge['bulge'] = mw['bulge']


def corot_func(r_cr, Omega, disknobar):
    vc = disknobar.circular_velocity([r_cr, 0., 0.])
    return abs(vc - Omega*r_cr * u.kpc).decompose().value[0]


def get_bar_model(Omega, Snlm, alpha=-27*u.deg, disknobar=None):
    if disknobar is None:
        disknobar = default_disk_bulge

    res = minimize(corot_func, x0=4., args=(Omega, disknobar))
    r_cr = res.x[0]
    r_s = r_cr / 3.2 # 3.2 scales this to the value WZ2012 use (60 km/s/kpc)

    return gp.SCFPotential(m=5e9 / 10., r_s=r_s, # 10 is a MAGIC NUMBER: believe
                           Snlm=Snlm,
                           units=galactic,
                           R=rotation_matrix(alpha, 'z'))
