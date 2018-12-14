import numpy as np
import scipy.integrate as si
from schwimmbad import SerialPool

# Project
from biff.scf._computecoeff import Snlm_integrand, Tnlm_integrand, STnlm_discrete, STnlm_var_discrete

__all__ = ['compute_Snlm']


def worker(nlm):
    n, l, m = nlm
    
    try:
        Snlm, Snlm_e = si.nquad(
            Snlm_integrand, ranges=limits,
            args=(density_func, n, l, m, M, r_s, args),
            opts=nquad_opts)
    except Exception as e:
        return np.nan, np.nan
    
    return Snlm, Snlm_e

def compute_Snlm(density_func, nmax, lmax, M, r_s, args=(),
                 progress=False, pool=None, **nquad_opts):
    """
    Compute the expansion coefficients for representing the input density function using a basis
    function expansion.

    Computing the coefficients involves computing triple integrals which are computationally
    expensive. For an example of how to parallelize the computation of the coefficients, see
    ``examples/parallel_compute_Anlm.py``.

    Parameters
    ----------
    density_func : function, callable
        A function or callable object that evaluates the density at a given position. The call
        format must be of the form: ``density_func(x, y, z, M, r_s, args)`` where ``x,y,z`` are
        cartesian coordinates, ``M`` is a scale mass, ``r_s`` a scale radius, and ``args`` is an
        iterable containing any other arguments needed by the density function.
    nmax : int
        Maximum value of ``n`` for the radial expansion.
    lmax : int
        Maximum value of ``l`` for the spherical harmonics.
    M : numeric
        Scale mass.
    r_s : numeric
        Scale radius.
    args : iterable (optional)
        A list or iterable of any other arguments needed by the density
        function.
    skip_odd : bool (optional)
        Skip the odd terms in the angular portion of the expansion. For example, only
        take :math:`l=0,2,4,...`
    skip_even : bool (optional)
        Skip the even terms in the angular portion of the expansion. For example, only
        take :math:`l=1,3,5,...`
    skip_m : bool (optional)
        Ignore terms with :math:`m > 0`.
    S_only : bool (optional)
        Only compute the S coefficients.
    **nquad_opts
        Any additional keyword arguments are passed through to
        `~scipy.integrate.nquad` as options, `opts`.

    Returns
    -------
    Snlm : float, `~numpy.ndarray`
        The value of the cosine expansion coefficient.
    Snlm_err : , `~numpy.ndarray`
        An estimate of the uncertainty in the coefficient value (from `~scipy.integrate.nquad`).
    Tnlm : , `~numpy.ndarray`
        The value of the sine expansion coefficient.
    Tnlm_err : , `~numpy.ndarray`
        An estimate of the uncertainty in the coefficient value. (from `~scipy.integrate.nquad`).

    """
    lmin = 0
    lstride = 2 # skip odd terms

    Snlm = np.zeros((nmax+1, lmax+1, lmax+1))
    Snlm_e = np.zeros((nmax+1, lmax+1, lmax+1))
    
    nlms = []
    for n in range(nmax+1):
        for l in range(lmin, lmax+1, lstride):
            for m in range(l+1):
                nlms.append((n,l,m))
                
    print("computing {0} coefficients".format(len(nlms)))

    nquad_opts.setdefault('limit', 256)
    nquad_opts.setdefault('epsrel', 1E-10)
    
    if progress:
        if pool is not None and not isinstance(pool, SerialPool):
            raise ValueError("progress bars can't be used with "
                             "multiprocessing pools."
        
        from tqdm import tqdm
        iterfunc = tqdm
                             
    elif pool is not None:
        iterfunc = pool.map
    else:
        iterfunc = lambda x: x

    limits = [[0, 2*np.pi], # phi
              [-1, 1.], # X (cos(theta))
              [-1, 1.]] # xsi
        
    for n, l, m in iterfunc(nlms):
        

    return (Snlm, Snlm_e)