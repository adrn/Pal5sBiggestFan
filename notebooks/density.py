# Third-party packages
import astropy.coordinates as coord
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as u
from astropy.wcs import WCS
import gala.coordinates as gc
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn

from utilities import get_uniform_idx


def get_projected_coords(c, ref_c):
    """
    Parameters
    ----------
    c : `astropy.coordinates.SkyCoord`
        The coordinates that you want to project.
    ref_c : `astropy.coordinates.SkyCoord`
        The reference coordinates, e.g., could be of the progenitor like in
        the case of Pal 5.

    Returns
    -------
    xy : `numpy.ndarray`
        The projected (x,y) coordinates.

    """
    c = coord.SkyCoord(c)
    offset_fr = coord.SkyOffsetFrame(origin=ref_c.transform_to(c.frame))
    c2 = c.transform_to(offset_fr)

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [0., 0.]
    wcs.wcs.cdelt = [1., 1.]
    wcs.wcs.crval = [0, 0]
    wcs.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]

    return np.stack(wcs.wcs_world2pix(c2.lon.degree, c2.lat.degree, 0))


def get_tail_masks(c, pal5_c, pal5_frame=None):
    """Return boolean arrays to select the leading and trailing stars.

    Parameters
    ----------
    c : `astropy.coordinates.SkyCoord`
        The coordinates of stars or star particles to classify.
    pal5_c : `astropy.coordinates.SkyCoord`
        The coordinates of Pal 5 (the cluster).

    Returns
    -------
    trailing_mask : `numpy.ndarray`
        A boolean array for selecting trailing tail stars.
    leading_mask : `numpy.ndarray`
        A boolean array for selecting leading tail stars.
    """
    if pal5_frame is None:
        pal5_frame = gc.Pal5()

    pal5_c_pal5 = pal5_c.transform_to(pal5_frame)
    c_pal5 = c.transform_to(pal5_frame)

    phi1 = c_pal5.phi1.wrap_at(180*u.deg)
    phi1_pal5 = pal5_c_pal5.phi1.wrap_at(180*u.deg)

    trail_mask = phi1 < phi1_pal5
    return trail_mask, np.logical_not(trail_mask)


def get_u_v(k, dense_x, dense_y, poly, eps=1e-3):
    """Given a node index k, a dense grid of polynomial points, and
    numpy polynomial, return the tangent and normal vectors (u, v) at
    the node location.

    Parameters
    ----------
    k : int
    dense_x : array_like
    dense_y : array_like
    poly : `numpy.poly1d`
    """
    # Convention: set v direction to point in the direction of positive curvature
    deriv2_at_node = poly.deriv(m=2)(dense_x[k])
    sign = np.sign(deriv2_at_node)

    dx = eps
    dy = poly.deriv(m=1)(dense_x[k]) * dx
    u_vec = np.array([dx, dy])
    u_vec /= np.linalg.norm(u_vec)

    R = rotation_matrix(sign*90*u.deg, 'z')[:2, :2]
    v_vec = R @ u_vec

    return u_vec, v_vec


class DensityModel2D:

    def __init__(self, X, poly_deg=5):
        """TODO:

        Parameters
        ----------

        """

        # Projected coordinates
        self.X = X # (N, D)
        self.N, self.D = self.X.shape

        if self.D != 2:
            raise NotImplementedError('D={0}'.format(self.D))

        # Fit a polynomial to the projected coordinates
        self.poly = np.poly1d(np.polyfit(self.X[:, 0], self.X[:, 1],
                                         deg=poly_deg))

        self.nodes = None
        self.K = None

    def get_dense_poly_track(self, size=None, xgrid=None):
        if ((xgrid is None and size is None) or
                (xgrid is not None and size is not None)):
            raise ValueError('You must either pass in a `size` for the poly '
                             'grid, or pass in `xgrid`.')

        if xgrid is None:
            xgrid = np.linspace(self.X[:, 0].min(),
                                self.X[:, 0].max(),
                                size)

        track = np.stack((xgrid, self.poly(xgrid))).T
        return track # (many, D)

    def set_nodes(self, track, nodes=None, spacing=None):
        if nodes is not None and spacing is not None:
            raise ValueError('Set either spacing or nodes, not both!')

        if nodes is not None:
            self.nodes = np.array(nodes)
            self.K = len(self.nodes)

            idx = []
            for k in range(self.K):
                i = np.linalg.norm(track - self.nodes[k][None], axis=1).argmin()
                idx.append(i)
            idx = np.array(idx)

        else:
            idx = get_uniform_idx(track[:, 0], track[:, 1],
                                  spacing=spacing)
            self.nodes = track[idx]
            self.K = len(idx)

        # cache the u_k's at each node
        self._u_k = np.zeros((self.K, self.D))
        self._R_k = np.zeros((self.K, self.D, self.D))
        for k, j in enumerate(idx):
            self._u_k[k] = get_u_v(j, track[:, 0], track[:, 1],
                                   self.poly, eps=1e-3)[0] # HACK: eps?
            self._R_k[k] = self._get_R(self._u_k[k])

        return self.nodes

    def _get_R(self, u_vec):
        # TODO: only works for 2D
        theta = np.arctan2(u_vec[1], u_vec[0])

        # rotation from u,v to x,y (sky-tangent-plane, area conserved)
        R = np.zeros([2, 2])
        R[0, 0] = R[1, 1] = np.cos(theta)
        R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta)

        return R

    def get_Ck(self, k, s_k, h):
        """Get the 2D covariance matrix in x-y coordinates at the node k.

        Parameters
        ----------
        u_k : array_like

        Given tangent normalized vector at a given node k,
        this function takes the "bandwidth", h, and the function, s_k,
        and returns the covarience matrix"""

        # 2x2 covariance matrix in the u-v plane
        Ck_prim = np.zeros([2, 2])
        Ck_prim[0, 0] = h ** 2
        Ck_prim[1, 1] = s_k ** 2

        # retrieve pre-computed matrix
        R = self._R_k[k]

        # rotate the covariance matrix to the x-y space
        Ck = R @ Ck_prim @ R.T
        return Ck

    def ln_density(self, X, a_k, s_k, h):
        ln_dens = np.zeros((self.K, len(X)))
        for k in range(self.K):
            C = self.get_Ck(k, s_k[k], h)
            try:
                ln_dens[k] = mvn.logpdf(X, self.nodes[k], C,
                                        allow_singular=True)
            except ValueError as e:
                raise e
        return logsumexp(ln_dens + np.log(a_k)[:, None], axis=0)
