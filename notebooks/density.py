# Third-party packages
import astropy.coordinates as coord
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as u
from astropy.wcs import WCS
import gala.coordinates as gc
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

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
    offset_fr = coord.SkyOffsetFrame(origin=ref_c)
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


class StreamSurfaceDensityModel:

    def __init__(self, data_c, ref_c,
                 lon_limits=None, lat_limits=None,
                 poly_deg=5):
        """TODO:

        Parameters
        ----------
        data_c : `astropy.coordinates.SkyCoord`
            TODO:
        ref_c : `astropy.coordinates.SkyCoord`
            The coordinates of a reference point along the stream. For streams
            with a progenitor, this could be the coordinates of the cluster,
            e.g., Pal 5.
        K : int
            The number of nodes to place along the stream.
        h : float
            The bandwidth parameter, or the along-stream width (standard
            deviation) of each 2D node Gaussian.
        lon_limits : iterable, optional
            TODO

        """
        data_c = coord.SkyCoord(data_c)

        if lon_limits is None:
            lon_limits = [-180*u.deg, 180*u.deg]
        if lat_limits is None:
            lat_limits = [-90*u.deg, 90*u.deg]

        data_lon = data_c.spherical.lon.wrap_at(180*u.deg)
        data_lat = data_c.spherical.lat
        data_mask = ((data_lon > lon_limits[0]) &
                     (data_lon < lon_limits[1]) &
                     (data_lat > lat_limits[0]) &
                     (data_lat < lat_limits[1]))
        self.data_c = data_c[data_mask]

        self._frame = self.data_c.frame
        self.ref_c = coord.SkyCoord(ref_c.transform_to(self._frame))

        # Projected coordinates
        self.proj_xy = get_projected_coords(self.data_c, self.ref_c)

        # Fit a polynomial to the projected coordinates
        self.poly = np.poly1d(np.polyfit(self.proj_xy[0],
                                         self.proj_xy[1],
                                         deg=poly_deg))

        # other cached things:
        self._data_lon = self.data_c.spherical.lon.wrap_at(180*u.deg)
        self._data_lat = self.data_c.spherical.lat
        self._nodes = None
        self.K = None

    def get_dense_poly_track(self, size=None, xgrid=None):
        if ((xgrid is None and size is None) or
                (xgrid is not None and size is not None)):
            raise ValueError('You must either pass in a `size` for the poly '
                             'grid, or pass in `xgrid`.')

        if xgrid is None:
            xgrid = np.linspace(self._data_lon.value.min(),
                                self._data_lon.value.max(),
                                size)

        return np.stack((xgrid, self.poly(xgrid)))

    def set_nodes(self, spacing=None, K=None, dense_poly_size=10000):
        if K is not None and spacing is not None:
            raise ValueError('Set either spacing or K, not both!')

        if K is not None:
            raise NotImplementedError('So far we only support passing in `spacing`')

        track = self.get_dense_poly_track(size=dense_poly_size)
        idx = get_uniform_idx(track[0], track[1], spacing=spacing)
        self._nodes = track[:, idx]
        self.K = len(idx)

        # cache the u_k's at each node
        self._u_k = np.zeros((2, self.K))
        for i, k in enumerate(idx):
            self._u_k[:, i] = get_u_v(k, track[0], track[1],
                                      self.poly, eps=1e-3)[0]

        return self._nodes

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

        theta = np.arctan2(self._u_k[1, k], self._u_k[0, k])

        # rotation from u,v to x,y (sky-tangent-plane, area conserved)
        R = np.zeros([2, 2])
        R[0, 0] = R[1, 1] = np.cos(theta)
        R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta)

        # rotate the covariance matrix to the x-y space
        Ck = R @ Ck_prim @ R.T
        return Ck

    def ln_density(self, xy, a_k, s_k, h):
        ln_dens = np.zeros((self.K, len(xy)))
        for k in range(self.K):
            C = self.get_Ck(k, s_k[k], h)
            ln_dens[k] = multivariate_normal.logpdf(xy, self._nodes[:, k], C)
        return logsumexp(ln_dens + np.log(a_k)[:, None], axis=0)
