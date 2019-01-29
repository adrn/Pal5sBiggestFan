# Third-party packages
import astropy.units as u
import astropy.coordinates as coord

# TODO: can someone add some references as comments for each of these
# measurements?
pal5_c = coord.SkyCoord(ra=229.018*u.degree, dec=-0.124*u.degree,
                        distance=22.9*u.kpc,
                        pm_ra_cosdec=-2.296*u.mas/u.yr,
                        pm_dec=-2.257*u.mas/u.yr,
                        radial_velocity=-58.7*u.km/u.s)

# Setting Sun's params - should update distance
v_lsr = [11.1, 24.0, 7.25]*u.km/u.s
v_circ = 220 * u.km/u.s
v_sun = coord.CartesianDifferential(v_lsr + [0, 1, 0]*v_circ)
galcen_frame = coord.Galactocentric(galcen_distance=8.1*u.kpc,
                                    galcen_v_sun=v_sun)
