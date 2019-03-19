# Third-party packages
import astropy.units as u
import astropy.coordinates as coord
import gala.coordinates as gc

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
v_sun = coord.CartesianDifferential(v_lsr + [0, 1, 0] * v_circ)
galcen_frame = coord.Galactocentric(galcen_distance=8.1*u.kpc,
                                    galcen_v_sun=v_sun)


trail_a = [230, 242]
trail_d = [0.75, 6.6]
trail_epts = coord.SkyCoord(ra=trail_a*u.deg, dec=trail_d*u.deg)

lead_a = [228, 225]
lead_d = [-1.2, -4.3]
lead_epts = coord.SkyCoord(ra=lead_a*u.deg, dec=lead_d*u.deg)

pal5_lead_frame = gc.GreatCircleICRSFrame.from_endpoints(lead_epts[0], lead_epts[1],
                                                         ra0=pal5_c.ra)
pal5_trail_frame = gc.GreatCircleICRSFrame.from_endpoints(trail_epts[0], trail_epts[1],
                                                          ra0=pal5_c.ra)
