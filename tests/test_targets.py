from astropy.time import Time

from spherex.targets import WISE_0855


def test_wise0855_moves_substantially_by_spherex_epoch():
    start = WISE_0855.at(WISE_0855.reference_time)
    later = WISE_0855.at(Time("2025-07-01"))
    assert start.separation(later).arcsec > 80
    assert later.ra.deg < start.ra.deg


def test_uncertainty_grows_from_astrometric_errors():
    at_reference = WISE_0855.uncertainty_arcsec(WISE_0855.reference_time)
    later = WISE_0855.uncertainty_arcsec(Time("2026-01-01"))
    assert at_reference > 0
    assert later > at_reference


def test_published_declination_motion_has_positive_sign():
    start = WISE_0855.at(WISE_0855.reference_time)
    later = WISE_0855.at(Time("2026-01-01"))
    assert WISE_0855.pm_dec_mas_yr == 680.0
    assert later.dec.deg > start.dec.deg
