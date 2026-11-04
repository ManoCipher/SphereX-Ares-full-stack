from __future__ import annotations

from dataclasses import dataclass

from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation,
    Distance,
    SkyCoord,
    get_body_barycentric,
)
from astropy.time import Time


@dataclass(frozen=True)
class Target:
    key: str
    name: str
    ra_deg: float
    dec_deg: float
    reference_time: Time
    pm_ra_cosdec_mas_yr: float
    pm_dec_mas_yr: float
    parallax_mas: float
    position_sigma_mas: float
    pm_ra_sigma_mas_yr: float
    pm_dec_sigma_mas_yr: float
    parallax_sigma_mas: float
    source_url: str
    source_note: str

    @property
    def coordinate(self) -> SkyCoord:
        return SkyCoord(
            ra=self.ra_deg * u.deg,
            dec=self.dec_deg * u.deg,
            distance=Distance(parallax=self.parallax_mas * u.mas),
            pm_ra_cosdec=self.pm_ra_cosdec_mas_yr * u.mas / u.yr,
            pm_dec=self.pm_dec_mas_yr * u.mas / u.yr,
            radial_velocity=0 * u.km / u.s,
            obstime=self.reference_time,
            frame="icrs",
        )

    def at(self, time: Time | str | float) -> SkyCoord:
        """Propagate the barycentric ICRS position by proper motion.

        A scalar float is interpreted as MJD.  The source position is a measured
        sky position at its stated epoch, not a fictitious "J2000 epoch".
        Use :meth:`at_observer` for an observer-dependent apparent direction.
        """
        destination = time if isinstance(time, Time) else Time(time, format="mjd") if isinstance(time, (int, float)) else Time(time)
        return self.coordinate.apply_space_motion(new_obstime=destination)

    def at_observer(
        self,
        time: Time | str | float,
        observer_geocentric: CartesianRepresentation | None = None,
    ) -> SkyCoord:
        """Return the proper-motion and annual-parallax corrected ICRS direction.

        Astropy's built-in solar-system ephemeris supplies the Earth's
        barycentric position.  A geocentric spacecraft offset can be included
        when a FITS product supplies a reliable Cartesian position.  Light-time
        and perspective acceleration are negligible at the precision of these
        SPHEREx cutouts.
        """
        destination = time if isinstance(time, Time) else Time(time, format="mjd") if isinstance(time, (int, float)) else Time(time)
        moved = self.at(destination)
        observer = get_body_barycentric("earth", destination.tdb)
        if observer_geocentric is not None:
            observer = observer + observer_geocentric
        relative = moved.icrs.cartesian.without_differentials() - observer
        return SkyCoord(relative, frame="icrs", obstime=destination)

    def uncertainty_arcsec(self, time: Time | str | float) -> float:
        destination = time if isinstance(time, Time) else Time(time, format="mjd") if isinstance(time, (int, float)) else Time(time)
        years = abs((destination - self.reference_time).to_value(u.yr))
        pm_sigma = (self.pm_ra_sigma_mas_yr**2 + self.pm_dec_sigma_mas_yr**2) ** 0.5 * years
        random_mas = (self.position_sigma_mas**2 + pm_sigma**2 + self.parallax_sigma_mas**2) ** 0.5
        return random_mas / 1000.0


# Position measured on 2014-11-25 (MJD 56986.817), precision 0.05 arcsec,
# from Beamín et al. 2016, A&A 592 A80, Table 1.  The proper motion and
# parallax are from Luhman & Esplin 2016, AJ 152, 78, Table 4.
WISE_0855 = Target(
    key="wise0855",
    name="WISE J085510.83-071442.5",
    ra_deg=133.7851375,  # 08:55:08.433
    dec_deg=-7.2443028,  # -07:14:39.49
    reference_time=Time(56986.817, format="mjd", scale="utc"),
    pm_ra_cosdec_mas_yr=-8118.0,
    pm_dec_mas_yr=680.0,
    parallax_mas=449.0,
    position_sigma_mas=50.0,
    pm_ra_sigma_mas_yr=8.0,
    pm_dec_sigma_mas_yr=7.0,
    parallax_sigma_mas=8.0,
    source_url="https://doi.org/10.3847/0004-6256/152/3/78",
    source_note=(
        "Position measured at MJD 56986.817 by Beamin et al. (2016); "
        "proper motion and parallax from Luhman & Esplin (2016), Table 4."
    ),
)

TARGETS = {WISE_0855.key: WISE_0855}


def get_target(key: str) -> Target:
    try:
        return TARGETS[key.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown target {key!r}; choose from {', '.join(sorted(TARGETS))}") from exc
