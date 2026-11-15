"""Backend astronomical target definitions."""

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
    name="WISE 0855 (Brown Dwarf)",
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

BARNARDS_STAR = Target(
    key="barnard",
    name="Barnard's Star (High PM)",
    ra_deg=269.452075,
    dec_deg=4.693392,
    reference_time=Time(57205.0, format="mjd", scale="utc"),
    pm_ra_cosdec_mas_yr=-798.58,
    pm_dec_mas_yr=10328.12,
    parallax_mas=546.98,
    position_sigma_mas=1.0,
    pm_ra_sigma_mas_yr=0.5,
    pm_dec_sigma_mas_yr=0.5,
    parallax_sigma_mas=0.5,
    source_url="https://doi.org/10.1051/0004-6361/201832964",
    source_note="Gaia DR2 astrometry; highest known proper motion star (~10.3 arcsec/yr).",
)

EPSILON_INDI_B = Target(
    key="eps_indi_b",
    name="Epsilon Indi B (Brown Dwarf)",
    ra_deg=330.8402,
    dec_deg=-56.78597,
    reference_time=Time(53000.0, format="mjd", scale="utc"),
    pm_ra_cosdec_mas_yr=3961.4,
    pm_dec_mas_yr=-2538.3,
    parallax_mas=275.8,
    position_sigma_mas=20.0,
    pm_ra_sigma_mas_yr=5.0,
    pm_dec_sigma_mas_yr=5.0,
    parallax_sigma_mas=2.0,
    source_url="https://doi.org/10.1051/0004-6361:20031358",
    source_note="Scholz et al. (2003); nearby brown dwarf binary pair orbiting Epsilon Indi (4.7 arcsec/yr).",
)

RW_AURIGAE = Target(
    key="rw_aur",
    name="RW Aurigae (Variable T Tauri)",
    ra_deg=77.1064,
    dec_deg=30.4014,
    reference_time=Time(57205.0, format="mjd", scale="utc"),
    pm_ra_cosdec_mas_yr=1.2,
    pm_dec_mas_yr=-20.5,
    parallax_mas=6.1,
    position_sigma_mas=1.0,
    pm_ra_sigma_mas_yr=0.5,
    pm_dec_sigma_mas_yr=0.5,
    parallax_sigma_mas=0.5,
    source_url="https://doi.org/10.1051/0004-6361/201832964",
    source_note="Classical T Tauri eruptive variable binary showing occultation-driven infrared dimming.",
)

R_DRACONIS = Target(
    key="r_dra",
    name="R Draconis (Mira Variable)",
    ra_deg=248.1677,
    dec_deg=66.7529,
    reference_time=Time(57205.0, format="mjd", scale="utc"),
    pm_ra_cosdec_mas_yr=-12.5,
    pm_dec_mas_yr=22.4,
    parallax_mas=1.5,
    position_sigma_mas=1.0,
    pm_ra_sigma_mas_yr=0.5,
    pm_dec_sigma_mas_yr=0.5,
    parallax_sigma_mas=0.5,
    source_url="https://doi.org/10.1051/0004-6361/201832964",
    source_note="Mira-type high-amplitude pulsating red giant with ~245-day cycle.",
)

TARGETS = {
    WISE_0855.key: WISE_0855,
    BARNARDS_STAR.key: BARNARDS_STAR,
    EPSILON_INDI_B.key: EPSILON_INDI_B,
    RW_AURIGAE.key: RW_AURIGAE,
    R_DRACONIS.key: R_DRACONIS,
}


def get_target(key: str) -> Target:
    try:
        return TARGETS[key.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown target {key!r}; choose from {', '.join(sorted(TARGETS))}") from exc
