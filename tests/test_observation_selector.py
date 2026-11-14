from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from spherex.observation_selector import ObservationSelectionService, SelectionMode


RA = 150.0
DEC = 2.0


def _obs(
    oid: str,
    detector: int,
    mjd: float,
    *,
    wavelength: float = 1.0,
    covered: bool = True,
    quality: float = 0.9,
):
    return {
        "observation_id": oid,
        "detector": str(detector),
        "observation_start_mjd": mjd,
        "observation_timestamp": f"MJD {mjd}",
        "target_wavelength_um": wavelength,
        "target_covered": covered,
        "quality_score": quality,
        "fits_url": f"https://example.test/{oid}.fits",
    }


def test_spectral_selects_one_per_detector_at_similar_times():
    records = []
    for detector in range(1, 7):
        records.extend(
            [_obs(f"near-{detector}", detector, 60000 + detector / 100),
             _obs(f"far-{detector}", detector, 60100 + detector)]
        )
    result = ObservationSelectionService().select(records, RA, DEC, SelectionMode.SPECTRAL)
    assert len(result.observations) == 6
    assert {record["detector"] for record in result.observations} == {"1", "2", "3", "4", "5", "6"}
    assert all(str(record["observation_id"]).startswith("near-") for record in result.observations)


def test_spectral_returns_fewer_when_detectors_are_missing():
    records = [_obs("d1", 1, 60000), _obs("d3", 3, 60000.1), _obs("d6", 6, 60000.2)]
    result = ObservationSelectionService().select(records, RA, DEC, "Spectral Comparison")
    assert [record["detector"] for record in result.observations] == ["1", "3", "6"]


def test_time_uses_one_detector_and_spreads_across_visits():
    records = [
        _obs("a", 2, 60000), _obs("b", 2, 60000.1),
        _obs("c", 2, 60100), _obs("d", 2, 60300),
        _obs("other", 5, 60000),
    ]
    result = ObservationSelectionService().select(records, RA, DEC, SelectionMode.TIME)
    assert {record["detector"] for record in result.observations} == {"2"}
    assert {record["observation_id"] for record in result.observations} >= {"a", "c", "d"}


def test_time_rejects_incompatible_wavelength_cluster():
    records = [
        _obs("one", 4, 60000, wavelength=1.0),
        _obs("two", 4, 60100, wavelength=1.01),
        _obs("incompatible", 4, 60300, wavelength=3.0),
    ]
    result = ObservationSelectionService().select(records, RA, DEC, SelectionMode.TIME)
    assert [record["observation_id"] for record in result.observations] == ["one", "two"]


def test_duplicate_records_are_returned_only_once():
    record = _obs("same", 1, 60000)
    result = ObservationSelectionService().select([record, dict(record)], RA, DEC, SelectionMode.SPECTRAL)
    assert len(result.observations) == 1


def test_incomplete_or_explicitly_false_coverage_is_rejected():
    unknown = _obs("unknown", 1, 60000)
    unknown.pop("target_covered")
    outside = _obs("outside", 2, 60000, covered=False)
    inside = _obs("inside", 3, 60000, covered=True)
    result = ObservationSelectionService().select([unknown, outside, inside], RA, DEC, SelectionMode.SPECTRAL)
    assert [record["observation_id"] for record in result.observations] == ["inside"]


def test_insufficient_time_observations_returns_available_records():
    result = ObservationSelectionService().select([_obs("only", 6, 60000)], RA, DEC, SelectionMode.TIME)
    assert [record["observation_id"] for record in result.observations] == ["only"]


def test_manual_selection_is_unchanged_and_limited_to_six():
    records = [_obs(str(index), 1, 60000 + index) for index in range(7)]
    service = ObservationSelectionService()
    result = service.select(records, RA, DEC, SelectionMode.MANUAL, manual_selection=records[:3])
    assert list(result.observations) == records[:3]
    with pytest.raises(ValueError, match="limited to 6"):
        service.select(records, RA, DEC, SelectionMode.MANUAL, manual_selection=records)


def test_local_fits_wcs_validation_rejects_target_outside(tmp_path):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.0, 10.0]
    wcs.wcs.cdelt = [-0.01, 0.01]
    wcs.wcs.crval = [RA, DEC]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    inside_path = tmp_path / "inside.fits"
    outside_path = tmp_path / "outside.fits"
    fits.PrimaryHDU(np.ones((20, 20)), header=wcs.to_header()).writeto(inside_path)
    outside_wcs = wcs.deepcopy()
    outside_wcs.wcs.crval = [RA + 10, DEC]
    fits.PrimaryHDU(np.ones((20, 20)), header=outside_wcs.to_header()).writeto(outside_path)
    records = [_obs("inside", 1, 60000), _obs("outside", 2, 60000)]
    paths = {"inside": inside_path, "outside": outside_path}
    service = ObservationSelectionService(path_resolver=lambda record: paths[record["observation_id"]])
    result = service.select(records, RA, DEC, SelectionMode.SPECTRAL)
    assert [record["observation_id"] for record in result.observations] == ["inside"]


def test_lightweight_validator_can_supply_coverage_wavelength_and_quality():
    record = _obs("remote", 3, 60000)
    record.pop("target_covered")
    record.pop("target_wavelength_um")
    calls = []

    def validate(candidate, ra, dec):
        calls.append((candidate["observation_id"], ra, dec))
        return {"covered": True, "wavelength_um": 2.2, "quality_ok": True, "quality_score": 0.8}

    result = ObservationSelectionService(remote_validator=validate).select(
        [record], RA, DEC, SelectionMode.SPECTRAL
    )
    assert [item["observation_id"] for item in result.observations] == ["remote"]
    assert calls == [("remote", RA, DEC)]
