import pytest

from spherex.archive_client import OBSERVATION_COLUMNS, IrsaSiaClient, parse_archive_rows
from spherex.targets import WISE_0855


def test_parse_archive_response():
    rows = [{
        "obs_id": "example-1",
        "t_min": 60850.0,
        "em_min": 3.83e-6,
        "em_max": 4.41e-6,
        "access_url": "https://irsa.ipac.caltech.edu/ibe/data/spherex/qr3/level2/a.fits",
        "access_format": "application/fits",
    }]
    result = parse_archive_rows(rows, "spherex_qr3", WISE_0855)
    assert len(result) == 1
    assert result.loc[0, "data_release"] == "qr3"
    assert result.loc[0, "band_min_um"] == pytest.approx(3.83)
    assert result.loc[0, "target_ra_deg"] != WISE_0855.ra_deg


def test_empty_results(monkeypatch):
    client = IrsaSiaClient()
    monkeypatch.setattr(client, "_search_collection", lambda *args, **kwargs: parse_archive_rows([], "spherex_qr3", WISE_0855))
    result = client.discover(WISE_0855)
    assert result.empty
    assert list(result.columns) == OBSERVATION_COLUMNS
