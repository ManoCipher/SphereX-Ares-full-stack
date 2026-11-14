import pytest

from spherex.archive_client import OBSERVATION_COLUMNS, IrsaSiaClient, parse_archive_rows
from spherex.config import ArchiveConfig
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


def test_current_releases_are_combined(monkeypatch):
    client = IrsaSiaClient(ArchiveConfig(collections=("spherex_dr1", "spherex_qr3")))

    def fake_search(target, collection, epoch):
        return parse_archive_rows(
            [{"obs_id": collection, "access_url": f"https://example.test/{collection}.fits"}],
            collection,
            target,
        )

    monkeypatch.setattr(client, "_search_collection", fake_search)
    result = client.discover(WISE_0855)

    assert result["collection"].tolist() == ["spherex_dr1", "spherex_qr3"]
