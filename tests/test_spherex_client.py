from __future__ import annotations

from io import BytesIO

import pytest
import requests
from astropy.table import Table

from spherex.spherex_client import SPHEREX_COLLECTIONS, SPHERExArchiveError, SPHERExClient
from frontend.tkinter_viewer import _size_label, _wavelength_label


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses=None, error: Exception | None = None) -> None:
        self.headers = {}
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.responses.pop(0)


def votable_bytes(rows: list[dict]) -> bytes:
    if rows:
        table = Table(rows=rows)
    else:
        table = Table(names=("obs_id", "access_url"), dtype=("U20", "U200"))
    output = BytesIO()
    table.write(output, format="votable")
    return output.getvalue()


def make_client(rows: list[dict], **kwargs) -> tuple[SPHERExClient, FakeSession]:
    session = FakeSession([FakeResponse(votable_bytes(rows))])
    client = SPHERExClient(session=session, collections=("spherex_qr3",), **kwargs)
    return client, session


def test_default_collections_use_current_releases():
    assert SPHEREX_COLLECTIONS == (
        "spherex_dr1",
        "spherex_dr1_deep",
        "spherex_qr3",
        "spherex_qr3_deep",
    )


def test_valid_coordinate_search_and_query_parameters():
    client, session = make_client([{"obs_id": "one", "access_url": "https://example.test/one.fits"}])
    results = client.search(ra=359.99, dec=-7.245, radius=0.05, start_date="2025-01-01")
    assert len(results) == 1
    params = session.calls[0][1]["params"]
    assert params["POS"] == "CIRCLE 359.99 -7.245 0.05"
    assert params["COLLECTION"] == "spherex_qr3"
    assert params["TIME"].endswith(" +Inf")
    assert session.calls[0][1]["timeout"] == 60.0


@pytest.mark.parametrize("ra,dec", [(-0.01, 0), (360, 0), (0, -90.1), (0, 90.1)])
def test_invalid_ra_dec(ra, dec):
    client, _ = make_client([])
    with pytest.raises(ValueError):
        client.search(ra=ra, dec=dec)


def test_no_observations_found():
    client, _ = make_client([])
    assert client.search(ra=10, dec=20) == []


def test_network_timeout_is_wrapped():
    session = FakeSession(error=requests.Timeout("too slow"))
    client = SPHERExClient(session=session, collections=("spherex_qr3",), timeout=1.5)
    with pytest.raises(SPHERExArchiveError, match="spherex_qr3"):
        client.search(ra=10, dec=20)
    assert session.calls[0][1]["timeout"] == 1.5


def test_missing_metadata_uses_none():
    client, _ = make_client([{"obs_id": "sparse", "access_url": ""}])
    result = client.search(ra=10, dec=20)[0]
    assert result["observation_id"] == "sparse"
    assert result["observation_timestamp"] is None
    assert result["detector"] is None
    assert result["wavelength_min_um"] is None
    assert result["fits_url"] is None
    assert result["exposure_time_s"] is None
    assert result["file_size_bytes"] is None


def test_fits_url_and_metadata_extraction():
    url = (
        "https://irsa.ipac.caltech.edu/ibe/data/spherex/qr3/level2/"
        "2025W40_1A/l2b-v1-2025-001/4/image.fits"
    )
    client, _ = make_client([{
        "obs_id": "image-1",
        "access_url": url,
        "access_format": "application/fits",
        "em_min": 2.42e-6,
        "em_max": 3.82e-6,
        "t_exptime": 112.5,
        "access_estsize": 10.0,
    }])
    result = client.search(ra=10, dec=20)[0]
    assert result["fits_url"] == url
    assert result["detector"] == "4"
    assert result["wavelength_min_um"] == pytest.approx(2.42)
    assert result["exposure_time_s"] == 112.5
    assert result["file_size_bytes"] == 10 * 1024


def test_chronological_sorting_with_missing_time_last():
    client, _ = make_client([
        {"obs_id": "late", "t_min": 61000.0, "access_url": "https://example.test/a.fits"},
        {"obs_id": "missing", "t_min": float("nan"), "access_url": "https://example.test/b.fits"},
        {"obs_id": "early", "t_min": 60000.0, "access_url": "https://example.test/c.fits"},
    ])
    results = client.search(ra=10, dec=20)
    assert [row["observation_id"] for row in results] == ["early", "late", "missing"]


def test_search_result_limit_is_applied_after_sorting():
    client, session = make_client([
        {"obs_id": "third", "t_min": 60003.0, "access_url": "https://example.test/3.fits"},
        {"obs_id": "first", "t_min": 60001.0, "access_url": "https://example.test/1.fits"},
        {"obs_id": "second", "t_min": 60002.0, "access_url": "https://example.test/2.fits"},
    ])
    results = client.search(ra=10, dec=20, max_results=2)
    assert [row["observation_id"] for row in results] == ["first", "second"]
    assert session.calls[0][1]["params"]["MAXREC"] == 2


def test_invalid_archive_response():
    session = FakeSession([FakeResponse(b"not a VOTable")])
    client = SPHERExClient(session=session, collections=("spherex_qr3",))
    with pytest.raises(SPHERExArchiveError, match="invalid VOTable"):
        client.search(ra=10, dec=20)


def test_viewer_formats_archive_values():
    observation = SPHERExClient._parse_row(
        {"em_min": 0.75e-6, "em_max": 1.09e-6, "access_estsize": 2048},
        "spherex_qr3",
    )
    assert _wavelength_label(observation) == "0.750–1.090"
    assert _size_label(observation["file_size_bytes"]) == "2.0 MiB"
