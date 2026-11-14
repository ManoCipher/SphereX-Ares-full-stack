import pytest

from spherex.time_selection import group_epochs, observation_mjd, select_time_series

RA, DEC = 133.795, -7.245


def _obs(oid, mjd, detector="4", ra=RA, dec=DEC):
    return {
        "observation_id": oid,
        "observation_start_mjd": mjd,
        "observation_timestamp": None,
        "detector": detector,
        "ra": ra,
        "dec": dec,
        "fits_url": f"https://irsa.example/{oid}_{detector}.fits",
    }


MAY, OCT = 60801.5, 60979.7  # two SPHEREx visits about six months apart


def test_observation_mjd_from_iso_timestamp():
    assert observation_mjd({"observation_timestamp": "1970-01-02T00:00:00"}) == pytest.approx(40588.0)
    assert observation_mjd({"observation_timestamp": "not a date"}) is None


def test_group_epochs_splits_on_gaps():
    records = [_obs("a", MAY), _obs("b", MAY + 0.01), _obs("c", OCT), _obs("d", OCT + 0.05)]
    epochs = group_epochs(records)
    assert [[r["observation_id"] for r in epoch] for epoch in epochs] == [["a", "b"], ["c", "d"]]


def test_picks_five_spread_over_epochs_and_time_ordered():
    may = [_obs(f"m{i}", MAY + i * 0.001, ra=RA + 0.01 * i) for i in range(4)]
    october = [_obs(f"o{i}", OCT + i * 0.001, ra=RA + 0.01 * i) for i in range(10)]
    result = select_time_series(october + may, RA, DEC)
    ids = [r["observation_id"] for r in result["observations"]]
    assert len(ids) == 5
    # Best (closest to centre) of each epoch first, then alternate: m0,o0,m1,o1,m2
    assert set(ids) == {"m0", "m1", "m2", "o0", "o1"}
    mjds = [r["observation_start_mjd"] for r in result["observations"]]
    assert mjds == sorted(mjds)
    assert result["epoch_count"] == 2 and result["detector"] == "4"


def test_uses_all_frames_when_fewer_than_requested():
    result = select_time_series([_obs("a", MAY), _obs("b", OCT), _obs("c", OCT + 0.01)], RA, DEC)
    assert len(result["observations"]) == 3
    assert "Only 3" in result["note"]


def test_chooses_detector_with_most_epochs():
    records = [_obs(f"x{i}", MAY + i * 0.001, detector="2") for i in range(6)]  # many frames, one epoch
    records += [_obs("y1", MAY, detector="4"), _obs("y2", OCT, detector="4")]  # fewer frames, two epochs
    result = select_time_series(records, RA, DEC)
    assert result["detector"] == "4"
    assert [r["observation_id"] for r in result["observations"]] == ["y1", "y2"]


def test_explicit_detector_duplicates_and_single_epoch_note():
    records = [_obs("a", MAY), _obs("a", MAY), _obs("b", MAY + 0.01), _obs("z", OCT, detector="6")]
    result = select_time_series(records, RA, DEC, detector="4")
    assert [r["observation_id"] for r in result["observations"]] == ["a", "b"]
    assert "single epoch" in result["note"]


def test_handles_no_usable_records_and_bad_count():
    assert select_time_series([{"observation_id": "x"}], RA, DEC)["observations"] == []
    with pytest.raises(ValueError):
        select_time_series([], RA, DEC, count=0)
