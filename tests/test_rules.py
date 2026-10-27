import numpy as np

from tamper_detector import simulate
from tamper_detector.features import build_features, geohash, time_slot
from tamper_detector.geo import haversine
from tamper_detector.wire_issues import analysis_flow


def test_haversine_one_degree_latitude():
    assert abs(haversine([77.0, 28.0], [77.0, 29.0]) - 111_195) < 200


def test_each_fault_fires_its_own_flag():
    rng = np.random.default_rng(1)
    for fault in simulate.FAULTS[1:]:
        lc, hb = simulate.vehicle_day(rng, 1, 1_600_000_000, fault)
        assert bool(analysis_flow(lc, hb).iloc[0][fault]), fault


def test_normal_day_raises_nothing():
    lc, hb = simulate.vehicle_day(np.random.default_rng(2), 1, 1_600_000_000, 'normal')
    assert not analysis_flow(lc, hb).iloc[0, 2:].astype(bool).any()


def test_helpers():
    assert time_slot(0) == 1 and time_slot(86399) == 6 and time_slot(14400) == 2
    assert geohash(57.64911, 10.40744, 11) == 'u4pruydqqvj'  # reference example from the geohash spec


def test_leaky_feature_is_opt_in():
    ev = simulate.no_info_events(200)
    X, _ = build_features(ev)
    assert 'displacement_travelled' not in X.columns
