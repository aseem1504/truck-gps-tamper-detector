"""Synthetic telematics data so the project runs without any company systems.

Nothing here comes from real vehicles. Routes are random walks around Indian highway
coordinates, and each vehicle-day gets one injected wiring condition so the rule engine
can be scored against a known answer.
"""
import numpy as np
import pandas as pd

FAULTS = ['normal', 'intentional_wire_cut', 'maintainence_wire_cut', 'wire_loose',
          'charge_to_ign', 'ign_wire_cut', 'ign_to_charge']


def _trip(rng, start_ts, n=24 * 60, step=60):
    """One day of pings, one per minute, alternating driving and stop blocks."""
    t = start_ts + np.arange(n) * step
    lon, lat = rng.uniform(72, 86), rng.uniform(18, 28)
    driving = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        block = rng.integers(60, 240)
        driving[i:i + block] = rng.random() < 0.55
        i += block
    speed = np.where(driving, rng.normal(45, 10, n).clip(15, 80), 0.0)  # km/h
    heading = rng.uniform(0, 2 * np.pi)
    locs = []
    for s in speed:
        heading += rng.normal(0, 0.05)
        d = s / 3.6 * step / 111_000  # metres to degrees (rough)
        lon += d * np.cos(heading)
        lat += d * np.sin(heading)
        locs.append([round(lon + rng.normal(0, 2e-5), 6), round(lat + rng.normal(0, 2e-5), 6)])
    return t, speed, driving, locs


def vehicle_day(rng, vehicle_id, start_ts, fault='normal', with_heartbeat=False):
    t, speed, driving, locs = _trip(rng, start_ts)
    n = len(t)
    ign = driving.copy()
    ign |= rng.random(n) < 0.03  # engine idling while parked sometimes
    charge = np.ones(n, dtype=bool)
    volt = rng.normal(95, 2, n).clip(80, 100)

    if fault in ('intentional_wire_cut', 'maintainence_wire_cut'):
        mask = driving if fault == 'intentional_wire_cut' else ~driving
        idx = np.where(mask)[0]
        # longest run of the wanted state, cut power for most of it
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        run = max(runs, key=len)
        cut = run[5:-5] if len(run) > 80 else run
        if fault == 'intentional_wire_cut':
            charge[:] = True
        charge[cut] = False
        volt[cut] = np.linspace(90, 40, len(cut))  # internal battery draining
    elif fault == 'wire_loose':
        flick = rng.choice(n, size=12, replace=False)
        charge[flick] = False
    elif fault == 'charge_to_ign':
        charge = ign.copy()
        volt = np.where(charge, volt, 70)
    elif fault == 'ign_wire_cut':
        ign[:] = False
    elif fault == 'ign_to_charge':
        ign[:] = True

    created = t + rng.exponential(20, n).astype(int)
    created[rng.random(n) < 0.01] += 900  # a few late deliveries
    lc = pd.DataFrame({
        'vehicleId': vehicle_id, 'time': t, 'created': created, 'location': locs,
        'speed': speed.round(1), 'gsmPct': rng.normal(70, 15, n).clip(0, 100).round(),
        'usedSat': rng.integers(4, 14, n),
    })
    if not with_heartbeat:
        lc['chargeOn'], lc['ignOn'], lc['voltPct'] = charge, ign, volt.round(1)
        return lc, pd.DataFrame()
    hb = pd.DataFrame({
        'vehicleId': vehicle_id, 'created': (t + 5) * 1000, 'updated': (t + 5) * 1000,
        'chargeOn': charge, 'engineOn': ign, 'voltPct': volt.round(1),
    })
    return lc, hb


def fleet(n_vehicles=140, seed=7, start_ts=1_600_000_000):
    """Returns a list of (vehicle_id, fault, lc_df, hb_df)."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_vehicles):
        fault = FAULTS[k % len(FAULTS)]
        vid = 100000 + k
        lc, hb = vehicle_day(rng, vid, start_ts, fault, with_heartbeat=(k % 3 == 0))
        out.append((vid, fault, lc, hb))
    return out


def no_info_events(n=3000, seed=11):
    """Device blackout events with the same raw fields the original feature pipeline pulled
    from deviceHistory, itinerary, vehicle_details and the wire_issues table.

    `moved_km` is the distance between the last ping before and first ping after the blackout.
    Moving more than 1 km while dark is the tamper signal the model tries to predict."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'vehicle_id': 200000 + rng.integers(0, 900, n),
        'device_model_id': rng.choice([3, 5, 8, 11], n, p=[.4, .3, .2, .1]),
        'sales_dept': rng.choice(['SME', 'ENTERPRISE', 'BANK'], n),
        'network': rng.choice(['AIRTEL', 'VODAFONE', 'IDEA'], n),
        'installation_subtype': rng.choice(['NEW', 'REINSTALL', 'RELAY'], n, p=[.6, .3, .1]),
        'alert_count': rng.poisson(3, n),
        'current_state': rng.choice(['DRIVING', 'STOPPAGE'], n, p=[.45, .55]),
        'fromTime': 1_600_000_000 + rng.integers(0, 30 * 86400, n),
        'duration': rng.exponential(5 * 3600, n).astype(int) + 600,
        'lat': rng.uniform(18, 28, n).round(3), 'long': rng.uniform(72, 86, n).round(3),
    })
    for f in FAULTS[1:]:
        df[f] = rng.random(n) < 0.06
    df['avg_duration'] = rng.exponential(4 * 3600, n)
    df['driving_count'] = rng.binomial(df['alert_count'], 0.45)
    df['stoppage_count'] = df['alert_count'] - df['driving_count']
    df['displacements_mean'] = rng.exponential(3000, n)

    # hidden truth: what makes a truck move while its tracker is dark
    hour = (df.fromTime % 86400) / 3600
    logit = (-2.0 + 1.6 * (df.current_state == 'DRIVING') + 2.2 * df.intentional_wire_cut
             + 0.9 * df.ign_wire_cut + 0.5 * np.log1p(df.duration / 3600)
             + 0.7 * ((hour > 22) | (hour < 5)) + 0.25 * (df.driving_count - df.stoppage_count)
             + 0.0002 * df.displacements_mean - 0.6 * (df.device_model_id == 11)
             + rng.normal(0, 0.8, n))
    moved = rng.random(n) < 1 / (1 + np.exp(-logit))
    df['moved_km'] = np.where(moved, rng.exponential(25, n) + 1.0, rng.exponential(0.25, n))
    df['moved_km'] = df['moved_km'].round(3)
    return df
