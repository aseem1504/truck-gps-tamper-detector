"""Feature building for the "did the truck move while the tracker was dark?" model.

Ported from notebooks/original/03_no_info_feature_generation_and_model.ipynb.
The original pulled each feature with a live query per event (Mongo deviceHistory,
itinerary, wire_issues, vehicle_details, GPS API). Here the same fields arrive as one
DataFrame so the logic can run offline.
"""
import numpy as np
import pandas as pd

CATEGORICAL = ['sales_dept', 'network', 'installation_subtype', 'current_state']
WIRE_FLAGS = ['intentional_wire_cut', 'maintainence_wire_cut', 'wire_loose',
              'charge_to_ign', 'ign_wire_cut', 'ign_to_charge']
_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'


def no_info_duration_bucket(duration_sec):
    i = duration_sec
    if i < 1 * 86400:
        return "less than one day"
    elif i < 3 * 86400:
        return "1-3 days"
    elif i < 7 * 86400:
        return "3-7 days"
    elif i < 15 * 86400:
        return "7-15 days"
    elif i < 30 * 86400:
        return "15-30 days"
    return "30+ days"


def time_slot(x):
    """Six 4-hour slots in the day, 1 = midnight to 4 am. (Original returned nothing for
    a ping landing exactly on a slot boundary; that edge case is fixed here.)"""
    seconds = x % 86400
    return int(min(seconds // 14400, 5)) + 1


def geohash(lat, lon, precision=5):
    """Plain geohash encoder so we don't need pygeohash for one function."""
    lat_r, lon_r, bits, out, even, bit, ch = [-90.0, 90.0], [-180.0, 180.0], [16, 8, 4, 2, 1], '', True, 0, 0
    while len(out) < precision:
        rng, val = (lon_r, lon) if even else (lat_r, lat)
        mid = (rng[0] + rng[1]) / 2
        if val > mid:
            ch |= bits[bit]
            rng[0] = mid
        else:
            rng[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out += _BASE32[ch]
            bit, ch = 0, 0
    return out


def build_features(events, include_leaky=False):
    """events: output of simulate.no_info_events() (or the same columns from real data).

    Returns X (DataFrame), y (1 = moved more than 1 km during the blackout).

    include_leaky=True reproduces the original notebook, which fed `displacement_travelled`
    into the model even though the target is computed from that same number."""
    df = events.copy()
    df['time_slot'] = df['fromTime'].apply(time_slot)
    df['sum_of_duration'] = df['avg_duration'] * df['alert_count']
    df['log_duration'] = np.log1p(df['duration'])
    for c in CATEGORICAL:  # ordinal encoding, like category_encoders.OrdinalEncoder
        df[c] = df[c].astype('category').cat.codes + 1
    for c in WIRE_FLAGS:
        df[c] = df[c].astype(int)
    # geohash -> integer code -> 6 binary columns (like category_encoders.BinaryEncoder)
    gh = df.apply(lambda r: geohash(r['lat'], r['long'], 3), axis=1).astype('category').cat.codes.astype(int).to_numpy()
    for b in range(6):
        df[f'encoded_loc_{b}'] = (gh >> b) & 1

    cols = (['device_model_id'] + CATEGORICAL + ['alert_count', 'avg_duration', 'sum_of_duration',
            'log_duration'] + WIRE_FLAGS + ['driving_count', 'stoppage_count', 'time_slot',
            'displacements_mean'] + [f'encoded_loc_{b}' for b in range(6)])
    if include_leaky:
        df['displacement_travelled'] = df['moved_km'] * 1000
        cols = cols + ['displacement_travelled']
    y = (df['moved_km'] > 1.0).astype(int)
    return df[cols], y
