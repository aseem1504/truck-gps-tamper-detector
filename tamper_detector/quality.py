"""GPS data quality metrics per vehicle and a simple device-model ranking.

Ported from notebooks/original/04_jump_delay_gsm_gps_quality.ipynb and 06_device_model_ranking.ipynb.
A "jump" is a gap of more than 500 m between two back-to-back pings. A "delay" is a ping that
reached the server more than 5 minutes after the device recorded it.
"""
import pandas as pd

from .geo import haversine


def ping_quality(data1, vehicleId):
    data1 = data1.sort_values('time', ascending=True).copy()
    data1['loc2'] = data1.location.shift(1)
    data1 = data1[1:]
    data1['distance'] = [haversine(a, b) for a, b in data1[['location', 'loc2']].values.tolist()]
    total = data1['distance'].sum()
    jump_percent_distance = data1[data1['distance'] > 500]['distance'].sum() / total if total else 0
    data1['server_delay'] = data1.created - data1.time

    pct = [0.75, 0.85, 0.90, 0.95, 0.99]
    jumps = data1[data1['distance'] > 500]['distance'].describe(percentiles=pct)
    delays = data1[data1['server_delay'] > 300]['server_delay'].describe(percentiles=pct)
    low = [0.01, 0.05, 0.2]
    gsm = data1['gsmPct'].describe(percentiles=low) if 'gsmPct' in data1 else None
    sats = data1['usedSat'].describe(percentiles=low) if 'usedSat' in data1 else None

    row = {
        'vehicleId': vehicleId,
        'total_pings': len(data1),
        'jump_percent_distance': round(jump_percent_distance, 4),
        'jump_count': int(jumps['count']),
        'jump_95%': jumps.get('95%', 0),
        'delay_count': int(delays['count']),
        'delay_95%': delays.get('95%', 0),
        'gsm_mean': None if gsm is None else gsm['mean'],
        'gsm_5%': None if gsm is None else gsm['5%'],
        'gps_mean': None if sats is None else sats['mean'],
        'gps_5%': None if sats is None else sats['5%'],
    }
    return pd.DataFrame([row]).fillna(0)


def rank_device_models(device_performance):
    """device_performance: one row per vehicle-day with model_name and quality columns.
    Returns one row per model, ranked by how often it pings while the truck is stopped
    (same ordering used in the original notebook)."""
    g = device_performance.groupby('model_name')
    table = pd.DataFrame({
        'averagedelay': g.averagedelay.mean(),
        'jumpdistance': g.jumpdistance.mean(),
        'usedsat': g.usedsat.mean(),
        'connected_percent': g.connected_percent.mean(),
        'jump_percent': g.jump_percent.mean(),
        'driving_ping_frequency': g.drivingtime.mean() / g.drivingrawcount.mean(),
        'stoppage_ping_frequency': g.stoppagetime.mean() / g.stoppagerawcount.mean(),
    }).reset_index()
    table.sort_values('stoppage_ping_frequency', ascending=False, inplace=True)
    table.reset_index(inplace=True, drop=True)
    table['stoppage_rank'] = range(1, len(table) + 1)
    return table
