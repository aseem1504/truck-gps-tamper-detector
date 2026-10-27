"""End-to-end demo on synthetic data.

    python demo.py

1. simulates a fleet where every vehicle-day has one known wiring condition
2. runs the rule engine and scores it against the injected truth
3. builds blackout features, shows the leakage trap in the original model, trains the honest one
"""
import os

import pandas as pd

from tamper_detector import simulate
from tamper_detector.features import build_features
from tamper_detector.model import evaluate, rfe_ranking, top_importances
from tamper_detector.quality import ping_quality
from tamper_detector.wire_issues import COLUMNS, analysis_flow

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
os.makedirs('outputs', exist_ok=True)

print("\n=== 1. Rule engine on a simulated fleet ===")
rows, quality = [], []
for vid, fault, lc, hb in simulate.fleet():
    res = analysis_flow(lc, hb)
    if res is None:
        continue
    r = res.iloc[0].to_dict()
    r['true_fault'] = fault
    r['channel'] = 'hb+lc' if len(hb) else 'lc_only'
    rows.append(r)
    quality.append(ping_quality(lc, vid))
flags = pd.DataFrame(rows)
flag_cols = COLUMNS[2:]
flags[flag_cols] = flags[flag_cols].astype(bool)

hit = flags.apply(lambda r: r['true_fault'] == 'normal' and not r[flag_cols].any()
                  or (r['true_fault'] != 'normal' and r[r['true_fault']]), axis=1)
print(f"vehicle-days scored: {len(flags)}   correct primary flag: {hit.mean():.0%}")
print("\nHow often each flag fired, by injected condition:")
print(flags.groupby('true_fault')[flag_cols].mean().round(2))
flags.to_csv('outputs/wire_issue_flags.csv', index=False)
pd.concat(quality).to_csv('outputs/ping_quality.csv', index=False)

print("\n=== 2. Blackout model: the leakage trap ===")
events = simulate.no_info_events()
events.to_csv('outputs/no_info_events_synthetic.csv', index=False)
X_leak, y = build_features(events, include_leaky=True)
leaky, _ = evaluate(X_leak, y)
print("With displacement_travelled as a feature (what the original notebook did):")
print(leaky.to_string(index=False))

print("\n=== 3. Honest model (only what is known when the blackout starts) ===")
X, y = build_features(events)
honest, models = evaluate(X, y)
print(f"base rate of 'moved >1 km while dark': {y.mean():.0%}")
print(honest.to_string(index=False))
print("\nTop random forest features:")
print(top_importances(models['random_forest'], X.columns).to_string())
print("\nRFE ranking (1 = kept):")
for rank, name in rfe_ranking(X, y)[:10]:
    print(f"  {rank:>2}  {name}")
print("\nOutputs written to ./outputs")
