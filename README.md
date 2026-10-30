# Truck GPS Tamper Detector

Find trucks whose GPS trackers are being tampered with, using nothing but the pings the tracker already sends.

Built in 2020 for a truck GPS fleet. All data in this repo is synthetic.

Sister repos: [gps-device-health-scorecard](https://github.com/aseem1504/gps-device-health-scorecard) is the daily device health job (AWS Lambda) that runs on the same ping data. [truck-blackout-movement-predictor](https://github.com/aseem1504/truck-blackout-movement-predictor) takes the Step 2 model below further: odds ratios, visit planning and cost-based cut-offs.

## The problem

A truck tracker is only useful if it stays connected. In practice, a lot of trackers go quiet. Some of that is honest: a loose wire, a dead SIM, a truck parked in a garage with the battery pulled. Some of it is not. A driver who wants to run an unofficial trip, or skip a toll, cuts the power wire and the truck goes dark.

From the dashboard, all of these look the same: "No info." The ops team had no way to tell a loose wire from a deliberate cut, so every case got the same slow field visit. Fleet owners, and lenders who financed the trucks, cared about one question. Is this truck really parked, or is someone hiding it?

## What this does

It answers that in two steps.

**Step 1. A rule engine reads one day of pings and names the wiring fault.** Each ping carries a few signals: is external power on, is the ignition on, battery voltage, location and time. Faults leave fingerprints in those signals:

| Flag | Fingerprint |
|---|---|
| `intentional_wire_cut` | power off for over an hour, voltage below 60%, and the truck still moving at speed |
| `maintainence_wire_cut` | power off for over an hour, voltage below 60%, truck standing still |
| `wire_loose` | power flickers on and off more than twice in a day |
| `charge_to_ign` | power wire hooked to ignition: power is on only when the engine is on |
| `ign_wire_cut` | ignition never reads "on", yet the truck drives at 10+ km/h |
| `ign_to_charge` | ignition wire hooked to permanent power: ignition reads "on" all the time |

The key move is telling a deliberate cut from a garage repair. Both cut power. Only one happens while the truck is moving.

**Step 2. A model guesses whether the truck moved while it was dark.** For every blackout, it builds features from what is known the moment the tracker goes quiet: device model, network, recent wiring flags, how many blackouts this truck had in the last 30 days and how long they lasted, whether it was driving or parked just before, the time of day and a geohash of the location. The target is simple. Was the first ping after the blackout more than 1 km from the last ping before it? If yes, the truck moved while hidden.

The rule engine also produces GPS quality metrics per vehicle (how far pings jump, how late they arrive, signal strength, satellites used) and ranks device models on them.

## Results on synthetic data

`python demo.py` simulates 140 vehicle-days, each with one wiring condition planted, plus 3,000 blackout events.

The rule engine catches the planted fault in every vehicle-day. The more useful finding is where flags overlap. When the power wire is hooked to ignition, `wire_loose` fires too, because power toggles with the engine. On devices with a separate heartbeat channel, `ign_to_charge` also fires on a third of those days. In production, that means the rules need a priority order before they go to a field team, or one truck gets two different repair tickets.

For the model:

| Version | Accuracy | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| Logistic regression, leaky feature in (as in the original) | 0.94 | 1.00 | 0.89 | 0.99 |
| Random forest, leaky feature in | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic regression, honest features | 0.71 | 0.72 | 0.77 | 0.78 |
| Random forest, honest features | 0.70 | 0.71 | 0.79 | 0.76 |

## The mistake worth talking about

The original notebook got great accuracy. It was fake. The feature list included `displacement_travelled`, which is the exact number the target is calculated from. The model was being shown the answer.

This is called target leakage, and it is easy to miss when the feature list is long and was built by many helper functions. The fix is a simple rule: only use what you would know at the moment you need to make the call. Here, that is the moment the tracker goes quiet. The honest model scores lower, and it is the only one that would work in the real world.

It also changes the product question. At 0.72 precision, about 1 in 4 flagged trucks did not actually move. Is that good enough to send a field agent? Or to tell a bank a truck might be hidden? That depends on what a wasted visit costs compared with a missed one. A model score alone does not settle it.

## Run it

```bash
pip install -r requirements.txt
python demo.py        # prints results, writes CSVs to outputs/
pytest                # quick checks
```

## Repo layout

```
tamper_detector/
  wire_issues.py   rule engine (ported from the original, thresholds unchanged)
  features.py      blackout features: time slot, duration bucket, geohash, encodings
  model.py         train/test split, logistic regression vs random forest, RFE ranking
  quality.py       ping jump / delay / signal metrics, device-model ranking
  simulate.py      synthetic pings, heartbeats and blackout events
  geo.py           haversine distance
demo.py            end-to-end run
data/sample/       small synthetic samples so you can see the data shape
notebooks/original/  the real notebooks, cleaned (see the README inside)
```

## What changed from the original code

The logic and thresholds are the same. The changes were needed to make it run without company systems, or on current library versions:

- Pings come in as DataFrames instead of being fetched from internal APIs and Mongo.
- Boolean columns are cast to integers before subtracting. Newer pandas refuses `bool - bool`.
- `df[a][b]` chained filters became `df[a & b]`. Same result, no warnings.
- `time_slot` returned nothing for a ping landing exactly on a 4-hour boundary. Fixed.
- `category_encoders` and `pygeohash` were replaced with a few lines of plain code.
- The leaky feature is now opt-in (`include_leaky=True`), so the demo can show the trap.

Two quirks are kept on purpose so the port stays faithful. The cut-speed thresholds divide km/h by 3.6 a second time, which makes them stricter than they look. And the heartbeat version of `ign_to_charge` has no flicker check, which causes the overlap described above.

## Data and privacy

Nothing here came from real trucks. Vehicle IDs, routes and pings are all generated by `simulate.py`. The original notebooks had their outputs, credentials, hosts and IDs removed before upload.
