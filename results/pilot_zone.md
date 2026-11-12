# Zone-level "swap the brain" arm: Opus picks the zone order, our solver does the rest

Routes scored: 50 of 50 pilot routes. Produced by `scripts/score_pilot_zone.py` from `runs/pilot_zone/opus.jsonl`. No v0 model is scored here.

## The idea, from zero

A delivery route has about 150 stops. Amazon groups nearby stops into **zones** (IDs like `J-16.2C`), about 20 per route. Drivers almost always finish one zone before moving to the next. So you can build a route in two steps:

1. **Choose the order of the zones** (the big decision: which neighbourhood first, which next).
2. **Inside each zone, order its few stops** (a small puzzle a solver handles well: shortest path from where you came in to where you leave towards the next zone).

Our tiny model does step 1, and a fixed solver (OR-Tools, the function `stops_from_zone_order` in `scripts/v0_decode.py`) does step 2. In this arm we **swap the brain**: Opus does step 1 and the *same* solver does step 2. If our model later beats this arm, the difference can only come from who picked the zone order, because everything else is identical.

What Opus saw for each route: the station; each zone's ID, centre point and stop count; travel times between zones (the average over all pairs of stops, the same numbers our model sees), from the station to each zone and back; and the zone orders of up to 3 earlier routes from the same station that share the most zones, all driven **strictly before** this route's date and all from the training split. It answered with a list of zone IDs.

## Results

Score: the challenge's official metric. Lower is better; 0 means exactly the driver's order. The bracket is a 95% bootstrap confidence interval: resample the routes 10,000 times and see where the average lands.

| Arm (same routes) | Mean score [95% CI] |
|---|---|
| Opus zone order + our solver, strict (invalid answers get the challenge's invalid score) | 0.0355 [0.0277, 0.0440] |
| Opus zone order + our solver, with cheapest-insertion fallback | 0.0355 [0.0277, 0.0440] |
| Opus direct (orders all stops itself, no history) | 0.0693 [0.0581, 0.0814] |
| Opus direct + history (on its 50 finished routes) | 0.0355 [0.0279, 0.0437] |
| Shortest route (no learning) | 0.0784 [0.0690, 0.0882] |

Valid zone lists: 50 of 50.

### Paired comparisons (zone arm minus the other; negative = zone arm closer to the driver)

Paired means: compare the two arms on the same route, then average the differences. The sign test asks: if each arm were equally likely to win any route, how surprising is a win/loss split this lopsided? (ties dropped)

- vs Opus direct, strict: -0.0338 [-0.0442, -0.0234]; zone arm better on 43, worse on 7, tied 0; sign test p = 2.1e-07
- vs Opus direct, fallback: -0.0338 [-0.0442, -0.0234]; zone arm better on 43, worse on 7, tied 0; sign test p = 2.1e-07
- vs Opus + history (50 routes), strict: -0.0000 [-0.0056, +0.0055]; zone arm better on 30, worse on 20, tied 0; sign test p = 0.203
- vs Opus + history, fallback: -0.0000 [-0.0056, +0.0055]; zone arm better on 30, worse on 20, tied 0; sign test p = 0.203
  (Note: the history arm's rule allowed references dated after the route; this arm's did not.)
- vs shortest, strict: -0.0428 [-0.0515, -0.0340]; zone arm better on 47, worse on 3, tied 0; sign test p = 3.71e-11
- vs shortest, fallback: -0.0428 [-0.0515, -0.0340]; zone arm better on 47, worse on 3, tied 0; sign test p = 3.71e-11

### Zone revisits and zone-order agreement

Zone revisits (leaving a zone and coming back later): zone arm 0 in total, which is zero **by construction** (the solver finishes each zone in one go); driver mean 0.96 per route; Opus direct mean 10.66. Since zero is guaranteed, the useful question is whether the zone *order* matches the driver's (the order in which the driver first entered each zone):

| Whose zone order | Kendall tau (1 = same order, 0 = unrelated, -1 = reversed) | First zone right | Next-zone pairs shared with driver | Exactly the driver's order |
|---|---|---|---|---|
| Opus zone arm | 0.399 | 22/50 | 62.3% | 7/50 |
| Opus direct (zones in first-visit order) | 0.016 | 13/50 | 28.1% | 0/50 |
| Shortest route (zones in first-visit order) | 0.037 | 9/50 | 29.3% | 0/50 |

Kendall tau, zone arm minus Opus direct (paired, 50 routes): +0.383 [+0.200, +0.572] (positive = the zone arm's order is closer to the driver's). The zone arm's tau is higher on 34 routes and lower on 14. Note: tau punishes a reversed order, while the official score treats an exact reverse as perfect.

### Cost and latency per route

- Opus zone arm: cost field $0.037 mean, $1.852 total; 5.8 s mean, 5.9 s median per call; 163,580 input and 27,204 output tokens in total.
- Inside-zone solver (our laptop): 0.19 s CPU per route.
- Opus direct on the same routes: $6.804 total, 31.1 s mean per call.

## One worked route

Route `RouteID_ce9be28a-b6b0-4e9b-b290-09a1463aee7a` (155 locations, 20 zones, 3 earlier reference route(s)), chosen because its score is closest to the median.

- Opus's zone order: B-26.3J B-26.3H B-26.2H B-26.1G B-26.2G B-26.3G B-26.3E B-26.2E B-26.1E B-26.1D B-26.2D B-26.3D B-26.3C B-26.2C B-26.1C B-26.1B B-26.2B B-26.3B B-26.3A B-26.2A
- Driver's zone order: B-26.2H B-26.3J B-26.3H B-26.1G B-26.2G B-26.3G B-26.3E B-26.2E B-26.1E B-26.1D B-26.1B B-26.2B B-26.3B B-26.3A B-26.2A B-26.2D B-26.3D B-26.3C B-26.2C B-26.1C
- Agreement: Kendall tau 0.716; first zone wrong; 79% of the driver's zone-to-zone steps also appear in Opus's order.
- The solver then ordered the stops inside each zone. Score: zone arm 0.0291, Opus direct 0.0684, shortest 0.0676 (lower is better).

## How to read this honestly

- 50 routes is small; trust the confidence intervals and sign tests, not the point estimates.
- This arm is an opponent for our model, not a result for it. The model has not been scored on these routes; that happens once, later, by Angad's decision.
- Time-respecting history leaves some routes with fewer than 3 references (see `runs/pilot_zone/prompts_index.json`), because the split is random rather than by date.
