# Reading list: published work on the Amazon Last Mile Routing Research Challenge (2021)

A summary of the published literature, with sources. You cannot open the links (no internet); the summaries are what you have. Lower scores are better.

## 1. The top of the field

- The 2021 challenge winner was Just Passing Through (Cook, Held, Helsgaun), with an evaluation score of 0.0248. 2nd place was 0.0353 and 3rd place 0.0391.[1][2]
- The winner uses no learned model at all. Its build phase only reads the training data, fills in missing zone IDs, picks zone-ID patterns and builds reference-route component paths, in 109 s on one core.[1]
- On 1,107 High-quality, fully delivered training routes, its best version scores a mean of 0.01978 and a median of 0.00732, at 1 s + 1 s per route.[1]
- Wu et al.'s 0.0374 is a post-competition AWS paper; its own table places that score next to the 3rd-placed team.[4]
- The winner's method: C (LKH-AMZ, a version of the LKH solver) plus Python scripts, about 1 s per route on CPU.[1][3]

## 2. Where the score comes from (the score ladder in Cook et al., same 1,107 routes)

| Step | Mean score |
|---|---|
| Optimal ATSP (pure shortest) | 0.07030 |
| Zones forced to be contiguous (clustered ATSP) | 0.04866 |
| + precedence from ONE reference route at the same station (most shared zones), transitive closure | 0.03169 |
| Multi-level zone-ID clusters alone | 0.03817 |
| Sorted multi-level zone-ID clusters + reference-route precedence | 0.02045 |
| + super-cluster path constraints from the reference route | 0.01993 |
| Best-of-two (full vs alternate constraint set) | 0.01978 |
| Upper bound: driver's true zone order given | 0.00505 |

All rows are from Tables 3, 7, 9, 11, 12, 13 and 6.[1] Wu et al. independently get 0.0116 from the true zone order plus TSP, against 0.08 for alphabetical zone order.[4]

Takeaways:
- **Zone order is nearly the whole game.** Within-zone stop order is worth very little under this metric.
- **The biggest single lever is history at the same station.** Copying the zone precedence of one past route from the same station that shares the most zones takes 0.0487 to 0.0317, a 35% cut.
- **The hierarchical zone-ID structure is the second lever.** Zone IDs look like `A-2.3D`. Drivers sweep the zones inside each super-cluster in sorted or reverse-sorted order and chain super-clusters at matching sub-IDs.[1]
- **Time windows do not help.** Adding them made the best model worse (0.02148 vs 0.01993).[1]

## 3. What the learning approaches got

- Wu et al. (AWS) used a PPM high-order Markov model over zone-ID strings, then rollout, then LKH inside zones: 0.0374 on evaluation, 0.0313 on training.[4] They first tried imitation-learning neural nets on 6,112 routes and "those initial efforts did not achieve the same level of result."[4] They also found that zone names can be treated as identical across stations.[4]
- Mo et al. (MIT, the 2nd-place team's later paper) used a pair-wise attention pointer network: 0.0369 on their own split, not the official set. That is barely better than their own TSP baselines (0.0430 to 0.0443) and far behind Cook et al.'s 0.0198 in the same table.[5]
- Scroccaro et al. learned TSP arc costs by inverse optimisation at zone level: 0.0302 on evaluation.[6]
- Ghosh et al. (Learn Global, Optimize Local, 2023) used a first-order zone-transition count matrix from training routes, blended with zone travel times, as the cost of a zone-level TSP, then a stop-level TSP inside each zone: 0.0299 on their 1,000-route test set and 0.0300 on the official evaluation.[7] They report that history decides the starting zone, that station-to-zone transitions should weight travel time less, and that zone-to-station should use travel time only. With the true zone sequence the same pipeline gets 0.0094.[7]
- Sun and Arslan (2025), counted zone-ID feature weights (R+LFIM): 0.0294 out of sample on the evaluation set; their headline "40.1% better" figure is an in-sample training score (0.0119).[8]

## 4. Published scores on the official evaluation set

| Method | Eval score | Learned? |
|---|---|---|
| Cook, Held, Helsgaun (winner) [1] | 0.0248 | No |
| Sun and Arslan 2025, R+LFIM [8] | 0.0294 | Counted zone-ID feature weights |
| Ghosh et al. 2023 [7] | 0.0300 | Count matrix |
| Scroccaro et al. 2024, inverse optimisation [6] | 0.0302 | Yes |
| 2nd place (Guo, Mo, Wang) [9] | 0.0353 | No |
| Wu et al. (AWS) [4] | 0.0374 | Markov model |

The evaluation set's routes are all of High quality (Wu et al.; the data-set paper [10]).

## Sources
[1] Cook, Held, Helsgaun, Constrained Local Search for Last-Mile Routing, arXiv 2112.15192
[2] Just Passing Through team page (leaderboard), math.uwaterloo.ca/tsp/amz
[3] jpt-amz code (MIT licence), github.com/heldstephan/jpt-amz
[4] Wu et al., Learning from Drivers to Tackle the Amazon Last Mile Routing Research Challenge, arXiv 2205.04001
[5] Mo et al., Predicting Drivers' Route Trajectories in Last-Mile Delivery Using A Pair-wise Attention-based Pointer Neural Network, arXiv 2301.03802
[6] Scroccaro et al., Inverse Optimization for Routing Problems, arXiv 2307.07357
[7] Ghosh, Kuiper, Mahes, Maragno, Learn global and optimize local: A data-driven methodology for last-mile routing, Computers & Operations Research 159 (2023)
[8] Sun and Arslan, A Contextual Framework for Learning Routing Experiences in Last-Mile Delivery, Transportation Research Part B (2025)
[9] Guo, Mo, Wang, Hierarchical TSP with Customized Cost Matrix, arXiv 2302.02102
[10] Merchán et al., 2021 Amazon Last Mile Routing Research Challenge: Data Set, Transportation Science (2022)
