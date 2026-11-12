# Search error or model error? (v0-hist, held-out TRAIN routes)

Made by `scripts/search_error_check.py`; numbers in `results/search_error.json`. 234 held-out High training routes, full (leave-one-out) history. Validation and the final test were not touched.

The decoder maximises J = sum of log-probabilities − λ·travel/scale. For each route we compare J for the zone order the beam returned with J for the driver's real zone order.
- **Search error:** the model scores the driver's order HIGHER than what the beam returned. A better search could find it.
- **Model error:** the beam's order scores higher than the driver's. Only a better model can help.

| Decoder | Beam found driver's order | Search error | Model error | Score share on search-error routes | Score share on model-error routes |
|---|---|---|---|---|---|
| beam 5, λ = 0.5 (pre-registered) | 81 | 2 | 151 | 1% | 91% |
| beam 5, λ = 0 (pure model) | 79 | 1 | 154 | 1% | 92% |

**Rule (agreed in advance):** a decoding change (a wider or stratified beam) is considered only if search errors carry a meaningful share of the score. Otherwise decoding stays frozen.
