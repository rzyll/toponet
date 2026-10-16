# E9: topology-guided BEV re-query

E9 starts from E5, preserves all existing node identities and relation
matrices, and gathers fresh local BEV evidence around each E5-refined lane.
Topology-derived endpoint offsets define a guided reference curve. Five local
samples per lane point (center and four axial offsets), learned offsets, and
attention weights produce a second bounded residual correction.

## OpenLane-V2 subset A validation

All results use the complete 4,806-frame validation set and the official
`openlanev2==1.1.0` evaluator.

| model | OLS | DET_l | DET_t | TOP_ll | TOP_lt | Lane F-score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reproduced TopoNet E0 | 0.383147 | 0.268143 | 0.452547 | 0.102808 | 0.241339 | 0.149613 |
| E5 | 0.389696 | 0.273521 | 0.452607 | 0.114262 | 0.244658 | 0.149924 |
| E9 | **0.390409** | **0.274664** | **0.452607** | **0.114865** | **0.245466** | **0.150166** |

E9 combined diagnostic micro edge precision/recall/F1 is
`0.407264 / 0.162997 / 0.232815`. Under canonical 2 m matching, it yields a
net gain of 741 GT lanes over E0, including 371 connectors. Official TOP and
diagnostic micro edge metrics are reported separately and are not added.

These are local reproductions under the official metric protocol, not the
TopoNet paper's published leaderboard number.
