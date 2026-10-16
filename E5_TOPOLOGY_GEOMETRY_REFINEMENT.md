# E5: topology-conditioned lane geometry refinement

E5 keeps TopoNet's 200 lane queries, class scores, and relation matrices. It
uses confident predicted lane-lane edges to aggregate predecessor/successor
query features and endpoint continuity cues, then predicts a gated, bounded
11-point residual for each existing lane. No completion queries are emitted.

## OpenLane-V2 subset A validation

Both rows were produced locally on the complete 4,806-frame validation set
with the official `openlanev2==1.1.0` evaluator.

| model | OLS | DET_l | DET_t | TOP_ll | TOP_lt | Lane F-score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reproduced TopoNet E0 | 0.383147 | 0.268143 | 0.452547 | 0.102808 | 0.241339 | 0.149613 |
| E5 | **0.389696** | **0.273521** | **0.452607** | **0.114262** | **0.244658** | **0.149924** |

These are local reproductions under the official metric protocol, not the
TopoNet paper's published leaderboard number (approximately 0.398 OLS).

## Training

Set `load_from` in the E5 config to the reproduced 24-epoch TopoNet checkpoint,
then run the repository's distributed training command with two GPUs. Only the
new refinement head receives optimizer updates.
