_base_ = ['./toponet_r50_2gpu_accum4_6e_topology_geometry_refinement_official_olv2_subset_A.py']

# E5-O input ablation. Keep the original E5 refinement module, parameter
# shapes, frozen TopoNet backbone, official initialization, optimizer, losses,
# six-epoch budget and seed. Sigmoid edge probabilities are <= 1, so a strict
# 1.1 cutoff removes all predicted incoming/outgoing edges. The remaining
# active path is the current lane query feature plus a trainable residual/gate.
# This tests the combined role of predicted-neighbor context and continuity
# cues; it does not separate them or equalize effective active input rank.
model = dict(lane_head=dict(geometry_refinement_cfg=dict(graph_threshold=1.1)))
