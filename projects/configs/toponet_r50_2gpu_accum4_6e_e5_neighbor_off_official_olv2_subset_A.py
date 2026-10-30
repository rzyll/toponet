_base_ = ["./toponet_r50_2gpu_accum4_6e_topology_geometry_refinement_official_olv2_subset_A.py"]

# E5 single-input ablation: preserve predicted graph endpoint-continuity
# offsets and graph-presence bits, but mask incoming/outgoing query features.
# This tests whether neighbor embeddings add value beyond endpoint cues.
model = dict(lane_head=dict(geometry_refinement_cfg=dict(use_neighbor_features=False)))
