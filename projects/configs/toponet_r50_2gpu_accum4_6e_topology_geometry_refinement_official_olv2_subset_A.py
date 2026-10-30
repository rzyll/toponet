_base_ = ['./toponet_r50_2gpu_accum4_6e_topology_geometry_refinement_olv2_subset_A.py']

# Strong-baseline validation of E5.  Keep the original E5 architecture,
# frozen parameter set and training recipe; only initialize the frozen
# TopoNet from the official released subset-A checkpoint.
load_from = 'ckpt/toponet_r50_8x1_24e_olv2_subset_A_official.pth'
