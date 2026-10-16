_base_ = ['./toponet_r50_8x1_24e_olv2_subset_A.py']

# Two GPUs x one sample/GPU x four accumulated micro-batches reproduces the
# official effective batch size of eight without scaling the base LR.
optimizer_config = dict(
    _delete_=True,
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=4,
    grad_clip=dict(max_norm=35, norm_type=2),
)

# The LR hook advances on every micro-iteration, while optimizer.step() runs
# every four micro-iterations. Preserve the official 500-step warmup.
lr_config = dict(warmup_iters=2000)
