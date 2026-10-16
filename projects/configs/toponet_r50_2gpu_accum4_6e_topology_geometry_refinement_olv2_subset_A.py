_base_ = ['./toponet_r50_2gpu_accum4_24e_olv2_subset_A.py']

# E5 refines the existing 200 lane queries and emits no additional nodes.
load_from = 'work_dirs/toponet_r50_accum4_24e/epoch_24.pth'

model = dict(
    lane_head=dict(
        geometry_refinement_cfg=dict(
            residual_scale=(6.0, 6.0, 1.0),
            detach_inputs=True,
            detach_graph=True,
            graph_threshold=0.5,
            gate_bias=-2.0,
            target_cfg=dict(
                hard_distance=2.0,
                hard_weight=3.0),
            loss_weights=dict(
                reg=1.0,
                chamfer=0.10,
                smooth=0.05,
                residual=0.01))))

# Freeze TopoNet and train only the identity-initialized refinement head.
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'bev_constructor': dict(lr_mult=0.0),
            'bbox_head': dict(lr_mult=0.0),
            'pts_bbox_head': dict(lr_mult=0.0),
            'pts_bbox_head.geometry_refinement_head': dict(lr_mult=1.0),
        }),
    weight_decay=0.01)

total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=6)
checkpoint_config = dict(
    _delete_=True,
    by_epoch=False,
    interval=2000,
    max_keep_ckpts=4,
    save_last=True)
