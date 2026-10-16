_base_ = ['./toponet_r50_2gpu_accum4_6e_topology_geometry_refinement_olv2_subset_A.py']

# E9 starts from E5 and trains only a zero-initialized BEV re-query stage.
load_from = 'work_dirs/toponet_r50_topology_geometry_refinement_6e/epoch_6.pth'

model = dict(
    lane_head=dict(
        bev_requery_cfg=dict(
            num_samples=5,
            sample_radius=1.5,
            learned_offset_scale=2.0,
            residual_scale=(2.0, 2.0, 0.5),
            graph_threshold=0.5,
            detach_inputs=True,
            detach_graph=True,
            num_heads=8,
            ffn_dims=512,
            dropout=0.1,
            target_cfg=dict(hard_distance=2.0, hard_weight=3.0),
            loss_weights=dict(
                reg=1.0, chamfer=0.10, smooth=0.05, residual=0.01))))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.0),
        'img_neck': dict(lr_mult=0.0),
        'bev_constructor': dict(lr_mult=0.0),
        'bbox_head': dict(lr_mult=0.0),
        'pts_bbox_head': dict(lr_mult=0.0),
        'pts_bbox_head.bev_requery_head': dict(lr_mult=1.0),
    }),
    weight_decay=0.01)

total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=6)
checkpoint_config = dict(
    _delete_=True, by_epoch=False, interval=2000,
    max_keep_ckpts=4, save_last=True)
