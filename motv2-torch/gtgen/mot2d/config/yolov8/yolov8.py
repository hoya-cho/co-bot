img_scale = (960, 960)

Configs = dict(
    default_scope = 'mmyolo',
    default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(
            type='YOLOv5ParamSchedulerHook',
            scheduler_type='linear',
            lr_factor=0.01,
            max_epochs=30),
        checkpoint=dict(
            type='CheckpointHook', interval=1, save_best='auto',
            max_keep_ckpts=10),
        # earlystopping=dict(
        #     type='EarlyStoppingHook',
        #         monitor='coco/bbox_mAP',
        #         rule='greater',
        #         min_delta=0,
        #         patience=7),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='mmdet.DetVisualizationHook')),
    env_cfg = dict(
        cudnn_benchmark=True,
        mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl')),
    vis_backends = [dict(type='LocalVisBackend')],
    visualizer = dict(
        type='mmdet.DetLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer'),
    log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True),
    log_level = 'INFO',
    #수정
    load_from = None,
    resume = None,
    file_client_args = dict(backend='disk'),
    _file_client_args = dict(backend='disk'),
    tta_model = dict(
        type='mmdet.DetTTAModel',
        tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300)),
    img_scales = [(640, 640), (320, 320), (960, 960)],
    _multiscale_resize_transforms = [
        dict(
            type='Compose',
            transforms=[
                dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
                dict(
                    type='LetterResize',
                    scale=(640, 640),
                    allow_scale_up=False,
                    pad_val=dict(img=114))
            ]),
        dict(
            type='Compose',
            transforms=[
                dict(type='YOLOv5KeepRatioResize', scale=(320, 320)),
                dict(
                    type='LetterResize',
                    scale=(320, 320),
                    allow_scale_up=False,
                    pad_val=dict(img=114))
            ]),
        dict(
            type='Compose',
            transforms=[
                dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
                dict(
                    type='LetterResize',
                    scale=(960, 960),
                    allow_scale_up=False,
                    pad_val=dict(img=114))
            ])
    ],
    tta_pipeline = [
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(
            type='TestTimeAug',
            transforms=[[{
                'type':
                'Compose',
                'transforms': [{
                    'type': 'YOLOv5KeepRatioResize',
                    'scale': (640, 640)
                }, {
                    'type': 'LetterResize',
                    'scale': (640, 640),
                    'allow_scale_up': False,
                    'pad_val': {
                        'img': 114
                    }
                }]
            }, {
                'type':
                'Compose',
                'transforms': [{
                    'type': 'YOLOv5KeepRatioResize',
                    'scale': (320, 320)
                }, {
                    'type': 'LetterResize',
                    'scale': (320, 320),
                    'allow_scale_up': False,
                    'pad_val': {
                        'img': 114
                    }
                }]
            }, {
                'type':
                'Compose',
                'transforms': [{
                    'type': 'YOLOv5KeepRatioResize',
                    'scale': (960, 960)
                }, {
                    'type': 'LetterResize',
                    'scale': (960, 960),
                    'allow_scale_up': False,
                    'pad_val': {
                        'img': 114
                    }
                }]
            }],
                        [{
                            'type': 'mmdet.RandomFlip',
                            'prob': 1.0
                        }, {
                            'type': 'mmdet.RandomFlip',
                            'prob': 0.0
                        }], [{
                            'type': 'mmdet.LoadAnnotations',
                            'with_bbox': True
                        }],
                        [{
                            'type':
                            'mmdet.PackDetInputs',
                            'meta_keys':
                            ('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'pad_param', 'flip', 'flip_direction')
                        }]])
    ],
    
    model_test_cfg = dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300),
    img_scale = img_scale,
    model = dict(
        type='YOLODetector',
        data_preprocessor=dict(
            type='YOLOv5DetDataPreprocessor',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            bgr_to_rgb=True),
        backbone=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=512,
            deepen_factor=1.0,
            widen_factor=1.25,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        neck=dict(
            type='YOLOv8PAFPN',
            deepen_factor=1.0,
            widen_factor=1.25,
            in_channels=[256, 512, 512],
            out_channels=[256, 512, 512],
            num_csp_blocks=3,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        bbox_head=dict(
            type='YOLOv8Head',
            head_module=dict(
                type='YOLOv8HeadModule',
                num_classes=1,
                in_channels=[256, 512, 512],
                widen_factor=1.25,
                reg_max=16,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='SiLU', inplace=True),
                featmap_strides=[8, 16, 32]),
            prior_generator=dict(
                type='mmdet.MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
            bbox_coder=dict(type='DistancePointBBoxCoder'),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='none',
                loss_weight=0.5),
            loss_bbox=dict(
                type='IoULoss',
                iou_mode='ciou',
                bbox_format='xyxy',
                reduction='sum',
                loss_weight=7.5,
                return_iou=False),
            loss_dfl=dict(
                type='mmdet.DistributionFocalLoss',
                reduction='mean',
                loss_weight=0.375)),
        train_cfg=dict(
            assigner=dict(
                type='BatchTaskAlignedAssigner',
                num_classes=1,
                use_ciou=True,
                topk=10,
                alpha=0.5,
                beta=6.0,
                eps=1e-09)),
        test_cfg=dict(
            multi_label=True,
            nms_pre=30000,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=300)),
    albu_train_transforms = [
        dict(type='Blur', p=0.01),
        dict(type='MedianBlur', p=0.01),
        dict(type='ToGray', p=0.01),
        dict(type='CLAHE', p=0.01)
    ],
    pre_transform = [
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(
            type='LoadAnnotations', with_bbox=True, with_mask=True, mask2bbox=True)
    ],
    last_transform = [
        dict(type='RemoveDataElement', keys=['gt_masks']),
        dict(
            type='mmdet.Albu',
            transforms=[
                dict(type='Blur', p=0.01),
                dict(type='MedianBlur', p=0.01),
                dict(type='ToGray', p=0.01),
                dict(type='CLAHE', p=0.01)
            ],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
            keymap=dict(img='image', gt_bboxes='bboxes')),
        dict(type='YOLOv5HSVRandomAug'),
        dict(type='mmdet.RandomFlip', prob=0.5),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
    ],
    train_pipeline = [
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(
            type='LoadAnnotations', with_bbox=True, with_mask=True,
            mask2bbox=True),
        dict(
            type='Mosaic',
            img_scale=img_scale,
            pad_val=114.0,
            pre_transform=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    mask2bbox=True)
            ]),
        dict(type='YOLOv5CopyPaste', prob=0.3),
        dict(
            type='YOLOv5RandomAffine',
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            max_aspect_ratio=100.0,
            scaling_ratio_range=(0.09999999999999998, 1.9),
            border=(-480, -480),
            border_val=(114, 114, 114),
            min_area_ratio=0.01,
            use_mask_refine=True),
        dict(
            type='YOLOv5MixUp',
            prob=0.15,
            pre_transform=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    mask2bbox=True),
                dict(
                    type='Mosaic',
                    img_scale=img_scale,
                    pad_val=114.0,
                    pre_transform=[
                        dict(
                            type='LoadImageFromFile',
                            file_client_args=dict(backend='disk')),
                        dict(
                            type='LoadAnnotations',
                            with_bbox=True,
                            with_mask=True,
                            mask2bbox=True)
                    ]),
                dict(type='YOLOv5CopyPaste', prob=0.3),
                dict(
                    type='YOLOv5RandomAffine',
                    max_rotate_degree=0.0,
                    max_shear_degree=0.0,
                    max_aspect_ratio=100.0,
                    scaling_ratio_range=(0.09999999999999998, 1.9),
                    border=(-480, -480),
                    border_val=(114, 114, 114),
                    min_area_ratio=0.01,
                    use_mask_refine=True)
            ]),
        dict(type='RemoveDataElement', keys=['gt_masks']),
        dict(
            type='mmdet.Albu',
            transforms=[
                dict(type='Blur', p=0.01),
                dict(type='MedianBlur', p=0.01),
                dict(type='ToGray', p=0.01),
                dict(type='CLAHE', p=0.01)
            ],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
            keymap=dict(img='image', gt_bboxes='bboxes')),
        dict(type='YOLOv5HSVRandomAug'),
        dict(type='mmdet.RandomFlip', prob=0.5),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
    ],
    train_pipeline_stage2 = [
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(
            type='LoadAnnotations', with_bbox=True, with_mask=True,
            mask2bbox=True),
        dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
        dict(
            type='LetterResize',
            scale=(960, 960),
            allow_scale_up=True,
            pad_val=dict(img=114.0)),
        dict(
            type='YOLOv5RandomAffine',
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(0.09999999999999998, 1.9),
            max_aspect_ratio=100,
            border_val=(114, 114, 114),
            min_area_ratio=0.01,
            use_mask_refine=True),
        dict(type='RemoveDataElement', keys=['gt_masks']),
        dict(
            type='mmdet.Albu',
            transforms=[
                dict(type='Blur', p=0.01),
                dict(type='MedianBlur', p=0.01),
                dict(type='ToGray', p=0.01),
                dict(type='CLAHE', p=0.01)
            ],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
            keymap=dict(img='image', gt_bboxes='bboxes')),
        dict(type='YOLOv5HSVRandomAug'),
        dict(type='mmdet.RandomFlip', prob=0.5),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
    ],
    train_dataloader = dict(
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='yolov5_collate'),
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root='/',
            metainfo=dict(
                classes=(None)
                ),
            ann_file=None,
            data_prefix=None,
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    mask2bbox=True),
                dict(
                    type='Mosaic',
                    img_scale=img_scale,
                    pad_val=114.0,
                    pre_transform=[
                        dict(
                            type='LoadImageFromFile',
                            file_client_args=dict(backend='disk')),
                        dict(
                            type='LoadAnnotations',
                            with_bbox=True,
                            with_mask=True,
                            mask2bbox=True)
                    ]),
                dict(type='YOLOv5CopyPaste', prob=0.3),
                dict(
                    type='YOLOv5RandomAffine',
                    max_rotate_degree=0.0,
                    max_shear_degree=0.0,
                    max_aspect_ratio=100.0,
                    scaling_ratio_range=(0.09999999999999998, 1.9),
                    border=(-480, -480),
                    border_val=(114, 114, 114),
                    min_area_ratio=0.01,
                    use_mask_refine=True),
                dict(
                    type='YOLOv5MixUp',
                    prob=0.15,
                    pre_transform=[
                        dict(
                            type='LoadImageFromFile',
                            file_client_args=dict(backend='disk')),
                        dict(
                            type='LoadAnnotations',
                            with_bbox=True,
                            with_mask=True,
                            mask2bbox=True),
                        dict(
                            type='Mosaic',
                            img_scale=img_scale,
                            pad_val=114.0,
                            pre_transform=[
                                dict(
                                    type='LoadImageFromFile',
                                    file_client_args=dict(backend='disk')),
                                dict(
                                    type='LoadAnnotations',
                                    with_bbox=True,
                                    with_mask=True,
                                    mask2bbox=True)
                            ]),
                        dict(type='YOLOv5CopyPaste', prob=0.3),
                        dict(
                            type='YOLOv5RandomAffine',
                            max_rotate_degree=0.0,
                            max_shear_degree=0.0,
                            max_aspect_ratio=100.0,
                            scaling_ratio_range=(0.09999999999999998, 1.9),
                            border=(-480, -480),
                            border_val=(114, 114, 114),
                            min_area_ratio=0.01,
                            use_mask_refine=True)
                    ]),
                dict(type='RemoveDataElement', keys=['gt_masks']),
                dict(
                    type='mmdet.Albu',
                    transforms=[
                        dict(type='Blur', p=0.01),
                        dict(type='MedianBlur', p=0.01),
                        dict(type='ToGray', p=0.01),
                        dict(type='CLAHE', p=0.01)
                    ],
                    bbox_params=dict(
                        type='BboxParams',
                        format='pascal_voc',
                        label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
                    keymap=dict(img='image', gt_bboxes='bboxes')),
                dict(type='YOLOv5HSVRandomAug'),
                dict(type='mmdet.RandomFlip', prob=0.5),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'flip', 'flip_direction'))
            ])),
    test_pipeline = [
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
        dict(
            type='LetterResize',
            scale=(960, 960),
            allow_scale_up=False,
            pad_val=dict(img=114)),
        dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
    ],
    val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=None,
            metainfo=dict(
                classes=(None)
                ),
            test_mode=True,
            data_prefix=None,
            ann_file=None,
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk')),
                dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
                dict(
                    type='LetterResize',
                    scale=(960, 960),
                    allow_scale_up=False,
                    pad_val=dict(img=114)),
                dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'pad_param'))
            ],
            batch_shapes_cfg=None)),
    test_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=None,
            metainfo=dict(
                classes=('Vehicle','Pedestrian')
                ),
            test_mode=True,
            data_prefix=None,
            ann_file=None,
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk')),
                dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
                dict(
                    type='LetterResize',
                    scale=(960, 960),
                    allow_scale_up=False,
                    pad_val=dict(img=114)),
                dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'pad_param'))
            ],
            batch_shapes_cfg=None)),
    param_scheduler = None,
    optim_wrapper = dict(
        type='OptimWrapper',
        clip_grad=dict(max_norm=10.0),
        optimizer=dict(
            type='SGD',
            lr=0.0001,
            momentum=0.937,
            weight_decay=0.0005,
            nesterov=True,
            batch_size_per_gpu=4),
        constructor='YOLOv5OptimizerConstructor'),
    custom_hooks = [
        dict(
            type='EMAHook',
            ema_type='ExpMomentumEMA',
            momentum=0.0001,
            update_buffers=True,
            strict_load=False,
            priority=49),
        dict(
            type='mmdet.PipelineSwitchHook',
            switch_epoch=27,
            switch_pipeline=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    mask2bbox=True),
                dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
                dict(
                    type='LetterResize',
                    scale=(960, 960),
                    allow_scale_up=True,
                    pad_val=dict(img=114.0)),
                dict(
                    type='YOLOv5RandomAffine',
                    max_rotate_degree=0.0,
                    max_shear_degree=0.0,
                    scaling_ratio_range=(0.09999999999999998, 1.9),
                    max_aspect_ratio=100,
                    border_val=(114, 114, 114),
                    min_area_ratio=0.01,
                    use_mask_refine=True),
                dict(type='RemoveDataElement', keys=['gt_masks']),
                dict(
                    type='mmdet.Albu',
                    transforms=[
                        dict(type='Blur', p=0.01),
                        dict(type='MedianBlur', p=0.01),
                        dict(type='ToGray', p=0.01),
                        dict(type='CLAHE', p=0.01)
                    ],
                    bbox_params=dict(
                        type='BboxParams',
                        format='pascal_voc',
                        label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
                    keymap=dict(img='image', gt_bboxes='bboxes')),
                dict(type='YOLOv5HSVRandomAug'),
                dict(type='mmdet.RandomFlip', prob=0.5),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'flip', 'flip_direction'))
            ])
    ],
    val_evaluator = dict(
        type='mmdet.CocoMetric',
        proposal_nums=(100, 1, 10),
        ann_file=None,
        metric='bbox'),
    test_evaluator = dict(
        type='mmdet.CocoMetric',
        proposal_nums=(100, 1, 10),
        ann_file=None,
        metric='bbox'),
    train_cfg = dict(
        type='EpochBasedTrainLoop',
        max_epochs=30,
        val_interval=1,
        dynamic_intervals=[(27, 1)]),
    val_cfg = dict(type='ValLoop'),
    test_cfg = dict(type='TestLoop'),
    use_mask2refine = True,
    min_area_ratio = 0.01,
    mixup_prob = 0.15,
    copypaste_prob = 0.3,
    mosaic_affine_transform = [
        dict(
            type='Mosaic',
            img_scale=img_scale,
            pad_val=114.0,
            pre_transform=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    mask2bbox=True)
            ]),
        dict(type='YOLOv5CopyPaste', prob=0.3),
        dict(
            type='YOLOv5RandomAffine',
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            max_aspect_ratio=100.0,
            scaling_ratio_range=(0.09999999999999998, 1.9),
            border=(-480, -480),
            border_val=(114, 114, 114),
            min_area_ratio=0.01,
            use_mask_refine=True)
    ],
    launcher = 'none',
    work_dir = './work_dirs/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco'
)