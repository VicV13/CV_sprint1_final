# 1. Подключаем базовый конфиг стандартной сборки FCOS
_base_ = 'fcos_r50-caffe_fpn_gn-head_1x_coco.py'

# Указываем метаинформацию нашего датасета (17 классов мобов)
metainfo = dict(
    classes=('bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 
             'ghast', 'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 
             'turtle', 'wolf', 'zombie'),
    palette=[(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
             (0, 64, 0), (128, 64, 0)] # Фиксированные цвета для визуализации
)

# 2. Модифицируем базовую модель под 17 классов
model = dict(
    bbox_head=dict(
        num_classes=17
    )
)

# 4. Настраиваем пайплайн данных с уменьшением масштаба (img_scale=(512, 512))
# Используем простые и быстрые аугментации, чтобы ускорить обучение
img_scale = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True), # Сжатие картинки
    dict(type='RandomFlip', prob=0.5), # Простая базовая аугментация
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True), # Необходимо для валидации
    # dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    dict(type='PackDetInputs')
]

# Настройка загрузчиков данных (samples_per_gpu=2 -> batch_size=2, workers_per_gpu=2)
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root='datasets/minecraft/',
        ann_file='annotations/train/annotations.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root='datasets/minecraft/',
        ann_file='annotations/valid/annotations.json',
        data_prefix=dict(img='valid/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True
    )
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root='datasets/minecraft/',
        ann_file='annotations/test/annotations.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True
    )
)

# Настройка валидатора метрик
val_evaluator = dict(
    type='CocoMetric',
    ann_file='datasets/minecraft/annotations/val/annotations.json',
    metric='bbox'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='datasets/minecraft/annotations/test/annotations.json',
    metric='bbox'
)

# 5. Параметры процесса обучения (12 эпох, чекпоинт раз в эпоху)
max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Настройка сохранения чекпоинтов раз в эпоху
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50) # Писать логи каждые 50 итераций
)

# # Включаем оптимизатор (стандартный для FCOS - SGD c Momentum)
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# )

# Задаем половинчатую точность (FP16) для дикой экономии памяти и ускорения ВМ
fp16 = dict(loss_scale='dynamic')

# Настройки оптимизатора
optimizer = dict(type='SGD', lr=0.02, weight_decay=0.0001)

# Настройки скорости обучения
lr_config = dict(
    policy='step',
    step=[8, 11]  # Уменьшить LR на 8 и 11 эпохах
)

# Общая продолжительность обучения
runner = dict(type='EpochBasedRunner', max_epochs=12) 

# Куда сохранять логи и веса в ходе обучения
work_dir = './artifacts/fcos'