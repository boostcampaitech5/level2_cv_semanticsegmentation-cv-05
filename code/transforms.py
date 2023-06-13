import albumentations as A

aug_list = [
    A.Resize(512, 512),
    A.GridDropout(holes_number_x=15, holes_number_y=15, p=0.7),
    A.PixelDropout(
        dropout_prob=0.1,
        per_channel=False,
        drop_value=0,
        mask_drop_value=None,
        always_apply=False,
        p=0.5,
    ),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
    A.Solarize(threshold=128, always_apply=False, p=0.5),
    A.RandomBrightness(limit=0.3, always_apply=False, p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.9,
        brightness_by_max=True,
        always_apply=False,
        p=0.5,
    ),
    A.RandomContrast(limit=0.9, always_apply=False, p=0.5),
    A.Spatter(
        mean=0.65,
        std=0.3,
        gauss_sigma=2,
        cutout_threshold=0.68,
        intensity=0.6,
        mode="rain",
        always_apply=False,
        p=0.5,
    ),
]


def baseaugmentation():
    return A.Compose([aug_list[0]])


def base_grid_mask_augmentation():
    return A.Compose([aug_list[0], aug_list[1]])


def base_pixeldropout_augmentation():
    return A.Compose([aug_list[0], aug_list[2]])


def base_clahe_augmentation():
    return A.Compose(
        [
            A.Resize(512, 512),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
        ]
    )


def base_sharpen_augmentation():
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Sharpen(
                alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=1.0
            ),
        ]
    )


def base_solarize_augmentation():
    return A.Compose(
        [A.Resize(512, 512), A.Solarize(threshold=128, always_apply=False, p=1.0)]
    )
