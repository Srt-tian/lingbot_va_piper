# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg
from .va_fold_clothes_cfg import va_fold_clothes_cfg
from .va_fold_clothes_train_cfg import va_fold_clothes_train_cfg
from .va_fold_clothes_train_100_cfg import va_fold_clothes_train_100_cfg
from .va_fold_clothes_train_400_cfg import va_fold_clothes_train_400_cfg
from .va_fold_clothes_train_500_cfg import va_fold_clothes_train_500_cfg
from .va_fold_clothes_train_200_cfg import va_fold_clothes_train_200_cfg
from .va_fold_clothes_smoke_cfg import va_fold_clothes_smoke_cfg
from .va_fold_clothes_i2va import va_fold_clothes_i2va_cfg

VA_CONFIGS = {
    "robotwin": va_robotwin_cfg,
    "franka": va_franka_cfg,
    "robotwin_i2av": va_robotwin_i2va_cfg,
    "franka_i2av": va_franka_i2va_cfg,
    "robotwin_train": va_robotwin_train_cfg,
    "demo": va_demo_cfg,
    "demo_train": va_demo_train_cfg,
    "demo_i2av": va_demo_i2va_cfg,
    "libero": va_libero_cfg,
    "libero_train": va_libero_train_cfg,
    "libero_i2av": va_libero_i2va_cfg,
    "fold_clothes": va_fold_clothes_cfg,
    "fold_clothes_train": va_fold_clothes_train_cfg,
    "fold_clothes_train_100": va_fold_clothes_train_100_cfg,
    "fold_clothes_train_400": va_fold_clothes_train_400_cfg,
    "fold_clothes_train_500": va_fold_clothes_train_500_cfg,
    "fold_clothes_train_200": va_fold_clothes_train_200_cfg,
    "fold_clothes_smoke": va_fold_clothes_smoke_cfg,
    "fold_clothes_i2va": va_fold_clothes_i2va_cfg,
}
