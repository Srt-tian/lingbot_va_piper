# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_fold_clothes_train_cfg import va_fold_clothes_train_cfg

va_fold_clothes_train_400_cfg = EasyDict(__name__="Config: VA fold clothes train first 400")
va_fold_clothes_train_400_cfg.update(va_fold_clothes_train_cfg)

va_fold_clothes_train_400_cfg.episode_start = 0
va_fold_clothes_train_400_cfg.episode_end = 400
