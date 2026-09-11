# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_fold_clothes_train_cfg import va_fold_clothes_train_cfg

va_fold_clothes_train_100_cfg = EasyDict(__name__="Config: VA fold clothes train first 100")
va_fold_clothes_train_100_cfg.update(va_fold_clothes_train_cfg)

# Faster first training pass: only use the first 100 episodes.
va_fold_clothes_train_100_cfg.episode_start = 0
va_fold_clothes_train_100_cfg.episode_end = 100
