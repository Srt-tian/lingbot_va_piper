# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_fold_clothes_train_cfg import va_fold_clothes_train_cfg

va_fold_clothes_train_200_cfg = EasyDict(__name__="Config: VA fold clothes train first 200")
va_fold_clothes_train_200_cfg.update(va_fold_clothes_train_cfg)

# First training pass: only use the first 200 episodes while full latent extraction continues.
va_fold_clothes_train_200_cfg.episode_start = 0
va_fold_clothes_train_200_cfg.episode_end = 200
