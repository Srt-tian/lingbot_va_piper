# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_fold_clothes_train_cfg import va_fold_clothes_train_cfg

va_fold_clothes_smoke_cfg = EasyDict(__name__="Config: VA fold clothes smoke")
va_fold_clothes_smoke_cfg.update(va_fold_clothes_train_cfg)
va_fold_clothes_smoke_cfg.obs_cam_keys = ["observation.images.cam_high"]
va_fold_clothes_smoke_cfg.load_worker = 0
va_fold_clothes_smoke_cfg.dataset_init_worker = 1
va_fold_clothes_smoke_cfg.gradient_accumulation_steps = 1
va_fold_clothes_smoke_cfg.num_steps = 1
va_fold_clothes_smoke_cfg.save_interval = 1000
va_fold_clothes_smoke_cfg.disable_fsdp = True
