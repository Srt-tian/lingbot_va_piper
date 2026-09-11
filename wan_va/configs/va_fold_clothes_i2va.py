# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_fold_clothes_cfg import va_fold_clothes_cfg

va_fold_clothes_i2va_cfg = EasyDict(__name__="Config: VA fold clothes i2va")
va_fold_clothes_i2va_cfg.update(va_fold_clothes_cfg)

va_fold_clothes_i2va_cfg.input_img_path = "example/fold_clothes_ep0"
va_fold_clothes_i2va_cfg.num_chunks_to_infer = 1
va_fold_clothes_i2va_cfg.prompt = "fold clothes with both robot arms"
va_fold_clothes_i2va_cfg.infer_mode = "i2va"
va_fold_clothes_i2va_cfg.resume_from = "/pfs/user/magiclab_works/lingbot_va/fold_clothes_train_100_20260716_065434/checkpoints/checkpoint_step_100"
va_fold_clothes_i2va_cfg.save_root = "/pfs/user/code/lingbot-va/train_out/fold_infer_step100"

# Keep first inference cheap for smoke; increase after the path is verified.
va_fold_clothes_i2va_cfg.num_inference_steps = 2
va_fold_clothes_i2va_cfg.action_num_inference_steps = 2
