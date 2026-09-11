# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os
from pathlib import Path

from easydict import EasyDict
from .shared_config import va_shared_cfg

va_fold_clothes_cfg = EasyDict(__name__="Config: VA fold clothes")
va_fold_clothes_cfg.update(va_shared_cfg)

va_fold_clothes_cfg.wan22_pretrained_model_name_or_path = os.environ.get("MODEL_ROOT", "models/base")
va_fold_clothes_cfg.attn_window = 30
va_fold_clothes_cfg.frame_chunk_size = 4
va_fold_clothes_cfg.env_type = "none"

va_fold_clothes_cfg.height = 256
va_fold_clothes_cfg.width = 256
va_fold_clothes_cfg.action_dim = 30
va_fold_clothes_cfg.action_per_frame = 12
va_fold_clothes_cfg.obs_cam_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

va_fold_clothes_cfg.guidance_scale = 5
va_fold_clothes_cfg.action_guidance_scale = 1
va_fold_clothes_cfg.num_inference_steps = 5
va_fold_clothes_cfg.video_exec_step = -1
va_fold_clothes_cfg.action_num_inference_steps = 10
va_fold_clothes_cfg.snr_shift = 5.0
va_fold_clothes_cfg.action_snr_shift = 1.0

# Raw action order is left 6 joints, left gripper, right 6 joints, right gripper.
va_fold_clothes_cfg.used_action_channel_ids = (
    list(range(14, 20)) + [28] + list(range(21, 27)) + [29]
)
inverse_used_action_channel_ids = [len(va_fold_clothes_cfg.used_action_channel_ids)] * va_fold_clothes_cfg.action_dim
for i, j in enumerate(va_fold_clothes_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_fold_clothes_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_fold_clothes_cfg.action_norm_method = "quantiles"
va_fold_clothes_cfg.norm_stat = {
    "q01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.933681309223175, 0.32410764694213867, -2.555510997772217, -1.3805753540992736, -0.14493213593959808, -0.8497135639190674, 0.0007999999797903001, -0.22907446324825287, 0.32271137833595276, -2.603717088699341, -0.629697322845459, -0.12199851125478745, -0.9569291472434998, 0.000699999975040555, 0.0, 0.0],
    "q99": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2710147202014923, 2.469780445098877, -0.3611260652542114, 0.7884175777435303, 1.236914873123169, 1.301246138811111, 0.07729999721050262, 0.9119519591331482, 2.4471707487106302, -0.36332517862319946, 0.9795299082994359, 1.2355884313583374, 0.70621258020401, 0.07029999792575836, 0.0, 0.0],
}

_fixed_norm_path = Path(os.environ.get("LINGBOT_NORM_PATH",
    str(Path(__file__).resolve().parents[2] / "meta/lingbot_action_norm.json")))
if _fixed_norm_path.is_file():
    with _fixed_norm_path.open("r", encoding="utf-8") as _norm_file:
        va_fold_clothes_cfg.norm_stat = json.load(_norm_file)
