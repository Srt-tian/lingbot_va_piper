"""Conversions between the workstation inference protocol and LingBot-VA."""
import numpy as np

CAMERAS = ("top_head", "hand_left", "hand_right")
CHANNELS = [14, 15, 16, 17, 18, 19, 28, 21, 22, 23, 24, 25, 26, 29]


def validate_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("Inference payload must be a mapping")
    unsupported = set(payload) - {"images", "state", "prompt", "num_steps"}
    if unsupported:
        raise ValueError(f"LingBot supports base/sync requests, not RTC/Legato/EAPN fields: {sorted(unsupported)}")
    images = payload.get("images")
    if not isinstance(images, dict) or set(images) != set(CAMERAS):
        raise ValueError(f"images must contain exactly {CAMERAS}")
    observation = {}
    for camera in CAMERAS:
        image = np.asarray(images[camera])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"{camera}: expected RGB uint8 CHW image, got {image.shape} {image.dtype}")
        if not all(1 <= n <= 2048 for n in image.shape[1:]):
            raise ValueError(f"{camera}: invalid spatial resolution")
        observation[f"observation.images.{camera}"] = np.ascontiguousarray(image.transpose(1, 2, 0))
    state = np.asarray(payload.get("state"), dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("state must be a finite 14-dimensional vector")
    prompt = payload.get("prompt", "flat the cloth with both robot arms")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2048:
        raise ValueError("prompt must be a nonempty string of at most 2048 characters")
    steps = payload.get("num_steps", 10)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or not 1 <= int(steps) <= 50:
        raise ValueError("num_steps must be an integer between 1 and 50")
    return {"obs": [observation]}, state, prompt, int(steps)


def future_actions(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (14, 4, 12) or not np.isfinite(raw).all():
        raise ValueError(f"Invalid LingBot action output: {raw.shape}")
    return np.ascontiguousarray(raw[:, 1:, :].transpose(1, 2, 0).reshape(36, 14))
