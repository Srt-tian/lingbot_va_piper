"""Match tools/extract_fold_latents.py RGB uint8 spatial preprocessing."""
import cv2
import numpy as np


def resize_observation(observation, height=256, width=256):
    return {'obs': [{key: np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
                     for key, image in frame.items()} for frame in observation['obs']]}
