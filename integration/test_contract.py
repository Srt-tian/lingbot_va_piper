import unittest
import numpy as np
from contract import CAMERAS, future_actions, validate_request


class ContractTests(unittest.TestCase):
    def payload(self):
        return {"images": {k: np.full((3, 5, 7), i + 1, dtype=np.uint8) for i, k in enumerate(reversed(CAMERAS))},
                "state": np.zeros(14), "prompt": "flat the cloth", "num_steps": 10}

    def test_camera_keys_control_order_not_dict_insertion(self):
        obs, _, _, _ = validate_request(self.payload())
        self.assertEqual(list(obs["obs"][0]), [f"observation.images.{k}" for k in CAMERAS])
        self.assertEqual(obs["obs"][0]["observation.images.top_head"].shape, (5, 7, 3))
        self.assertEqual(int(obs["obs"][0]["observation.images.hand_right"][0, 0, 0]), 1)

    def test_conditioning_block_and_channel_order(self):
        raw = np.arange(14 * 4 * 12).reshape(14, 4, 12)
        action = future_actions(raw)
        self.assertEqual(action.shape, (36, 14))
        np.testing.assert_array_equal(action[0], np.arange(14) * 48 + 12)
        np.testing.assert_array_equal(action[-1], np.arange(14) * 48 + 47)

    def test_rejects_missing_camera_bad_state_and_unsupported_mode(self):
        for change in [lambda p: p["images"].pop("hand_left"),
                       lambda p: p.update(state=np.full(14, np.nan)),
                       lambda p: p.update(enable_rtc=True),
                       lambda p: p.update(num_steps=10.5)]:
            p = self.payload(); change(p)
            with self.assertRaises(ValueError): validate_request(p)

    def test_rejects_float_images_and_nonfinite_output(self):
        p = self.payload(); p["images"]["top_head"] = p["images"]["top_head"].astype(float)
        with self.assertRaises(ValueError): validate_request(p)
        with self.assertRaises(ValueError): future_actions(np.full((14, 4, 12), np.inf))


if __name__ == "__main__": unittest.main()
