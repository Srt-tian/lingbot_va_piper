import unittest
import numpy as np
import torch
import torch.nn.functional as F
from training_preprocessing import resize_observation

class PreprocessingTests(unittest.TestCase):
    def test_area_antialiasing_rgb_and_existing_bilinear_identity(self):
        # A 3x3 block has one red impulse: area should average to 28, not sample 255.
        rgb=np.zeros((6,6,3),np.uint8);rgb[:,:,1]=42;rgb[:,:,2]=7;rgb[1::3,1::3,0]=255
        result=resize_observation({'obs':[{'left':rgb,'right':rgb.copy()}]},2,2)
        for image in result['obs'][0].values():
            np.testing.assert_array_equal(image,np.broadcast_to([28,42,7],(2,2,3)))
            t=torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0)
            # The unchanged model still calls interpolate, now at the identical size.
            torch.testing.assert_close(F.interpolate(t,size=(2,2),mode='bilinear',align_corners=False),t,rtol=0,atol=0)
        self.assertEqual(rgb[1,1,0],255)

if __name__=='__main__':unittest.main()
