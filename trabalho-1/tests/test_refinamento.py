"""Guards for inverse rendering and source-priority coverage."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refinar_unicamp import original_maps, prioritize
from step4_homography_connection.alignment import cylindrical_points


class RefinementTests(unittest.TestCase):
    def test_inverse_mapping_recovers_original_photo_coordinates(self):
        shape = (768, 1024)
        original = (3456, 4608)
        focal = 762.
        H = np.array([[.8, .1, 120], [-.1, .7, 250], [.0001, -.0002, 1.]])
        for px, py in [(100., 120.), (512., 384.), (900., 650.)]:
            cyl = cylindrical_points(np.array([[px, py]]), focal, shape)[0]
            dest = H @ np.r_[cyl, 1.]; dest = dest[:2]/dest[2]
            mx, my = original_maps(H, shape, original, focal, dest, (1, 1), 1)
            self.assertAlmostEqual(float(mx[0, 0]), (px+.5)*4.5-.5, places=3)
            self.assertAlmostEqual(float(my[0, 0]), (py+.5)*4.5-.5, places=3)

    def test_high_resolution_pixel_centers(self):
        H = np.eye(3)
        # Center pixel of an odd-scale rendering coincides with low-res center.
        mx, my = original_maps(H, (100, 120), (100, 120), 200., (60, 50), (3, 3), 3)
        self.assertAlmostEqual(float(mx[1, 1]), 60.)
        self.assertAlmostEqual(float(my[1, 1]), 50.)

    def test_priority_cannot_claim_invalid_pixels_or_remove_coverage(self):
        masks = [np.full((40, 50), 255, np.uint8), np.zeros((40, 50), np.uint8)]
        masks[1][10:30, 15:35] = 255
        seams = [masks[0].copy(), np.zeros_like(masks[1])]
        prioritize(seams, masks, ['a', 'b'], [{'name': 'b', 'polygon': [[0,0],[49,0],[49,39],[0,39]]}], (0,0,50,40))
        self.assertTrue(np.all(np.any(np.array(seams)>0, axis=0)))
        self.assertFalse(np.any((seams[1]>0)&(masks[1]==0)))
        self.assertEqual(int(seams[1][20,25]), 255)
        self.assertEqual(int(seams[0][20,25]), 0)
        self.assertEqual(int(seams[0][0,0]), 255)


if __name__ == '__main__':
    unittest.main()
