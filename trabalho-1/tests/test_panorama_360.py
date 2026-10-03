"""Geometria periódica e rejeição de uma coleta sem volta completa."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panorama_360 import spherical_rays, inverse_maps, full_coverage_band, connected


class Panorama360Tests(unittest.TestCase):
    def test_rays_are_unit_and_close_at_wrap(self):
        rays = spherical_rays(360, 90, 91)[0]
        np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-6)
        # The cyclic last/first separation must equal one interior angular pixel.
        self.assertAlmostEqual(float(np.linalg.norm(rays[-1]-rays[0])),
                               float(np.linalg.norm(rays[1]-rays[0])), places=5)

    def test_partial_panorama_is_rejected(self):
        camera = dict(R=np.eye(3), shape=[1200, 676], focal=800)
        coverage = inverse_maps(camera, spherical_rays(360), (1200, 676))[2]
        with self.assertRaisesRegex(ValueError, '360'):
            full_coverage_band(coverage)

    def test_full_rotation_is_covered(self):
        rays = spherical_rays(360)
        coverage = np.zeros(rays.shape[:2], np.uint16)
        for theta in np.linspace(0, 2*np.pi, 24, endpoint=False):
            c, s = np.cos(theta), np.sin(theta)
            camera = dict(R=[[c, 0, s], [0, 1, 0], [-s, 0, c]], shape=[1200, 676], focal=800)
            coverage += inverse_maps(camera, rays, (1200, 676))[2] > 0
        first, last = full_coverage_band(coverage)
        self.assertLess(first, 90)
        self.assertGreater(last, 90)
        self.assertTrue(np.all(coverage[first:last] > 0))

    def test_original_resolution_uses_resize_pixel_centers(self):
        camera = dict(R=np.eye(3), shape=[1200, 676], focal=800)
        ray = np.array([[[0.1, 0.2, 1.0]]], np.float32)
        x, y, mask = inverse_maps(camera, ray, (4000, 2252))
        self.assertAlmostEqual(float(x[0, 0]), (80+338+.5)*2252/676-.5, places=3)
        self.assertAlmostEqual(float(y[0, 0]), (160+600+.5)*4000/1200-.5, places=3)
        self.assertEqual(mask[0, 0], 255)

    def test_disconnected_graph_is_rejected(self):
        self.assertFalse(connected(4, [(0, 1), (2, 3)]))
        self.assertTrue(connected(4, [(0, 1), (1, 2), (2, 3), (3, 0)]))


if __name__ == '__main__':
    unittest.main()
