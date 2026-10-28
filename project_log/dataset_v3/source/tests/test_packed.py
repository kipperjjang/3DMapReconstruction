"""Storage accuracy and frame reader checks, compatible with old and dyna Python."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
from neural_scene import packed


class PackedTests(unittest.TestCase):
    def test_cell_identity_error_bound_and_multiplicity(self):
        rng = np.random.default_rng(413)
        points = rng.uniform(-1.6, 1.6, (10000,3)).astype(np.float32)
        points = np.concatenate((points, points[:100]))
        encoded = packed.encode(points)
        decoded = packed.decode(encoded)
        self.assertEqual(len(decoded),len(points))
        np.testing.assert_array_equal(np.sort(packed.keys(decoded)),np.sort(packed.keys(points)))
        # Same sorting policy allows a strict per-observation error bound.
        c = packed.cells(points)
        q = np.clip(np.floor(((points.astype(np.float64)-packed.ORIGIN)/.05-c)*256),0,255)
        order = np.lexsort((q[:,2],q[:,1],q[:,0],c[:,2],c[:,1],c[:,0]))
        self.assertLess(float(np.abs(points[order]-decoded).max()),.05/512+1e-6)
        self.assertLess(float(np.linalg.norm(points[order]-decoded,axis=1).max()),.000171)

    def test_packed_frames_are_isolated_and_decoded(self):
        try:
            import h5py
        except ImportError:
            self.skipTest('h5py is provided by the dyna collection/training environment')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'frames.h5'
            points = np.array([[0,0,0],[1,1,1]],dtype=np.float32)
            with h5py.File(str(path),'w') as f:
                for kind in ('current','target'):
                    f[kind] = packed.encode(points).T
                    f[kind+'_offsets'] = [0,1,2]
                for field in ('robot_pose','sensor_pose','transformation'):
                    f[field] = np.repeat(np.eye(4)[None],2,axis=0)
                f['timestamp_ns'] = [0,100000000]
            frame = packed.read_frame(path,1)
            self.assertEqual(int(frame['timestamp_ns']),100000000)
            np.testing.assert_allclose(frame['current_points'],[[1,1,1]],atol=.0001)


if __name__=='__main__':
    unittest.main()
