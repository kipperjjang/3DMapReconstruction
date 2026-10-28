"""Run with DynaRaySim's Python; exercises the real current DynaMapping API."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.find_spec('torch')

# Preparation requires modern torch; the network's Python 3.7 test suite must
# continue to run without importing DynaMapping's Python 3.9+ implementation.
@unittest.skipIf(sys.version_info < (3, 9), 'Preparation uses DynaRaySim Python >=3.9')
class PreparationTests(unittest.TestCase):
    def test_nonidentity_extrinsic_and_robot_feedback(self):
        import prepare_dataset_v2 as prep
        import numpy as np
        import torch
        extrinsic = torch.eye(4, dtype=torch.float64)
        extrinsic[:3, :3] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        extrinsic[:3, 3] = torch.tensor([0.4, -0.1, 0.8])
        poses = [torch.eye(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64)]
        poses[1][:3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        poses[1][:3, 3] = torch.tensor([0.2, 0.3, -0.4])
        world = torch.tensor([[1., 2., 3.], [-0.3, 0.1, 0.2]], dtype=torch.float64)
        mapper = prep.DynaMapping(10, history_frames=1, device='cpu')
        for index, pose in enumerate(poses):
            sensor_pose = pose @ extrinsic
            sensor_points = prep.transform_points(prep.inverse(sensor_pose), world)
            raw = prep.PointCloudFrame(sensor_points, pose, index * 100000000,
                                       sensor_pose=sensor_pose, device='cpu')
            frame = prep.robot_frame(raw)
            np.testing.assert_allclose(frame.points.numpy(), prep.transform_points(prep.inverse(pose), world).numpy(), atol=3e-7)
            alignment = mapper.update(frame)
        expected = np.linalg.inv(poses[1].numpy()) @ poses[0].numpy()
        np.testing.assert_allclose(alignment.transformation.numpy(), expected, atol=1e-12)
        np.testing.assert_allclose(alignment.aligned_point_cloud.numpy(), frame.points.numpy(), atol=3e-7)

    def test_storage_guard_prevents_write(self):
        import prepare_dataset_v2 as prep
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'frame.npz'
            with self.assertRaises(RuntimeError):
                prep.checked_write(path, b'12345', root, prep.RESERVE + 4)
            self.assertFalse(path.exists())
            prep.checked_write(path, b'12345', root, prep.RESERVE + 10000)
            self.assertEqual(path.read_bytes(), b'12345')


if __name__ == '__main__':
    unittest.main()
