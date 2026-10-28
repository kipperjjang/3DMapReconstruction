import unittest

import torch
import MinkowskiEngine as ME

from neural_scene.blocks import ConvBlock, SparseBatchNorm, SpatialDownsample, SpatialUpsample, union_concat


class SparseBlockTests(unittest.TestCase):
    def test_singleton_batchnorm_uses_running_statistics_and_keeps_gradients(self):
        norm = SparseBatchNorm(2).train()
        with torch.no_grad():
            norm.bn.running_mean.copy_(torch.tensor([1., 2.]))
            norm.bn.running_var.copy_(torch.tensor([4., 9.]))
        features = torch.tensor([[3., 5.]], requires_grad=True)
        source = ME.SparseTensor(features, coordinates=torch.zeros(1, 5, dtype=torch.int32))
        tracked = norm.bn.num_batches_tracked.clone()
        norm.bn.momentum = None  # cumulative-statistics calibration also supports a singleton
        output = norm(source)
        expected = (features - norm.bn.running_mean) / torch.sqrt(norm.bn.running_var + norm.bn.eps)
        torch.testing.assert_close(output.F, expected)
        self.assertTrue(torch.equal(tracked, norm.bn.num_batches_tracked))
        output.F.sum().backward()
        torch.testing.assert_close(features.grad, 1 / torch.sqrt(norm.bn.running_var + norm.bn.eps).unsqueeze(0))
        self.assertIsNotNone(norm.bn.weight.grad)

    def test_union_skip_preserves_generated_sites_and_feature_gradients(self):
        decoder_features = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        encoder_features = torch.tensor([[5.], [6.]], requires_grad=True)
        decoder = ME.SparseTensor(decoder_features, coordinates=torch.tensor([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.int32))
        encoder = ME.SparseTensor(encoder_features, coordinates=torch.tensor([[0, 1, 0, 0, 0], [0, 2, 0, 0, 0]], dtype=torch.int32), coordinate_manager=decoder.coordinate_manager)
        merged = union_concat(decoder, encoder)
        values = {tuple(c): f.tolist() for c, f in zip(merged.C.tolist(), merged.F)}
        self.assertEqual(values[(0, 0, 0, 0, 0)], [1, 2, 0])
        self.assertEqual(values[(0, 1, 0, 0, 0)], [3, 4, 5])
        self.assertEqual(values[(0, 2, 0, 0, 0)], [0, 0, 6])
        merged.F.sum().backward()
        self.assertTrue(torch.equal(decoder_features.grad, torch.ones_like(decoder_features)))
        self.assertTrue(torch.equal(encoder_features.grad, torch.ones_like(encoder_features)))

    def test_generative_upsample_keeps_temporal_resolution_and_adds_sites(self):
        coords = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=torch.int32)
        source = ME.SparseTensor(torch.ones(2, 2), coordinates=coords)
        down = SpatialDownsample(2, 4).eval()(source)
        up = SpatialUpsample(4, 2).eval()(down)
        self.assertEqual(down.tensor_stride, [2, 2, 2, 1])
        self.assertEqual(up.tensor_stride, [1, 1, 1, 1])
        self.assertEqual(set(up.C[:, -1].tolist()), {0, 1})
        self.assertEqual(len(up), 16)
        self.assertGreater(len(up), len(source))
        output = ConvBlock(2, 2).eval()(up)
        output.F.sum().backward()


if __name__ == "__main__":
    unittest.main()
