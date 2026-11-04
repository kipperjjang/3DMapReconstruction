import unittest
from unittest.mock import patch

import torch

from neural_scene.inputs import MERGED_POLICY, TEMPORAL_POLICY, merged_input
from neural_scene.loss import reconstruction_loss
from neural_scene.model import EmptyOffsets, ReconstructionUNet4D, UNetConfig
from neural_scene.voxel import VoxelGrid


class ModelTests(unittest.TestCase):
    def test_temporal_collapse_uses_both_sources_without_duplicate_outputs(self):
        model = self.model(alpha=0).eval()
        # Positive weights/activations give a reliable gradient from both times.
        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, 'kernel'):
                    module.kernel.fill_(0.02)
            for norm in model.modules():
                if isinstance(norm, torch.nn.BatchNorm1d):
                    norm.bias.fill_(0.1)
        current = torch.tensor([[.011, .022, .033]], requires_grad=True)
        previous = torch.tensor([[.031, .012, .023]], requires_grad=True)
        grid = VoxelGrid(shape=(16, 16, 16), origin=(0, 0, 0))
        packed = merged_input([current], [previous], grid)
        result = model(packed)
        self.assertEqual(set(packed.coordinates[:, -1].tolist()), {0, 1})
        for stage in result.stages:
            self.assertEqual(set(stage.logits.C[:, -1].tolist()), {0, 1})
        self.assertEqual(result.offsets.C[:, -1].unique().tolist(), [0])
        self.assertEqual(len(result.offsets), 4096)
        self.assertEqual(len(torch.unique(result.offsets.C[:, :4], dim=0)), 4096)
        self.assertEqual(result.confidence.shape, (4096,))
        result.offsets.F.sum().backward()
        for source in (current, previous):
            self.assertTrue(torch.isfinite(source.grad).all())
            self.assertGreater(float(source.grad.abs().sum()), 0)
        self.assertEqual(model.offset_head.kernel.shape[0], 2)
        self.assertTrue((model.offset_head.kernel.grad.abs().sum(dim=(1, 2)) > 0).all())

    def test_model_rejects_mismatched_input_policy(self):
        model = self.model().eval()
        with self.assertRaisesRegex(ValueError, 'policies differ'):
            model(merged_input([torch.zeros(1, 3)], input_policy=MERGED_POLICY))

    def test_empty_output_bypasses_sparse_backend_on_available_devices(self):
        devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = self.model(alpha=0.5).to(device).eval()
                packed = merged_input([torch.empty((0, 3), device=device)])
                # An empty prediction must not register a sparse coordinate map.
                with patch('neural_scene.model.ME.SparseTensor', side_effect=AssertionError('Empty sparse backend call')):
                    result = model(packed)
                self.assertIsInstance(result.offsets, EmptyOffsets)
                self.assertEqual(result.offsets.F.device, packed.features.device)
                self.assertEqual(result.offsets.C.shape, (0, 5))
                self.assertEqual(result.offsets.tensor_stride, [1, 1, 1, 2])
                self.assertEqual(result.points()[0].shape, (0, 3))
                loss = reconstruction_loss(result, [torch.zeros((1, 3), device=device)])
                self.assertEqual(loss.total.item(), 0)
                loss.total.backward()
                self.assertIsNotNone(model.offset_head.kernel.grad)
                # Reproduce evaluation after every candidate is pruned.
                model.zero_grad()
                with torch.no_grad():
                    model.occupancy_heads[0].kernel.zero_()
                    model.occupancy_heads[0].bias.fill_(-20)
                target = [torch.zeros((1, 3), device=device)]
                result = model(merged_input(target))
                self.assertIsInstance(result.offsets, EmptyOffsets)
                from neural_scene.experiment import counts_and_errors
                metrics = counts_and_errors(result.offsets.C, result.offsets.F, target, result.grid)
                self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (0, 0, 1))
                reconstruction_loss(result, target).total.backward()
                self.assertTrue(torch.isfinite(model.occupancy_heads[0].kernel.grad).all())
                if device.startswith('cuda'):
                    torch.cuda.synchronize()

    def test_single_point_training_through_all_downsample_stages(self):
        model = self.model().train()
        grid = VoxelGrid(shape=(16, 16, 16), origin=(0, 0, 0))
        targets = [torch.tensor([[0.011, 0.022, 0.033]])]
        result = model(merged_input(targets, grid=grid))
        loss = reconstruction_loss(result, targets)
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        self.assertIsNotNone(model.encoders[0][0].kernel.grad)
        self.assertTrue(torch.isfinite(model.encoders[0][0].kernel.grad).all())
        self.assertEqual(loss.matched_voxels, 1)

    def model(self, alpha=0):
        torch.manual_seed(11)
        return ReconstructionUNet4D(UNetConfig(channels=(2, 4, 4, 4, 4), pruning_threshold=alpha))

    def test_four_scales_t_zero_and_loss_backward(self):
        model = self.model().train()
        # Two batch members keep BatchNorm training well-defined at the bottleneck.
        grid = VoxelGrid(shape=(16, 16, 16), origin=(0, 0, 0))
        targets = [torch.tensor([[0.011, 0.022, 0.033]]), torch.tensor([[0.071, 0.082, 0.093]])]
        result = model(merged_input(targets, grid=grid))
        self.assertEqual([s.logits.tensor_stride for s in result.stages], [[8, 8, 8, 1], [4, 4, 4, 1], [2, 2, 2, 1], [1, 1, 1, 1]])
        self.assertEqual(result.offsets.tensor_stride, [1, 1, 1, 2])
        self.assertEqual(result.offsets.C[:, -1].unique().tolist(), [0])
        self.assertTrue(((result.offsets.F > 0) & (result.offsets.F < 1)).all())
        self.assertEqual([len(p) for p in result.points()], [4096, 4096])
        loss = reconstruction_loss(result, targets)
        self.assertEqual(loss.matched_voxels, 2)
        loss.total.backward()
        for head in model.occupancy_heads:
            self.assertIsNotNone(head.kernel.grad)
            self.assertTrue(torch.isfinite(head.kernel.grad).all())
        self.assertIsNotNone(model.offset_head.kernel.grad)
        self.assertGreater(model.offset_head.kernel.grad.abs().sum(), 0)

    def test_predictive_pruning_can_remove_every_voxel_even_with_gt(self):
        model = self.model(alpha=0.5).eval()
        with torch.no_grad():
            model.occupancy_heads[0].kernel.zero_()
            model.occupancy_heads[0].bias.fill_(-20)
        targets = [torch.zeros(1, 3)]
        result = model(merged_input(targets))
        self.assertEqual(len(result.stages), 1)
        self.assertEqual(len(result.offsets), 0)
        self.assertEqual(result.points()[0].shape, (0, 3))
        loss = reconstruction_loss(result, targets)
        self.assertGreater(loss.occupancy.item(), 0)
        self.assertEqual(loss.matched_voxels, 0)
        loss.total.backward()

    def test_empty_batch_members_and_empty_input(self):
        model = self.model().eval()
        result = model(merged_input([torch.empty(0, 3), torch.empty(0, 3)]))
        self.assertEqual([p.shape for p in result.points()], [(0, 3), (0, 3)])
        self.assertEqual(len(result.stages), 0)
        loss = reconstruction_loss(result, [torch.empty(0, 3), torch.empty(0, 3)])
        self.assertEqual(loss.total.item(), 0)
        loss.total.backward()
        partial = model(merged_input([torch.empty(0, 3), torch.zeros(1, 3)]))
        self.assertEqual([len(p) for p in partial.points()], [0, 4096])


if __name__ == "__main__":
    unittest.main()
