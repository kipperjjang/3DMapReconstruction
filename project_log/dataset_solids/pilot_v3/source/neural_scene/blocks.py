"""4D sparse convolution blocks and coordinate-union skip concatenation."""

import torch
from torch import nn
import MinkowskiEngine as ME


class SparseBatchNorm(ME.MinkowskiBatchNorm):
    """Use accumulated statistics when sparse support has only one row.

    A singleton has no batch variance. Keep affine/feature gradients and the
    existing running statistics instead of injecting points or changing pruning.
    """

    def forward(self, tensor):
        if self.training and len(tensor) == 1:
            features = torch.nn.functional.batch_norm(tensor.F,
                self.bn.running_mean, self.bn.running_var, self.bn.weight,
                self.bn.bias, training=False, momentum=0.0,
                eps=self.bn.eps)
            return with_features(tensor, features)
        return super().forward(tensor)


class ConvBlock(nn.Sequential):
    """Two ordinary 3x3x3x3 convolutions, each followed by BatchNorm/ReLU."""

    def __init__(self, in_channels, out_channels):
        layers = []
        for source in (in_channels, out_channels):
            layers.extend([
                ME.MinkowskiConvolution(source, out_channels, kernel_size=[3, 3, 3, 3], bias=False, dimension=4),
                SparseBatchNorm(out_channels),
                ME.MinkowskiReLU(inplace=True),
            ])
        super().__init__(*layers)


class SpatialDownsample(nn.Sequential):
    """Spatial stride two with unchanged temporal coordinates."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=[2, 2, 2, 1], stride=[2, 2, 2, 1], bias=False, dimension=4),
            SparseBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )


class SpatialUpsample(nn.Sequential):
    """Generate new spatial coordinates rather than reusing input support."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels, out_channels, kernel_size=[2, 2, 2, 1], stride=[2, 2, 2, 1], bias=False, dimension=4),
            SparseBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )


def with_features(tensor, features):
    return ME.SparseTensor(features=features, coordinate_map_key=tensor.coordinate_map_key, coordinate_manager=tensor.coordinate_manager)


def union_concat(decoder, encoder):
    """Concatenate features on union support with zeros for missing entries.

    ME.cat requires identical coordinate maps and cannot directly join a
    generative decoder with an encoder. Disjoint channel padding lets ME's
    differentiable coordinate union sum implement a genuine concatenation.
    """
    if decoder.tensor_stride != encoder.tensor_stride:
        raise ValueError("Skip tensors must have identical tensor strides")
    padded_decoder = with_features(decoder, torch.cat((decoder.F, decoder.F.new_zeros((len(decoder), encoder.F.shape[1]))), dim=1))
    padded_encoder = with_features(encoder, torch.cat((encoder.F.new_zeros((len(encoder), decoder.F.shape[1])), encoder.F), dim=1))
    return ME.MinkowskiUnion()(padded_decoder, padded_encoder)
