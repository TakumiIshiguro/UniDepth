from .activation import GEGLU, SwiGLU
from .attention import AttentionBlock, AttentionDecoderBlock, AttentionLayer
from .convnext import CvnxtBlock
from .mlp import MLP
from .positional_encoding import PositionEmbeddingSine
from .upsample import (ConvUpsample, ConvUpsampleShuffle,
                       ConvUpsampleShuffleResidual, ResUpsampleBil)

__all__ = [
    "SwiGLU",
    "GEGLU",
    "CvnxtBlock",
    "AttentionBlock",
    "NystromBlock",
    "PositionEmbeddingSine",
    "ConvUpsample",
    "MLP",
    "ConvUpsampleShuffle",
    "AttentionDecoderBlock",
    "ConvUpsampleShuffleResidual",
]


def __getattr__(name):
    if name == "NystromBlock":
        from .nystrom_attention import NystromBlock

        globals()[name] = NystromBlock
        return NystromBlock
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
