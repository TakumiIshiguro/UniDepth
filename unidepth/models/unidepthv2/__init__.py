from .unidepthv2 import UniDepthV2

__all__ = [
    "UniDepthV2",
    "UniDepthV2old",
]


def __getattr__(name):
    if name == "UniDepthV2old":
        from .unidepthv2_old import UniDepthV2old

        globals()[name] = UniDepthV2old
        return UniDepthV2old
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
