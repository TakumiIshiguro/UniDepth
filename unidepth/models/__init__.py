from .unidepthv2 import UniDepthV2

__all__ = [
    "UniDepthV1",
    "UniDepthV2old",
    "UniDepthV2",
]


def __getattr__(name):
    if name == "UniDepthV1":
        from .unidepthv1 import UniDepthV1

        globals()[name] = UniDepthV1
        return UniDepthV1
    if name == "UniDepthV2old":
        from .unidepthv2 import UniDepthV2old

        globals()[name] = UniDepthV2old
        return UniDepthV2old
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
