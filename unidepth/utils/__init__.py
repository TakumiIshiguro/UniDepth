from .camera import invert_pinhole
# from .validation import validate
from .coordinate import coords_grid, normalize_coords
from .distributed import (barrier, get_dist_info, get_rank, is_main_process,
                          setup_multi_processes, setup_slurm,
                          sync_tensor_across_gpus)
from .geometric import spherical_zbuffer_to_euclidean, unproject_points
from .misc import (format_seconds, get_params, identity, recursive_index,
                   remove_padding, to_cpu)

_EVALUATION_EXPORTS = {"DICT_METRICS", "DICT_METRICS_3D", "eval_3d", "eval_depth"}
_VISUALIZATION_EXPORTS = {"colorize", "image_grid", "log_train_artifacts"}


def __getattr__(name):
    if name in _EVALUATION_EXPORTS:
        from . import evaluation_depth

        value = getattr(evaluation_depth, name)
        globals()[name] = value
        return value
    if name in _VISUALIZATION_EXPORTS:
        from . import visualization

        value = getattr(visualization, name)
        globals()[name] = value
        return value
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
