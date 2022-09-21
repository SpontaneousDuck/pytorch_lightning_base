from functools import partial
import operator
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.types import _PATH
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_info, rank_zero_warn
import os

def __last_ckpt_in_folder(dir_path: _PATH, name_key: str = ".ckpt"):
        # check directory existence
        fs = get_filesystem(dir_path)
        if not fs.exists(dir_path):
            return None

        # check corresponding file existence
        files = [os.path.basename(f["name"]) for f in fs.listdir(dir_path)]
        files = [os.path.join(dir_path, x) for x in files if name_key in x]
        if len(files) == 0:
            return None
            
        candidates_fs = {path: get_filesystem(path) for path in files if path}
        candidates_ts = {path: fs.modified(path) for path, fs in candidates_fs.items() if fs.exists(path)}
        if not candidates_ts:
            return None
        return max(candidates_ts.keys(), key=partial(operator.getitem, candidates_ts))

def test():
    dir_path_hpc = "artifacts/1"
    fs = get_filesystem(dir_path_hpc)
    if not fs.isdir(dir_path_hpc):
        return
    dir_path_hpc = str(dir_path_hpc)
    x = __last_ckpt_in_folder(dir_path_hpc, ".ckpt")
    print(x)

test()