"""
Export the actor of a trained adaptive-weight residual SAC brain (brain_final.pth of web3d_drones_edge_residual_adaptive/edge_v3_12_residual_adaptive_sac.py)
to a plain numpy file, so the ROS brain node needs no torch.

    python tools/export_residual_npz.py                                   -> edge_residual_adaptive/weights/residual_adaptive_actor.npz
    python tools/export_residual_npz.py some/brain_final.pth --out other.npz

The checkpoint is a torch zip file. This tool reads it WITHOUT torch (zipfile + a small unpickler that rebuilds the tensors as numpy arrays), so it runs on any
machine with numpy. Only the actor is needed to fly (the critics are for training). The file stores the three layers plus what the policy needs at inference:
the PID bias weights (pid_max / pid_min, the adaptive blend of the training: pid_scale(s) * pid_bias(s) is added to the actor's logits).
"""
import argparse
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRAIN = ROOT.parent / 'web3d_drones_edge_residual_adaptive' / 'results' / 'models' / 'sac_residual_adaptive' / 'brain_final.pth'
DEFAULT_OUT = ROOT / 'edge_residual_adaptive' / 'weights' / 'residual_adaptive_actor.npz'

_DTYPES = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16, 'LongStorage': np.int64, 'IntStorage': np.int32,
           'ShortStorage': np.int16, 'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_}


class _Anything:
    """Stands in for any class of the checkpoint we do not need (optimizer state, config objects ...): accepts whatever the pickle does to it."""
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): pass
    def __call__(self, *a, **k): return _Anything()


def load_checkpoint(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl = next(n for n in names if n.endswith('data.pkl'))
    prefix = pkl[:-len('data.pkl')]

    def rebuild(storage, offset, size, stride, *_):
        flat = storage[offset:]
        return np.lib.stride_tricks.as_strided(flat, shape=tuple(size), strides=tuple(s * flat.itemsize for s in stride)).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
                return rebuild
            if name == 'OrderedDict':
                return OrderedDict
            if module == 'torch' and name in _DTYPES:
                return name
            if module in ('builtins', 'collections', '_codecs', 'numpy', 'numpy.core.multiarray', 'numpy._core.multiarray'):
                return super().find_class(module, name)
            return _Anything

        def persistent_load(self, pid):
            _, storage_type, key, _loc, _numel = pid
            dtype = _DTYPES[storage_type if isinstance(storage_type, str) else getattr(storage_type, '__name__', 'FloatStorage')]
            return np.frombuffer(zf.read(f'{prefix}data/{key}'), dtype=dtype)

    return Unpickler(zf.open(pkl)).load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('brain', nargs='?', default=str(DEFAULT_BRAIN))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--pid-max', type=float, default=1.5, help='PID bias weight far from obstacles (the training default)')
    ap.add_argument('--pid-min', type=float, default=0.0, help='PID bias weight right next to an obstacle (the training default)')
    args = ap.parse_args()

    ck = load_checkpoint(args.brain)
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    layers = [(np.asarray(sd[f'actor.{i}.weight'], dtype=np.float32), np.asarray(sd[f'actor.{i}.bias'], dtype=np.float32)) for i in (0, 2, 4)]
    n_obs = layers[0][0].shape[1]
    meta = ck if isinstance(ck, dict) else {}
    np.savez(args.out, w1=layers[0][0], b1=layers[0][1], w2=layers[1][0], b2=layers[1][1], w3=layers[2][0], b3=layers[2][1],
             n_obs=n_obs, actions=layers[2][0].shape[0], pid_max=args.pid_max, pid_min=args.pid_min,
             stack=int(meta.get('stack', 1) or 1), shield=bool(meta.get('shield', False)), wind_comp=float(meta.get('wind_comp', 0.0) or 0.0))
    print(f'wrote {args.out}: inputs {n_obs}, hidden {layers[0][0].shape[0]} -> {layers[1][0].shape[0]}, actions {layers[2][0].shape[0]}, '
          f'pid weight {args.pid_max} (far) .. {args.pid_min} (next to an obstacle), stack {meta.get("stack", 1)}, shield {meta.get("shield", False)}')


if __name__ == '__main__':
    main()
