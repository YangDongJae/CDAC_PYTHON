# config/utils.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Union, Tuple
import numpy as np
import torch
import warnings


_HDF5_SIG = b"\x89HDF\r\n\x1a\n"

def _has_hdf5_signature(path: Union[str, Path]) -> bool:
    """파일 헤더 8바이트로 HDF5(v7.3) 여부를 안정적으로 판별."""
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
        return sig == _HDF5_SIG
    except Exception:
        return False

def _is_mat_v73(path: Union[str, Path]) -> bool:
    """
    MATLAB v7.3(HDF5) 포맷인지 점검.
    1) HDF5 시그니처로 1차 판별
    2) 필요 시 h5py로 오픈 가능 여부로 보조 판별
    """
    path = str(path)
    if _has_hdf5_signature(path):
        return True
    # h5py가 있고, 열리면 v7.3로 간주
    try:
        import h5py  # lazy
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False

def _safe_decode_bytes(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray)):
        for enc in ("utf-8", "latin-1"):
            try:
                return x.decode(enc, errors="ignore")
            except Exception:
                pass
    return x

def _to_python(x):
    """torch.Tensor/np.generic/bytes/중첩구조를 pandas-friendly한 파이썬 타입으로 변환"""
    if 'torch' in globals() and torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        # 0-dim은 스칼라로, 그 외는 list로
        return x.item() if x.ndim == 0 else x.tolist()
    if isinstance(x, bytes):
        return x.decode('utf-8', errors='ignore')
    if isinstance(x, dict):
        return {str(k): _to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_python(v) for v in x]
    return x

def _load_pt_info(path: Path):
    if torch is None:
        raise ImportError("torch가 설치되어 있지 않습니다. `.pt` 파일을 읽으려면 torch가 필요합니다.")
    # 신뢰 가능한 파일만 열어주세요(내부적으로 pickle 사용)
    obj = torch.load(str(path), map_location='cpu',weights_only=False)
    info = obj.get('Info', obj) if isinstance(obj, dict) else obj
    return _to_python(info)

def _h5_to_py(obj) -> Any:
    """h5py Group/Dataset → 파이썬 타입으로 재귀 변환."""
    import h5py
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        data = _safe_decode_bytes(data)
        if isinstance(data, np.ndarray):
            if data.dtype.kind == "S":  # bytes 문자열 배열
                try:
                    data = data.astype(str)
                except Exception:
                    data = np.vectorize(_safe_decode_bytes, otypes=[object])(data)
            elif data.dtype == object:   # 셀 등
                return [ _safe_decode_bytes(el) for el in data.ravel() ]
            if data.shape == ():
                try:
                    return data.item()
                except Exception:
                    pass
        return data
    elif isinstance(obj, h5py.Group):
        return {k: _h5_to_py(obj[k]) for k in obj.keys()}
    return obj

def _load_mat_v73(path: Union[str, Path]) -> Dict[str, Any]:
    import h5py
    out: Dict[str, Any] = {}
    with h5py.File(str(path), "r") as f:
        key_info = next((k for k in f.keys() if k.lower() == "info"), None)
        if key_info is not None:
            out["Info"] = _h5_to_py(f[key_info])
            return out
        for k in f.keys():
            out[k] = _h5_to_py(f[k])
    return out


# ========================= v7.2 이하(scipy.io) 경로 =========================

def _scipy_to_py(x: Any) -> Any:
    """scipy.loadmat 결과를 파이썬 타입으로 재귀 변환."""
    if isinstance(x, np.ndarray):
        if x.shape == ():
            try:
                return _scipy_to_py(x.item())
            except Exception:
                return x
        if x.dtype.names:  # struct array
            items = []
            for i in range(x.shape[0]):
                d: Dict[str, Any] = {}
                for name in x.dtype.names:
                    d[name] = _scipy_to_py(x[name][i])
                items.append(d)
            return items
        if x.dtype == object:  # cell 등
            return [ _scipy_to_py(el) for el in x.ravel() ]
        return x
    if hasattr(x, "_fieldnames"):  # mat_struct
        return {name: _scipy_to_py(getattr(x, name)) for name in x._fieldnames}
    return _safe_decode_bytes(x)

def _load_mat_legacy(path: Union[str, Path]) -> Dict[str, Any]:
    """v7.2 이하 로더. (실패 시 NotImplementedError → v7.3 폴백)"""
    from scipy.io import loadmat
    try:
        data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        # 실제 v7.3인데 scipy 경로로 들어온 경우 → v7.3 폴백
        return _load_mat_v73(path)

    data = {k: v for k, v in data.items() if not k.startswith("__")}
    key_info = next((k for k in data.keys() if k.lower() == "info"), None)
    if key_info is not None:
        return {"Info": _scipy_to_py(data[key_info])}
    return {k: _scipy_to_py(v) for k, v in data.items()}


# ========================= 공개 API =========================

def compute_derived_fields(cfg: OCTConfig) -> dict:
    numVolumes = cfg.subdivFactors[0] * cfg.subdivFactors[1]
    centerX = torch.round(torch.linspace(
        cfg.numImgLines / 2,
        cfg.numResampLines - cfg.numImgLines / 2,
        cfg.subdivFactors[0]
    )).to(torch.int32)
    centerY = torch.round(torch.linspace(
        cfg.numImgFrames / 2,
        cfg.numScanFrames - cfg.numImgFrames / 2,
        cfg.subdivFactors[1]
    )).to(torch.int32)
    sigmaX = cfg.segmentSigma[0] / cfg.scanSize[0] * cfg.numResampLines
    sigmaY = cfg.segmentSigma[1] / cfg.scanSize[1] * cfg.numScanFrames
    sigmaZ = cfg.segmentSigma[2] / cfg.depthPerPixel
    return dict(numVolumes=numVolumes, centerX=centerX, centerY=centerY,
                sigmaX=sigmaX, sigmaY=sigmaY, sigmaZ=sigmaZ)


def load_mat_info(path: str, force_engine: str | None = None):
    """
    Info 구조를 담은 파일을 로드하여 dict로 반환.
    - .pt/.pth/.pkl : torch.load
    - .mat         : (v7.3 → h5py / legacy → scipy.io)
    기존과 동일한 시그니처 유지.
    """
    p = Path(path)
    ext = p.suffix.lower()

    # 사용자가 engine을 강제한 경우는 기존 로직 우선
    if force_engine:
        if force_engine in ("legacy", "mat"):
            out = _load_mat_legacy(p)
        elif force_engine in ("v73", "hdf5"):
            out = _load_mat_v73(p)
        else:
            warnings.warn(f"알 수 없는 force_engine={force_engine}, 확장자로 판별합니다.")
    else:
        if ext in {".pt", ".pth", ".pkl"}:
            out = _load_pt_info(p)
        elif ext == ".mat":
            out = _load_mat_v73(p) if _is_mat_v73(p) else _load_mat_legacy(p)
        else:
            # 확장자가 없거나 생소하면: pt 시도 → mat 시도 순
            try:
                out = _load_pt_info(p)
            except Exception:
                out = _load_mat_v73(p) if _is_mat_v73(p) else _load_mat_legacy(p)

    # 어떤 포맷이든 최종적으로 Info 키를 우선 반환(기존 관례 유지)
    if isinstance(out, dict) and "Info" in out and isinstance(out["Info"], dict):
        return out["Info"]
    return out


def load_motion_info(mat_path: Union[str, Path], *, force_engine: str | None = None) -> Dict[str, Any]:
    """
    모션 .mat 로드 (v7.3/legacy 자동 + 폴백). 기대 키:
    - motionLineShift, motionFrames, cumPhaseX, cumPhaseY
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Motion file not found: {mat_path}")

    if force_engine == "h5py":
        mat = _load_mat_v73(mat_path)
    elif force_engine == "scipy":
        mat = _load_mat_legacy(mat_path)
    else:
        mat = _load_mat_v73(mat_path) if _is_mat_v73(mat_path) else _load_mat_legacy(mat_path)

    def _get_any(d: Dict[str, Any], key: str, default: Any) -> Any:
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict) and key in v:
                return v[key]
        return default

    motionLineShift = np.squeeze(_get_any(mat, 'motionLineShift', np.zeros(1)))
    motionFrames    = np.squeeze(_get_any(mat, 'motionFrames',    np.zeros(1)))
    cumPhaseX       = np.squeeze(_get_any(mat, 'cumPhaseX',       np.array([0.0])))
    cumPhaseY       = np.squeeze(_get_any(mat, 'cumPhaseY',       np.array([0.0])))

    return dict(
        motionFile=mat_path,
        motionLineShift=motionLineShift,
        motionFrames=motionFrames,
        cumPhaseX=cumPhaseX,
        cumPhaseY=cumPhaseY
    )


def sanitize_info(info_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Info dict 타입 보정:
    - ndarray → Tensor
    - 경로 기본값 보완
    - 필수 서브 dict 보완
    """
    for k in ['validVols', 'partialVols', 'centerX', 'centerY']:
        v = info_dict.get(k, None)
        if v is not None and not isinstance(v, torch.Tensor):
            try:
                info_dict[k] = torch.tensor(v)
            except Exception:
                pass

    if 'root' not in info_dict or not isinstance(info_dict['root'], (str, Path)):
        info_dict['root'] = Path('.')
    if 'matDir' not in info_dict or not isinstance(info_dict['matDir'], (str, Path)):
        info_dict['matDir'] = Path('.')

    for k in ['CAO', 'ISAM', 'Save', 'Motion']:
        if k not in info_dict or not isinstance(info_dict[k], dict):
            info_dict[k] = {}

    return info_dict