# core/io/loaders.py
from pathlib import Path
import numpy as np
import torch
import pickle
from scipy.io import loadmat

# 기존 config.utils.load_mat_info가 있다고 가정
from config.utils import load_mat_info as load_oct_info

def load_fringes(path: Path) -> torch.Tensor:
    """프린지 데이터를 .pt 파일에서 로드합니다."""
    fringes = torch.load(path, map_location="cpu")
    if not torch.is_complex(fringes):
        raise TypeError(f"{path} must contain a complex tensor.")
    return fringes

def load_phase_data(source) -> torch.Tensor:
    """다양한 소스(경로, 배열, 텐서)에서 누적 위상 데이터를 로드합니다."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"cumPhaseY path not found: {p}")
        if p.suffix == ".pt":
            v = torch.load(p, map_location="cpu")
            return v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32)
        elif p.suffix == ".mat":
            md = loadmat(p)
            key = "cumPhaseY" if "cumPhaseY" in md else list(md.keys())[-1]
            arr = np.array(md[key]).squeeze()
            return torch.as_tensor(arr, dtype=torch.float32)
        else:  # .npy / .npz
            arr = np.load(p, allow_pickle=True)
            if isinstance(arr, np.lib.npyio.NpzFile):
                key = "cumPhaseY" if "cumPhaseY" in arr.files else arr.files[0]
                arr = arr[key]
            return torch.as_tensor(np.array(arr).squeeze(), dtype=torch.float32)

    if isinstance(source, np.ndarray):
        return torch.as_tensor(source.astype(np.float32))
    if isinstance(source, (list, tuple)):
        return torch.as_tensor(np.array(source, dtype=np.float32))
    if torch.is_tensor(source):
        return source.to(dtype=torch.float32)
    raise TypeError(f"Unsupported cumPhaseY type: {type(source)}")

def load_layers(path: Path) -> dict[str, np.ndarray]:
    """torch 또는 numpy로 저장된 레이어 파일을 로드하여 numpy dict로 반환합니다."""
    # 1) torch.load 시도
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            # 모든 값을 numpy로 변환
            return {k: v.cpu().numpy() if torch.is_tensor(v) else np.array(v) for k, v in obj.items()}
    except (pickle.UnpicklingError, AttributeError, RuntimeError):
        pass
    # 2) numpy 로드 fallback
    try:
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.lib.npyio.NpzFile):
            return {k: obj[k] for k in obj.files}
        if hasattr(obj, "item") and isinstance(obj.item(), dict):
            return obj.item()
        return obj # dict가 아닌 경우 그대로 반환
    except Exception as e:
        raise RuntimeError(f"Cannot load layers from {path}: {e}")