# core/phase_stabilize/reverter.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import torch
except ImportError:
    torch = None

# 기존 함수가 이미 있다면 여기서 임포트해서 감싸주세요.
# from core.phase_stabilize import revert_fringes as _revert_fringes_fn

def _ensure_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for phase reversion")

def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

@dataclass
class PhaseReversionConfig:
    rad_per_pixel: float  # Info["radPerPixel"]
    num_img_lines: int    # Info["numImgLines"]
    num_img_frames: int   # Info["numImgFrames"]
    num_img_pixels: int   # Info["numImgPixels"]

class PhaseReversionService:
    """
    - 위상 되돌리기(revert)
    - wrapToPi 기반 보정 흐름
    - 레이어 z-쉬프트 보정
    """
    def __init__(self, cfg: PhaseReversionConfig):
        _ensure_torch()
        self.cfg = cfg

    def revert_by_phase(self, fringes: "torch.Tensor", phase_per_frame: "torch.Tensor") -> "torch.Tensor":
        """
        기존 프로젝트의 revert_fringes(…, phase) 함수를 그대로 감싸는 자리.
        만약 기존 함수가 없다면, 아래와 같이 broadcast 곱을 사용하세요.
        """
        # return _revert_fringes_fn(fringes, phase_per_frame)

        # [fallback 예시]: (Nz,Nx,Ny)에서 프레임 차원(Ny)에만 global 위상 곱
        phase = phase_per_frame.to(fringes.dtype)  # (F,)
        # (1,1,F)로 브로드캐스트
        ph = torch.exp(-1j * phase).view(1, 1, -1)
        return fringes * ph

    def wrap_then_revert(self, fringes: "torch.Tensor", cum_phase_1d: "torch.Tensor") -> tuple["torch.Tensor","torch.Tensor"]:
        """
        stabilizePhaseWrap 과 비슷한 흐름:
        - doppPhaseY = wrapToPi(diff(cumPhase))
        - cumPhaseYWrap = cumsum(doppPhaseY)
        - phase_diff = cumPhase - cumPhaseWrap
        - revert(phase_diff)
        """
        pi = np.pi
        dopp = torch.remainder(torch.diff(cum_phase_1d) + pi, 2 * pi) - pi
        cum_wrap = torch.cat([torch.zeros(1, dtype=dopp.dtype), torch.cumsum(dopp, dim=0)])
        phase_diff = cum_phase_1d - cum_wrap
        fringes2 = self.revert_by_phase(fringes, phase_diff)
        return fringes2, phase_diff

    def shift_layers(self, layers: dict[str, np.ndarray | "torch.Tensor"], phase_per_frame: "torch.Tensor") -> dict[str, np.ndarray]:
        """
        레이어들을 픽셀 단위로 y-방향 쉬프트.
        """
        px_shift_y = (phase_per_frame / self.cfg.rad_per_pixel)  # (F,)
        px = px_shift_y.view(1, -1)  # (1,F) → (W,F)로 broadcast

        out = {}
        for name in ("ILM", "NFL", "ISOS", "RPE"):
            arr = layers[name]
            if torch.is_tensor(arr):
                arr_np = arr.detach().cpu().numpy()
            else:
                arr_np = np.asarray(arr)
            out[name] = arr_np + px.numpy().repeat(self.cfg.num_img_lines, axis=0)
        return out