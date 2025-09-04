# /core/phase_stabilize/revert_and_save.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile as tiff

CACHE_DIR = Path("/home/work/OCT_DL/CDAC_OCT/CDAC_PYTHON/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """MATLAB wrapToPi와 동일: [-pi, pi]로 wrap."""
    return np.angle(np.exp(1j * x))

def revert_fringes(
    fringes: torch.Tensor,          # (L, W, F) complex64/complex128
    phase_y,                        # (F,) np.ndarray or torch.Tensor
    info: dict | None = None,       # backward-compat; not used
) -> torch.Tensor:
    """
    MATLAB revertFringes와 동등 동작(가정):
    프레임 축(F) 방향으로 exp(-j*phase)를 곱해 위상 보정.
    """
    if fringes.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("fringes must be complex tensor of complex64/complex128")

    device = fringes.device

    # np.ndarray or torch.Tensor 모두 허용
    if isinstance(phase_y, torch.Tensor):
        phase = phase_y.to(device=device, dtype=torch.float32)
    else:
        phase = torch.as_tensor(phase_y, device=device, dtype=torch.float32)

    # shape: (F,) -> (1,1,F)
    phase = phase.view(1, 1, -1)
    corr = torch.exp(-1j * phase)            # (1,1,F), complex64 by promotion
    return fringes * corr

def process_and_save_original(
    fringes: torch.Tensor,          # (L, W, F) complex
    cumPhaseY: np.ndarray,          # (F,) 누적 위상(rad)
    Info: dict,
    ILM: np.ndarray, NFL: np.ndarray,
    ISOS: np.ndarray, RPE: np.ndarray,
    vol: int,
    dFileName: str,
    *,
    plot_every: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    """
    MATLAB 코드의 동작을 그대로 재현.
    - 위상 복원(두 모드 중 하나)
    - ifft → 깊이축 crop → intensity(dB) 변환
    - TIFF 멀티페이지 저장
    - ILM/NFL/ISOS/RPE 좌표 Y-shift 반영
    반환: (ILM, NFL, ISOS, RPE, saved_path)
    """

    # --- 0) 입력/형상 체크 -----------------------------------------------------
    assert fringes.ndim == 3, "fringes must be (L,W,F)"
    L, W, F = fringes.shape
    assert cumPhaseY.shape == (F,), "cumPhaseY must be shape (F,)"

    # --- 1) 위상 복원 ----------------------------------------------------------
    if Info.get("stabilizePhaseRevert", False):
        # 직접 누적 위상 제거
        fr_corr = revert_fringes(fringes, cumPhaseY)
        img = torch.fft.ifft(fr_corr, dim=0)            # 깊이축(ifft)
        # 레이어 좌표 이동량 (픽셀)
        pixelShiftY = (cumPhaseY / float(Info["radPerPixel"]))  # (F,)
    elif Info.get("stabilizePhaseWrap", False):
        # wrap 방식: diff→wrap→누적합
        doppPhaseY = wrap_to_pi(np.diff(cumPhaseY))     # (F-1,)
        cumPhaseYWrap = np.cumsum(np.concatenate([[0.0], doppPhaseY]))  # (F,)
        delta = cumPhaseY - cumPhaseYWrap
        fr_corr = revert_fringes(fringes, delta)
        img = torch.fft.ifft(fr_corr, dim=0)
        pixelShiftY = (delta / float(Info["radPerPixel"]))        # (F,)
    else:
        # 복원 사용 안 함
        img = torch.fft.ifft(fringes, dim=0)
        pixelShiftY = np.zeros(F, dtype=np.float32)

    # --- 2) 깊이축 crop: img(1:Info.numImgPixels,:,:) --------------------------
    numImgPixels = int(Info["numImgPixels"])
    img = img[:numImgPixels, :, :]                      # (Lcrop, W, F)

    # --- 3) 레이어 좌표 보정 ---------------------------------------------------
    numImgLines = int(Info["numImgLines"])              # MATLAB 기준: Y축 길이
    # pixelShiftY를 (numImgLines, F)로 broadcast
    shift_map = np.tile(pixelShiftY.reshape(1, F), (numImgLines, 1))

    # 레이어 배열 크기는 (numImgLines, F)라고 가정 (MATLAB 코드와 동일)
    ILM = ILM + shift_map
    NFL = NFL + shift_map
    ISOS = ISOS + shift_map
    RPE = RPE + shift_map

    # --- 4) 저장 준비 ----------------------------------------------------------
    saveOriginal = bool(Info.get("saveOriginal", True))
    segmentOnly = bool(Info.get("segmentOnly", False))
    if not (saveOriginal and not segmentOnly):
        # 저장이 꺼져 있으면 아무것도 저장하지 않고 끝
        return ILM, NFL, ISOS, RPE, Path()

    dBRange = float(Info["dBRange"])
    noiseFloor = float(Info["noiseFloor"])
    numImgFrames = int(Info["numImgFrames"])
    numVolumes = int(Info.get("numVolumes", 1))

    # 출력 파일 경로 (멀티페이지 TIF)
    base = f"{dFileName}_Orig_{vol:02d}.tif"
    out_path = CACHE_DIR / base
    if out_path.exists():
        out_path.unlink()  # MATLAB처럼 기존 파일 삭제

    # --- 5) 프레임 루프: intensity(dB) → uint16 변환 → 멀티페이지 저장 ----------
    # img: complex → intensity = |img|^2
    with tiff.TiffWriter(str(out_path), bigtiff=True) as tfile:
        for frame in range(numImgFrames):  # 0-based
            if (frame % plot_every) == 0:
                # 진행상태 표기 (콘솔)
                print(f"OCT original saving vol {vol}/{numVolumes}, frame {frame+1}/{numImgFrames}...")

            # (Lcrop, W) complex → intensity
            img_f = img[:, :, frame]
            intensity = torch.abs(img_f) ** 2

            # dB 변환: 10*log10(I) - noiseFloor
            eps = 1e-12
            int_db = 10.0 * torch.log10(torch.clamp(intensity, min=eps)) - noiseFloor

            # [0, dBRange] → [0, 65535] 스케일
            int_db_np = int_db.detach().cpu().numpy()
            int_db_np = np.clip(int_db_np, 0.0, dBRange)
            int_u16 = (int_db_np / dBRange * (2**16 - 1)).astype(np.uint16)

            # TIFF append (멀티페이지)
            tfile.write(int_u16, contiguous=True)

            # 10프레임마다 플롯(로그 스케일 이미지)
            if (frame % plot_every) == 0:
                plt.figure(2, figsize=(8, 4))
                plt.clf()
                plt.imshow(int_db_np, cmap="gray", vmin=0, vmax=dBRange, origin="lower", aspect="auto")
                plt.title(f"OCT Original frame {frame+1}")
                plt.xlabel("A-line (W)")
                plt.ylabel("Depth (pixels)")
                plt.pause(0.01)

    print(f"Saved: {out_path}")
    return ILM, NFL, ISOS, RPE, out_path

