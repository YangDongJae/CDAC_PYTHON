# ─────────────────────────────────────────────────────────────────────────────
# 메인 함수 (MATLAB processFrameNoresamp 대응)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import torch
from typing import Any
from config.schema import OCTConfig 
import torch, torch.nn.functional as F
from scipy.interpolate import interp1d
import math
from pathlib import Path
from typing import List, Literal
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
DEBUG = True
def dbg(msg:str):
    if DEBUG:
        print(msg)

def _as_used_indices(cfg: dict, dev: torch.device) -> torch.Tensor:
    """Info.usedSamples (1-based 인덱스/불리언 마스크/벡터 모두 대응) → 0-based long 텐서."""
    us = np.asarray(cfg["usedSamples"])
    if us.dtype == np.bool_:
        idx = np.nonzero(us)[0]
    else:
        idx = (us.astype(np.int64) - 1)  # 1→0 base
    return torch.as_tensor(idx.reshape(-1), dtype=torch.long, device=dev)
                                 
def _as_1d_array(x):
    a = np.asarray(x).reshape(-1)
    return a

def _ensure_center_vol_arrays(cfg, sx, sy):
    # centerVolX/Y가 없으면 계산해서 만듦 (d5/d6 Info 차이를 흡수)
    numVolLines  = int(cfg.get("numVolLines",  cfg["numImgLines"]))
    numVolFrames = int(cfg.get("numVolFrames", cfg["numImgFrames"]))
    numScanLines  = int(cfg["numScanLines"])
    numScanFrames = int(cfg["numScanFrames"])

    cX = _as_1d_array(cfg.get("centerVolX", cfg.get("centerX", [])))
    cY = _as_1d_array(cfg.get("centerVolY", cfg.get("centerY", [])))

    if cX.size != sx:
        cX = np.rint(np.linspace(numVolLines/2.0,  numScanLines  - numVolLines/2.0,  sx)).astype(np.int64)
    if cY.size != sy:
        cY = np.rint(np.linspace(numVolFrames/2.0, numScanFrames - numVolFrames/2.0, sy)).astype(np.int64)
    return cX, cY

def _motion_shift_at(cfg, frameShift, frame_1b):
    mls = _as_1d_array(cfg.get("motionLineShift", []))
    if mls.size == 0:
        return 0.0
    # MATLAB: motionLineShift(frameShift + frame)
    idx_1b = frameShift + int(frame_1b)                 # 1-based
    idx_1b = max(1, min(idx_1b, mls.size))              # clamp
    return float(mls[idx_1b - 1])                       # to python 0-based

def process_frame_noresamp(fid, frame_1b, vol_1b, cfg, device="cuda"):
    dev = torch.device(device)

    # ---- robust ints -------------------------------------------------
    vol_1b = int(vol_1b)
    frame_1b = int(frame_1b)
    subdiv = _as_1d_array(cfg["subdivFactors"])
    sx, sy = int(subdiv[0]), int(subdiv[1])

    # MATLAB ind2sub (column-major) → m,n (0-based)
    m = int((vol_1b - 1) % sx)
    n = int((vol_1b - 1) // sx)

    numVolLines  = int(cfg.get("numVolLines",  cfg["numImgLines"]))
    numVolFrames = int(cfg.get("numVolFrames", cfg["numImgFrames"]))
    centerVolX, centerVolY = _ensure_center_vol_arrays(cfg, sx, sy)
    print(centerVolX, centerVolY)

    # ---- frame/line shift --------------------------------------------
    frameShift = int(round(centerVolY[n] - numVolFrames / 2.0))
    motion_shift = _motion_shift_at(cfg, frameShift, frame_1b)
    lineShift_f = float(centerVolX[m]) - float(numVolLines) / 2.0 - motion_shift
    
    print(f"[a] centerX={int(centerVolX[m])}, motion_shift={motion_shift:.1f}, line_shift={int(round(lineShift_f))}")

    # ---- RAW frame 위치 계산 -----------------------------------------
    numScanLines  = int(cfg["numScanLines"])
    numSamples    = int(cfg["numSamples"])
    lineIdx = int(
        cfg["initLineShift"]
        + (frameShift + frame_1b - 1) * (numScanLines + int(cfg["numFlybackLines"]))
    )
    byte_offset = int(2 * (int(cfg["trigDelay"]) + numSamples * (lineIdx - 1)))
    print(f"[c] fseek offset = {byte_offset/1024:.1f} KiB (line_idx={lineIdx})")

    fid.seek(byte_offset, 0)
    raw_bytes = fid.read(2 * numSamples * numScanLines)
    fringes = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.float32, copy=False)
    fringes = fringes.reshape((numSamples, numScanLines), order="F")

    # ---- usedSamples: 0-based 정수 인덱스화 --------------------
    used_idx_np = _as_used_indices(cfg, dev=torch.device("cpu")).cpu().numpy()
    fringes  = fringes[used_idx_np, :]
    print(f"[c] fringes shape after read & usedSamples = {fringes.shape}")

    # ── Background Subtraction ───────────────────────────────────────
    if cfg.get("adaptiveBG", False):
        # ... (adaptiveBG 로직) ...
        fringesBS = np.zeros_like(fringes)
        bg_mean = np.asarray(cfg["bgMean"], dtype=np.float32)
        bg_centered = bg_mean - 32768.0
        denom = np.sum(bg_centered**2)
        for line in range(fringes.shape[1]):
            signal = fringes[:, line] - 32768.0
            a = np.sum(signal * bg_centered) / denom
            fringesBS[:, line] = signal / a - bg_mean + 32768.0
    else:
        # 동적 배경 제거
        frame_mean = fringes.mean(axis=1, keepdims=True)
        fringesBS = fringes - frame_mean

    if cfg.get("adaptiveBGOsc", False) and np.any(np.asarray(cfg.get("bgOsc", 0)) != 0):
        # ... (bgOsc 로직) ...
        bg_osc = np.asarray(cfg["bgOsc"], dtype=np.float32)
        osc_energy = np.sum(bg_osc**2)
        if osc_energy > 0:
            proj = np.sum(fringesBS * bg_osc[:, np.newaxis], axis=0) / osc_energy
            fringesBS -= proj * bg_osc[:, np.newaxis]

    # ── ✅ A-scan resample (SciPy interp1d 사용) ───────────────────
    numUsed = int(cfg["numUsedSamples"])
    
    # 1. 원본 x좌표 준비 및 [0, 1] 정규화 (NumPy로 수행)
    x_orig_raw = np.asarray(cfg["resampTraceA"])
    x_min, x_max = x_orig_raw.min(), x_orig_raw.max()
    x_orig_normalized = (x_orig_raw - x_min) / (x_max - x_min)

    # 2. 새로운 x좌표 준비 (NumPy로 수행)
    x_target = np.linspace(0, 1, numUsed)

    # 3. Scipy interp1d로 보간 수행 (MATLAB interp1과 가장 유사)
    # axis=0: 각 A-line(열)을 따라 독립적으로 보간
    interp_func = interp1d(
        x_orig_normalized,      # 원본 x
        fringesBS,              # 원본 y (K, W)
        kind='cubic',           # 'spline'과 동일
        axis=0,                 # K축을 따라 보간
        bounds_error=False,     # 범위 밖의 값은 에러 대신 fill_value 사용
        fill_value=0.0          # 범위 밖은 0으로 채움
    )
    fringesBS_interp_np = interp_func(x_target)

    # 4. 이후 PyTorch 연산을 위해 다시 텐서로 변환
    fringesBS_interp = torch.as_tensor(fringesBS_interp_np, dtype=torch.float32, device=dev)
    print(f"[e] A-scan resample done (SciPy) → fringesBS_interp shape: {fringesBS_interp.shape}")

    # ─── (g) Dispersion → IFFT → noise normalisation ---------------------
    disp_full = torch.as_tensor(cfg["dispComp"], dtype=torch.complex64, device=dev)
    if disp_full.ndim == 2: # 행렬일 경우 대각선 성분 추출
        disp_full = torch.diag(disp_full)
    used_indices_torch = _as_used_indices(cfg, dev=dev)
    disp = disp_full[used_indices_torch].view(-1, 1)

    fringes_calib = fringesBS_interp.to(torch.complex64) * disp

    img   = torch.fft.ifft(fringes_calib, dim=0)
    noise = torch.as_tensor(cfg["noiseProfile"], dtype=torch.float32, device=dev).view(-1, 1)
    img_calib = img / (noise + 1e-9)
    print(f"[f] after IFFT img_calib shape: {img_calib.shape}")
    
    start_idx = int(math.ceil(numUsed / 2.0))
    img_calib[start_idx:, :] = 0

    # ── FFT → FDFlip 패딩 ────────────────────────────────────────────
    fringes_calib_fft = torch.fft.fft(img_calib, dim=0)
    pad_len = int(cfg["numFTSamples"]) - numUsed
    pad = torch.zeros((pad_len, fringes_calib_fft.shape[1]), dtype=fringes_calib_fft.dtype, device=dev)

    if cfg.get("FDFlip", False):
        fringes_result = torch.cat([fringes_calib_fft, pad], dim=0)
    else:
        fringes_result = torch.cat([pad, fringes_calib_fft], dim=0)

    return fringes_result.to(torch.complex64), img_calib.to(torch.complex64)