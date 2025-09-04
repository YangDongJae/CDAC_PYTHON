# core/frame_process/utils.py
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

def process_frame(
    fid:         "io.BufferedReader",  # 열린 eye*.data 파일 핸들 (binary)
    frame_idx:   int,
    vol_idx:     int,
    cfg:         dict,
    motion:      dict,
    *,
    device:      str | torch.device = "cuda",
) -> torch.Tensor:
    """GPU 版 of MATLAB *processFrame.m*  — returns **one** fringe frame.

    Return shape ⇒  (**numFTSamples**,  **L_frame**)  complex64
    L_frame varies (638 – ~670 lines) exactly like MATLAB.
    """

    dev = torch.device(device)

    # ─── (a) sub-volume 행/열 index and shifts ──────────────────────────
        # ── (a) sub-volume index & shifts ─────────────────────────────────────────
    m = (vol_idx - 1) % cfg["subdivFactors"][0]
    n = (vol_idx - 1) // cfg["subdivFactors"][0]

    frame_shift  = int(round(cfg["centerY"][n] - cfg["numImgFrames"] / 2))
    motion_shift = motion["motionLineShift"][frame_shift + frame_idx + 1]   # 1-base → 0-base
    line_shift_f = (
        cfg["centerX"][m]
        - cfg["numImgLines"] // 2
        - motion_shift
    )
    line_shift   = int(round(line_shift_f))

    dbg(f"[a] centerX={cfg['centerX'][m]}, motion_shift={motion_shift}, "
        f"line_shift={line_shift}")

    # ── (b) RAW line range 선택 ──────────────────────────────────────────────
    trB = torch.as_tensor(cfg["resampTraceB"], dtype=torch.float64, device=dev)
    ln0, ln1 = line_shift, line_shift + cfg["numImgLines"] - 1
    eps = 1e-6

    if torch.all(trB[1:] > ln0 + 1):
        first_raw = 0
    elif torch.any(trB[1:] > ln0 + 1):
        first_raw = int(torch.nonzero(trB > ln0 + 1, as_tuple=True)[0][0]) - 1
    else:
        dbg("[b] early exit: no RAW covers ln0")
        return torch.zeros((cfg["numFTSamples"], cfg["numImgLines"]),
                           dtype=torch.complex64, device=dev)

    if torch.all(trB[:-1] < ln1 + eps):
        last_raw = cfg["numScanLines"] - 1
    elif torch.any(trB[:-1] < ln1 + eps):
        last_raw = int(torch.nonzero(trB < ln1 + eps, as_tuple=True)[0][-1]) + 1
    else:
        dbg("[b] early exit: no RAW covers ln1")
        return torch.zeros((cfg["numFTSamples"], cfg["numImgLines"]),
                           dtype=torch.complex64, device=dev)

    num_raw_lines = last_raw - first_raw + 1
    dbg(f"[b] ln0={ln0}, ln1={ln1}   → first_raw={first_raw}, last_raw={last_raw} "
        f"(num_raw_lines={num_raw_lines})")

    # jump(>1) 탐색
    jump_idx = torch.nonzero(torch.diff(trB[first_raw:last_raw+1]) > 1, as_tuple=True)[0]
    if jump_idx.numel():
        dbg(f"[b] traceB BIG jump inside window at raw idx (local): {jump_idx.tolist()}")

    # ── (c) RAW 파일에서 frame 읽기 ──────────────────────────────────────────
    line_idx = int(cfg["initLineShift"]
                   + (frame_shift + frame_idx - 1) * (cfg["numScanLines"] + cfg["numFlybackLines"])
                   + first_raw)
    offset = 2 * (cfg["trigDelay"] + cfg["numSamples"] * (line_idx - 1))
    dbg(f"[c] fseek offset = {offset/1024:.1f} KiB (line_idx={line_idx})")
    fid.seek(offset, 0)

    raw_bytes = fid.read(2 * cfg["numSamples"] * num_raw_lines)
    fringes = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.float64)
    fringes = fringes.reshape((cfg["numSamples"], num_raw_lines), order="F")
    fringes = fringes[np.array(cfg["usedSamples"]) - 1, :]
    dbg(f"[c] fringes shape after read & usedSamples = {fringes.shape}")
 
    # ─── (e) Background & BG-osc subtraction ----------------------------
    if cfg["adaptiveBG"]:
        # output buffer
        fringesBS = np.zeros_like(fringes)
        
        bg_mean = np.asarray(cfg["bgMean"], dtype=np.float64)  # shape (numUsedSamples,)
        bg_centered = bg_mean - 32768

        for line in range(fringes.shape[1]):  # numRawLines
            signal = fringes[:, line] - 32768
            a = np.sum(signal * bg_centered) / np.sum(bg_centered ** 2)
            fringesBS[:, line] = signal / a - bg_mean + 32768

    else:
        # background array를 repeat해서 뺌
        bg_array = np.tile(np.asarray(cfg["bgMean"], dtype=np.float64).reshape(-1, 1),
                        (1, fringes.shape[1]))  # shape = (numUsedSamples, numRawLines)
        fringesBS = fringes - bg_array

    if cfg.get("adaptiveBGOsc", False) and np.any(cfg["bgOsc"] != 0):
        bg_osc = np.asarray(cfg["bgOsc"], dtype=np.float64)  # shape = (numUsedSamples,)
        osc_energy = np.sum(bg_osc ** 2)

        if osc_energy == 0:
            print("Warning: bgOsc energy is zero, skipping oscillation removal.")
        else:
            for line in range(fringesBS.shape[1]):
                projection = np.sum(fringesBS[:, line] * bg_osc) / osc_energy
                fringesBS[:, line] -= projection * bg_osc

    # --- (f) A-scan별 spline 보간 ---------------------------------
    x_orig = torch.tensor(cfg["resampTraceA"], dtype=torch.float64, device=device)          # (K,)

    fringes_torch = torch.as_tensor(fringesBS, dtype=torch.float64, device=device)          # (K, W)

    # natural_cubic_spline_coeffs : (..., time, channels)  구조 요구
    #   원하는 구조 ==> (1,   K,        W)
    fringes_reshape = fringes_torch.unsqueeze(0)                                            # (1, K, W)

    coeffs = natural_cubic_spline_coeffs(x_orig, fringes_reshape)  # time=K 와 일치
    spline = NaturalCubicSpline(coeffs)

    T        = cfg["numUsedSamples"]                  # == K
    x_target = torch.linspace(0, 1, T, device=device) # (K,)

    y_target = spline.evaluate(x_target)              # (1, K, W)  ← batch=1 유지
    fringesBS_interp = y_target.squeeze(0)            # (K, W)     ← 최종 결과
    dbg(f"[e] A-scan resample done  → fringesBS_interp shape: {fringesBS_interp.shape}")

    # ─── (g) Dispersion → IFFT → noise normalisation ---------------------
    disp = torch.as_tensor(cfg["dispComp"],dtype=torch.complex64,device=device)
    fringes_calib = fringesBS_interp.to(torch.complex64) * disp[:, None]

    img   = torch.fft.ifft(fringes_calib.to(torch.complex64), dim=0)
    noise = torch.as_tensor(cfg["noiseProfile"], dtype=torch.float64, device=dev)
    noise = noise.view(-1, 1).repeat(1, fringes.shape[1])   #reshaping from 280 -> 280 646 to broadcast
    img_calib   = img / noise
    dbg(f"[f] after IFFT img_calib shape: {fringes_calib.shape}")


    # return img,img_calib


    img_calib[: round(cfg["bgBW"] * cfg['numUsedSamples'] / cfg["numFTSamples"])] = 0
    start_idx = int(np.ceil(cfg["numUsedSamples"] / 2.0)) + 1  # MATLAB's 1 + ceil()
    img_calib[start_idx:, :] = 0

    # print(f"Mean mag: {torch.mean(torch.abs(img_calib)).item():.4f}")
    # return img, img_calib

    # ─── (h) B-scan resample (if needed) --------------------------------
    if (num_raw_lines != cfg["numImgLines"] and
            np.any(np.diff(cfg["resampTraceB"][first_raw:last_raw]) != 1)):

        # ① magnitude / phase 분해  ──────────────────────────────
        mag   = torch.abs(img_calib)                    # (P, total_raw)
        phase = img_calib / (mag + 1e-12)
        phase[mag == 0] = 1

        # ② 보간용 x 좌표 준비  ------------------------------------
        t      = torch.as_tensor(                      # (K,)
                cfg["resampTraceB"][first_raw : last_raw + 1],
                dtype=torch.float64, device=device)
        x_tgt  = torch.arange(line_shift + 1,          # (N,)
                            line_shift + cfg["numImgLines"] + 1,
                            dtype=torch.float64, device=device)

        # ③ MATLAB  interp1(...,'linear',0) 100% 복제 ------------
        mag_interp  = interp1_linear_zero(t, mag,       x_tgt, device=device)        # (P,N)

        ph_real     = interp1_linear_zero(t, phase.real, x_tgt, device=device)
        ph_imag     = interp1_linear_zero(t, phase.imag, x_tgt, device=device)
        ph_interp   = torch.complex(ph_real, ph_imag)                                   # (P,N)

        # unit-vector 재정규화  (= MATLAB 두 줄)
        abs_val   = torch.abs(ph_interp)
        mask      = abs_val > 0
        ph_interp[mask] = ph_interp[mask] / abs_val[mask]

        # ④ 새로운 img_calib  + FFT + 패딩 ------------------------
        img_calib      = mag_interp.to(torch.complex64) * ph_interp.to(torch.complex64)
        fringes_calib  = torch.fft.fft(img_calib, dim=0)

        pad_len = cfg["numFTSamples"] - cfg["numUsedSamples"]
        pad     = torch.zeros((pad_len, fringes_calib.shape[1]),
                            dtype=fringes_calib.dtype, device=fringes_calib.device)

        fringes_result = (torch.cat([fringes_calib, pad], dim=0)
                        if cfg.get("FDFlip", False)
                        else torch.cat([pad, fringes_calib], dim=0))
    # ────────────────────────────────────────────────────────────────
    else:
        fringes_calib = torch.fft.fft(img_calib, dim=0)
        pad_len = cfg["numFTSamples"] - cfg["numUsedSamples"]
        pad     = torch.zeros((pad_len, fringes_calib.shape[1]),
                            dtype=fringes_calib.dtype, device=fringes_calib.device)
        fringes_result = (torch.cat([fringes_calib, pad], dim=0)
                        if cfg.get("FDFlip", False)
                        else torch.cat([pad, fringes_calib], dim=0))

    return fringes_result, img_calib

# ----------------------------------------------------------------------
# 2)  ─────── read_fringe_frame_bin_gpu  (얇은 래퍼) ───────
# ----------------------------------------------------------------------

def read_fringe_frame_bin_gpu(file_obj, frame_idx, vol_idx, cfg, motion,
                              *, device="cuda") -> torch.Tensor:
    """얇은 래퍼: 인덱스 계산 후 `process_frame` 호출."""
    return process_frame(file_obj, frame_idx, vol_idx, cfg, motion, device=device)

# ----------------------------------------------------------------------
# 3)  ─────── volume-level loader & stack util ───────
# ----------------------------------------------------------------------

def load_volume_variable(data_path: str | Path, vol: int, info: dict,
                         *, device: str | torch.device = "cuda") -> List[torch.Tensor]:
    if "Motion" not in info:
        info["Motion"] = {
            "motionLineShift": info["motionLineShift"],
            "motionFrames"   : info["motionFrames"],
            "cumPhaseX"      : info["cumPhaseX"],
            "cumPhaseY"      : info["cumPhaseY"],
            # 필요하면 enable·motionFile 등도 추가
        }               
    dev = torch.device(device)
    frames: List[torch.Tensor] = []
    with open(data_path, "rb") as fid:
        print(f"▶ reading volume {vol} …")
        for fr_idx in range(info["numImgFrames"]):
            if fr_idx % 20 == 0:
                print(f"   frame {fr_idx+1}/{info['numImgFrames']}", end="\r")
            fr = read_fringe_frame_bin_gpu(fid, fr_idx, vol, info, motion=info["Motion"], device=dev)
            frames.append(fr)
    print("\n✔ done. minL={}, maxL={}".format(
        min(t.shape[1] for t in frames), max(t.shape[1] for t in frames)))
    return frames

# ----------------------------------------------------------------------

def stack_frames(frames: List[torch.Tensor], *,
                 mode: Literal["pad", "crop500"] = "pad", crop_width: int = 500) -> torch.Tensor:
    K = frames[0].shape[0]
    N = len(frames)
    if mode == "pad":
        L_max = max(t.shape[1] for t in frames)
        out = frames[0].new_zeros(K, L_max, N)
        for i, fr in enumerate(frames):
            out[:, :fr.shape[1], i] = fr
        return out
    elif mode == "crop500":
        out = frames[0].new_empty(K, crop_width, N)
        for i, fr in enumerate(frames):
            L = fr.shape[1]
            if L < crop_width:
                pad = fr.new_zeros(K, crop_width - L)
                tmp = torch.cat([fr, pad], dim=1)
            else:
                s = (L - crop_width) // 2
                tmp = fr[:, s:s+crop_width]
            out[:, :, i] = tmp
        return out
    else:
        raise ValueError("mode must be 'pad' or 'crop500'")

def interp1_linear_zero(xp: torch.Tensor | np.ndarray,
                        fp: torch.Tensor | np.ndarray,
                        xq: torch.Tensor | np.ndarray,
                        device=None):
    """
    MATLAB  interp1( xp.', fp.', xq.', 'linear', 0 ).'
             └─── 1-D *선형* 보간 + extrapval = 0
    • xp : (K,) 1-D (float64)  –  원본 x 좌표
    • fp : (B, K) or (K,)      –  원본 y (B = batch, 첫 축)
    • xq : (N,) 1-D            –  보간 대상 좌표
    • 반환 : (B, N)  (fp 가 1-D 이면 (1,N) 뒤 torch.squeeze 가능)
    """
    # ---- 보간 좌표를 CPU NumPy 로 준비 ---------------------------------
    xp_np = np.asarray(xp.cpu() if torch.is_tensor(xp) else xp, dtype=np.float64)
    xq_np = np.asarray(xq.cpu() if torch.is_tensor(xq) else xq, dtype=np.float64)

    # ---- fp 를 2-D (B,K) 로 변환 ---------------------------------------
    if torch.is_tensor(fp):
        fp_np = fp.cpu().numpy()
    else:
        fp_np = np.asarray(fp)
    if fp_np.ndim == 1:
        fp_np = fp_np[None, :]              # (1,K)
    B, K = fp_np.shape
    if xp_np.size != K:
        raise ValueError(f"xp length {xp_np.size} != fp.shape[1] {K}")

    # ---- 배치 선형 보간 (extrap→0) --------------------------------------
    out_np = np.empty((B, xq_np.size), dtype=fp_np.dtype)
    for b in range(B):
        out_np[b] = np.interp(xq_np, xp_np, fp_np[b],
                              left=0.0, right=0.0)

    # ---- torch tensor 로 복귀 ------------------------------------------
    if device is None and torch.is_tensor(xp):
        device = xp.device
    elif device is None:
        device = 'cpu'
    return torch.from_numpy(out_np).to(device)

#---KAIST
# /core/frame_process/process_frame_noresamp.py
import io, math
from pathlib import Path
import numpy as np
import torch

# 선택: torchcubicspline 있으면 사용, 없으면 SciPy로 폴백
try:
    from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
    _HAS_TCS = True
except Exception:
    _HAS_TCS = False
    from scipy.interpolate import interp1d  # CPU fallback only

# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _as_1d(a) -> np.ndarray | None:
    if a is None: return None
    arr = np.asarray(a)
    return arr.reshape(-1)

def _linspace_round(a: float, b: float, n: int) -> np.ndarray:
    return np.rint(np.linspace(a, b, n)).astype(np.int64)

def _clamped_1b_index(arr_len: int, idx_1b: int) -> int:
    """1-based 인덱스를 0-based로 변환 + 경계 클램프."""
    idx_1b = int(idx_1b)
    if arr_len <= 0: return 0
    if idx_1b < 1: idx_1b = 1
    if idx_1b > arr_len: idx_1b = arr_len
    return idx_1b - 1

def _as_used_indices(cfg: dict, dev: torch.device) -> torch.Tensor:
    """Info.usedSamples (1-based 인덱스/불리언 마스크/벡터 모두 대응) → 0-based long 텐서."""
    us = np.asarray(cfg["usedSamples"])
    if us.dtype == np.bool_:
        idx = np.nonzero(us)[0]
    else:
        idx = (us.astype(np.int64) - 1)  # 1→0 base
    return torch.as_tensor(idx.reshape(-1), dtype=torch.long, device=dev)

def _prepare_disp_comp(cfg: dict, used_idx: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """
    Info.dispComp:
      - 1D 길이 == numUsed 또는 == numSamples → used 인덱스로 선택
      - 2D 정방 행렬 → diag 추출
      - 구조체(real/imag)로 온 경우는 사전에 복소로 변환되어 있다고 가정
    결과: shape (numUsed, 1) complex64
    """
    dc = cfg["dispComp"]
    dc_np = np.asarray(dc)
    # 2D 정방 행렬이면 대각선
    if dc_np.ndim == 2 and dc_np.shape[0] == dc_np.shape[1]:
        dc_np = np.diag(dc_np)
    dc_t = torch.as_tensor(dc_np, dtype=torch.complex64, device=dev).flatten()
    # 길이 맞추기
    if dc_t.numel() == used_idx.numel():
        sel = dc_t
    else:
        # numSamples 길이라고 가정하고 used로 선택
        sel = dc_t[used_idx]
    return sel.view(-1, 1)

def _interp_resample_A_scans(
    x_src: torch.Tensor, y_src: torch.Tensor, num_used: int
) -> torch.Tensor:
    """
    x_src: (num_used,) in [0,1] (정렬 가정 X → spline가 자체적으로 처리)
    y_src: (num_used, W) float64
    반환: (num_used, W) float32
    """
    x_tgt = torch.linspace(0.0, 1.0, num_used, dtype=torch.float64, device=y_src.device)
    if _HAS_TCS:
        # torchcubicspline (GPU 가능)
        coeffs = natural_cubic_spline_coeffs(x_src, y_src.unsqueeze(0))  # (1, N, W)
        spline = NaturalCubicSpline(coeffs)
        out = spline.evaluate(x_tgt).squeeze(0).to(torch.float32)       # (N, W)
        return out
    else:
        # SciPy CPU 폴백
        xs = x_src.detach().cpu().numpy().astype(np.float64)
        ys = y_src.detach().cpu().numpy().astype(np.float64)            # (N, W)
        f = interp1d(xs, ys, kind="cubic", axis=0, assume_sorted=False, fill_value="extrapolate")
        yt = f(x_tgt.detach().cpu().numpy().astype(np.float64))         # (N, W)
        return torch.as_tensor(yt, dtype=torch.float32, device=y_src.device)