# core/frame_process/process_frame_noresamp.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from scipy.interpolate import interp1d  # Always import for CPU fallback

# Optional: Use torchcubicspline for GPU interpolation.
try:
    from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
    _HAS_TCS = True
except ImportError:
    _HAS_TCS = False

DEBUG = True
def dbg(msg: str):
    """Prints a debug message if DEBUG is True."""
    if DEBUG:
        print(msg)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _ind2sub_F(shape_xy: Tuple[int, int], k_1b: int) -> Tuple[int, int]:
    """
    Equivalent to MATLAB's ind2sub(shape, k) for column-major order.
    Returns 0-based indices (m_idx, n_idx).
    """
    x, y = int(shape_xy[0]), int(shape_xy[1])
    k = int(k_1b) - 1  # Convert to 0-based index
    m_idx = k % x
    n_idx = k // x
    return m_idx, n_idx

def _as_long_0b_used_indices(used_samples) -> torch.Tensor:
    """
    Converts a list or boolean array of used samples to a 0-based LongTensor.
    If boolean, it finds the non-zero indices. If numeric, it subtracts 1.
    """
    us = np.asarray(used_samples)
    if us.dtype == np.bool_:
        sel = np.nonzero(us)[0]
    else:
        sel = us.astype(np.int64) - 1  # Convert from 1-based to 0-based
    return torch.as_tensor(sel.reshape(-1), dtype=torch.long)

def _interp_A_scan(x_src: torch.Tensor, y_src: torch.Tensor, T: int) -> torch.Tensor:
    """
    Interpolates A-scan data using cubic splines.
    Uses torchcubicspline on GPU if available, otherwise falls back to SciPy on CPU.

    Args:
        x_src: Source x-coordinates (K,). Expected in [0,1], float64.
        y_src: Source y-values (K, W), float64.
        T: Number of points for the target uniform grid.

    Returns:
        Interpolated y-values (T, W) as a float32 tensor on the source device.
    """
    x_tgt = torch.linspace(0.0, 1.0, T, dtype=torch.float64, device=y_src.device)
    if _HAS_TCS and y_src.is_cuda:
        # GPU implementation
        coeffs = natural_cubic_spline_coeffs(x_src, y_src.unsqueeze(0))  # (1, K, W)
        spline = NaturalCubicSpline(coeffs)
        out = spline.evaluate(x_tgt).squeeze(0).to(torch.float32)
        return out
    else:
        # CPU fallback
        xs = x_src.detach().cpu().numpy().astype(np.float64)
        ys = y_src.detach().cpu().numpy().astype(np.float64)  # (K, W)
        f = interp1d(xs, ys, kind="cubic", axis=0, assume_sorted=False, fill_value="extrapolate")
        yt = f(x_tgt.detach().cpu().numpy().astype(np.float64))  # (T, W)
        return torch.as_tensor(yt, dtype=torch.float32, device=y_src.device)

# =========================================================
# (1) PER-FRAME PROCESSING (NO-RESAMPLE VERSION)
# =========================================================

def process_frame_noresamp(
    fid: "io.BufferedReader",
    frame_1b: int,
    vol_1b: int,
    info: dict,
    *,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch port of MATLAB's processFrameNoresamp function.
    Processes a single OCT frame without B-scan resampling. This involves
    A-scan interpolation, dispersion compensation, IFFT, noise normalization,
    and padding.

    Args:
        fid: Binary file handle for the raw data.
        frame_1b: The 1-based index of the frame to process.
        vol_1b: The 1-based index of the volume (sub-division).
        info: A dictionary containing all processing parameters.
        device: The target device ('cuda' or 'cpu'). Falls back to 'cpu' if
                'cuda' is requested but not available.

    Returns:
        A tuple containing:
        - fringes_result: Processed fringes in k-space (numFTSamples, numScanLines), complex64.
        - img_calib: Calibrated image data (numUsedSamples, numScanLines), complex64.
    """
    # Device selection with fallback
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA device '{device}' not available. Falling back to 'cpu'.")
        dev = torch.device('cpu')

    # (a) Calculate sub-volume index and shifts (column-major)
    m_idx, n_idx = _ind2sub_F(tuple(info["subdivFactors"]), vol_1b)

    frame_shift = int(round(info["centerVolY"][n_idx] - info["numVolFrames"] / 2))
    # Motion shift is disabled to match the original provided code's logic.
    motion_shift = 0
    line_shift = int(
        round(info["centerVolX"][m_idx] - info["numVolLines"] / 2 - motion_shift)
    )
    dbg(f"[a] vol={vol_1b} -> (m={m_idx}, n={n_idx}), frame_shift={frame_shift}, "
        f"line_shift={line_shift}, motion={motion_shift}")

    # (b) Define raw line window (no-resample version uses all scan lines)
    num_raw_lines = int(info["numScanLines"])
    raw_line_shift = 0

    # (c) Read the frame data from the binary file
    line_idx = int(
        info["initLineShift"]
        + (frame_shift + frame_1b - 1) * (info["numScanLines"] + info["numFlybackLines"])
        + raw_line_shift
    )
    offset_bytes = 2 * (int(info["trigDelay"]) + int(info["numSamples"]) * (line_idx - 1))
    fid.seek(offset_bytes, 0)

    buffer = fid.read(2 * int(info["numSamples"]) * num_raw_lines)
    raw = np.frombuffer(buffer, dtype=np.uint16)
    fringes = raw.astype(np.float64).reshape((int(info["numSamples"]), num_raw_lines), order="F")

    used_idx = _as_long_0b_used_indices(info["usedSamples"])
    fringes = fringes[used_idx.cpu().numpy(), :]  # (numUsed, num_raw_lines)
    K = fringes.shape[0]
    dbg(f"[c] read fringes: {fringes.shape}  (K=numUsed={K}, W=num_raw_lines={num_raw_lines})")

    # (d) Background and oscillation subtraction
    if info.get("adaptiveBG", False):
        bg_mean = np.asarray(info["bgMean"], dtype=np.float64).reshape(-1)
        bg_center = bg_mean - 32768.0
        den = np.sum(bg_center ** 2) if np.sum(bg_center ** 2) != 0 else 1.0

        fringesBS = np.empty_like(fringes)
        for col in range(num_raw_lines):
            sig = fringes[:, col] - 32768.0
            a = float(np.sum(sig * bg_center) / den)
            fringesBS[:, col] = sig / max(a, 1e-12) - bg_mean + 32768.0
    else:
        bg = np.asarray(info["bgMean"], dtype=np.float64).reshape(-1, 1)
        fringesBS = fringes - bg

    if info.get("adaptiveBGOsc", False) and np.any(np.asarray(info.get("bgOsc", 0)) != 0):
        bg_osc = np.asarray(info["bgOsc"], dtype=np.float64).reshape(-1)
        e = np.sum(bg_osc ** 2)
        if e > 0:
            for col in range(num_raw_lines):
                a = float(np.sum(fringesBS[:, col] * bg_osc) / e)
                fringesBS[:, col] -= a * bg_osc

    # (e) A-scan interpolation (from resampTraceA to a linear space)
    x_src = torch.as_tensor(np.asarray(info["resampTraceA"], dtype=np.float64).reshape(-1), device=dev)
    y_src = torch.as_tensor(fringesBS, dtype=torch.float64, device=dev)  # (K,W)
    y_res = _interp_A_scan(x_src, y_src, K)  # (K,W) float32

    # (f) Dispersion compensation, IFFT, noise normalization, and band cutting
    disp = np.asarray(info["dispComp"])
    if disp.ndim == 2 and disp.shape[0] == disp.shape[1]:
        disp = np.diag(disp)
    disp = torch.as_tensor(disp, dtype=torch.complex64, device=dev).flatten()

    if disp.numel() == K:
        disp_used = disp
    else:
        # Slice dispersion vector if its length matches numSamples
        disp_used = disp[used_idx.to(disp.device)]
    disp_used = disp_used.view(-1, 1)  # (K,1)

    fringes_calib = (y_res.to(torch.complex64) * disp_used)  # (K,W)
    img = torch.fft.ifft(fringes_calib, dim=0)               # (K,W) complex64

    noise = torch.as_tensor(np.asarray(info["noiseProfile"], dtype=np.float64).reshape(-1, 1),
                            dtype=torch.float32, device=dev).repeat(1, num_raw_lines)
    img_calib = (img / noise.to(torch.complex64)).contiguous()

    # Low-cut (background band) and high-cut (upper half)
    bg_bw = float(info.get("bgBW", 0))
    numFT = int(info["numFTSamples"])
    cut_lo = int(round(bg_bw * K / numFT))
    if cut_lo > 0:
        img_calib[:cut_lo, :] = 0
    img_calib[int(np.ceil(K / 2.0)):, :] = 0  # Match MATLAB's 1-based ceil logic

    # (g) FFT and zero-padding to final size
    fcal = torch.fft.fft(img_calib, dim=0)  # (K,W)
    pad_len = numFT - K
    pad = torch.zeros((pad_len, num_raw_lines), dtype=fcal.dtype, device=dev)
    
    # FDFlip option determines padding location
    fringes_result = (torch.cat([fcal, pad], dim=0)
                      if info.get("FDFlip", False)
                      else torch.cat([pad, fcal], dim=0))  # (numFT, W=numScanLines)

    return fringes_result, img_calib

# =========================================================
# (2) PER-VOLUME PROCESSING
# =========================================================

def process_frame_noresamp(fid, frame_idx: int, vol_idx: int, info: dict, device: str | torch.device):
    """
    .data 파일에서 단일 프레임을 읽어 전처리 후 복소수 fringes 텐서를 반환합니다.
    
    Args:
        fid (file object): 열려있는 .data 파일 핸들.
        frame_idx (int): 처리할 프레임 인덱스 (1-based).
        vol_idx (int): 처리할 볼륨 인덱스 (1-based).
        info (dict): 모든 파라미터를 담고 있는 딕셔너리.
        device: PyTorch 디바이스.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - fr (torch.Tensor): 전처리된 복소수 프린지 (K, L).
            - imgc (torch.Tensor): IFFT를 통해 얻은 복소수 B-scan 이미지 (Z, L).
    """
    # 1. 파라미터 추출
    num_samples = int(info['numSamples'])
    num_img_lines = int(info['numImgLines'])
    num_img_frames = int(info['numImgFrames'])
    num_flyback = int(info['numFlybackLines'])
    num_vol_frames = int(info['numVolFrames'])
    
    used_samples_idx = np.asarray(info['usedSamples']).ravel() - 1
    K = len(used_samples_idx)
    
    # 2. 파일 포인터 이동
    line_shift = (vol_idx - 1) * (num_img_lines + num_flyback) * num_vol_frames
    frame_shift = (frame_idx - 1) * num_img_lines
    total_shift = line_shift + frame_shift
    
    fid.seek(total_shift * num_samples * 2, 0) # 2 bytes per uint16

    # 3. 파일 읽기 (MATLAB의 Column-major 방식과 동일하게)
    raw_frame = np.fromfile(fid, count=num_samples * num_img_lines, dtype=np.uint16)
    raw_frame = raw_frame.reshape((num_img_lines, num_samples)).T
    
    # 4. 스펙트럼 자르기
    frame_k_space = raw_frame[used_samples_idx, :]

    # --- ✨ DC 옵셋 문제 해결을 위한 핵심 로직 ✨ ---
    # 5. 배경 신호(DC 옵셋) 제거
    bg_mean = info['bgMean']
    bg_osc = info.get('bgOsc', None)

    # uint16 -> float32로 변환 후 bgMean을 빼서 신호를 0 주변으로 이동
    frame_no_bg = frame_k_space.astype(np.float32) - bg_mean[:, np.newaxis]

    # 주기적 노이즈(bgOsc)가 있다면 추가로 제거
    if bg_osc is not None and np.any(bg_osc):
        a_osc = np.sum(frame_no_bg * bg_osc[:, np.newaxis], axis=0) / np.sum(bg_osc**2)
        frame_no_bg -= a_osc * bg_osc[:, np.newaxis]
    # ------------------------------------------------

    # 6. 분산 보정 및 스펙트럼 윈도우 적용
    disp_comp = info['dispComp']
    spectral_window = info['spectralWindow']
    correction = disp_comp * spectral_window
    
    fringes_np = frame_no_bg * correction[:, np.newaxis]

    # 7. PyTorch 텐서로 변환
    fr = torch.from_numpy(fringes_np.astype(np.complex64)).to(device)

    # 8. IFFT 및 z-축 자르기
    num_img_pixels = int(info['numImgPixels'])
    imgc = torch.fft.ifft(fr, dim=0)[:num_img_pixels, :]
    
    return fr, imgc

def process_and_save_volume(dfile_path: str, vol_idx: int, info: dict, device: str | torch.device) -> torch.Tensor:
    """
    볼륨 전체를 프레임 단위로 처리하여 하나의 3D fringes 텐서로 합칩니다.
    """
    num_img_frames = int(info['numImgFrames'])
    all_fringes = []
    
    with open(f"{dfile_path}.data", "rb") as fid:
        for i in range(1, num_img_frames + 1):
            if (i % 50 == 0) or (i == num_img_frames):
                print(f"  Processing frame {i}/{num_img_frames}...")
            
            # 각 프레임을 처리 (DC 옵셋 제거 포함)
            fr, _ = process_frame_noresamp(fid, i, vol_idx, info, device=device)
            all_fringes.append(fr)
            
    # 처리된 모든 2D 프린지들을 3D 볼륨으로 스택
    fringes_vol = torch.stack(all_fringes, dim=2)
    return fringes_vol