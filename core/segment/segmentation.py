import torch
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
import scipy.ndimage as ndi
from scipy.signal import find_peaks
from typing import Tuple, Optional, Union
from core.registration.dftregistration import dftregistration1d
from scipy.interpolate import griddata 
import math
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator



def gaussian_nanblur2d(layer: np.ndarray,
                       sig_x: float,
                       sig_y: float,
                       clip: tuple[int, int] | None = None) -> np.ndarray:
    """
    2-D Gaussian blur that preserves NaNs (MATLAB nanconv 'nanout').

    Parameters
    ----------
    layer : (W, H) float32   # 1-based depth map
    sig_x : float            # σ along X (scan line)
    sig_y : float            # σ along Y (frame)
    clip  : (min,max) or None  # depth range to clamp; pass (1, L)

    Returns
    -------
    blurred : same shape, NaN-safe gaussian-smoothed
    """
    mask = np.isfinite(layer).astype(np.float32)         # 1 where valid
    data = np.nan_to_num(layer, nan=0.0).astype(np.float32)

    smooth_data = ndi.gaussian_filter(data,  sigma=(sig_x, sig_y),
                                  mode="nearest", truncate=1.0)
    smooth_mask = ndi.gaussian_filter(mask,  sigma=(sig_x, sig_y),
                                  mode="nearest", truncate=1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        blurred = smooth_data / smooth_mask
    blurred[smooth_mask == 0] = np.nan                   # all-NaN patch

    if clip is not None:
        lo, hi = clip
        blurred = np.clip(blurred, lo, hi, where=np.isfinite(blurred))

    return blurred.astype(np.float32)

def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """이미 device에 있으면 그대로, 아니면 이동"""
    return x if x.device == device else x.to(device)

def to_numpy(x):
    """torch.Tensor → NumPy (자동 CPU 이동) / 그 외엔 그대로"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _complex_nanmean(x: torch.Tensor,
                    dim: int,
                    keepdim: bool = False) -> torch.Tensor:
    """
    복소 텐서용 NaN-무시 평균
    (MATLAB mean(x,dim,'omitnan') 완전 대응)
    """
    mask = torch.isnan(x.real) | torch.isnan(x.imag)        # 제외 마스크
    safe = x.clone()
    safe[mask] = 0.0 + 0.0j                                 # NaN → 0

    cnt = (~mask).sum(dim=dim, keepdim=keepdim)
    out = safe.sum(dim=dim, keepdim=keepdim)                # 복소 합

    # division – cnt==0 인 위치는 NaN 으로
    out = out / cnt.clamp(min=1)
    out[cnt == 0] = torch.nan + 1j*torch.nan
    return out

def _reg_error(ref_ft: torch.Tensor, mov_ft: torch.Tensor) -> float:
    """
    Guizar 논문과 동일한 품질지표:
      error = sqrt( 1 - |Σ ref*conj(mov)|^2 / (||ref||^2 · ||mov||^2) )
    """
    num = torch.sum(ref_ft * torch.conj(mov_ft))
    denom = torch.linalg.norm(ref_ft) * torch.linalg.norm(mov_ft)
    err = 1.0 - (torch.abs(num) / (denom + 1e-12))
    return float(torch.sqrt(torch.clamp(err, min=0.0)))

def _complex_nanmean_numpy(a: np.ndarray, axis: int):
    mask = np.isnan(a.real) | np.isnan(a.imag)
    safe = a.copy()
    safe[mask] = 0
    cnt = (~mask).sum(axis=axis)
    cnt[cnt == 0] = 1
    return safe.sum(axis=axis) / cnt

def _central_diff(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float64, copy=False)
    L, W, H = vol.shape       
    upper = np.zeros((1, W, H), dtype=vol.dtype)
    diff  = np.diff(vol, axis=0) / 2.0
    lower = np.zeros_like(upper)
    return np.concatenate([upper, diff], axis=0) + np.concatenate([diff, lower], axis=0)

def _apply_subpixel_shift(ft_line: torch.Tensor, shift: float) -> torch.Tensor:
    L = ft_line.shape[0]
    k = torch.fft.fftfreq(L, d=1.0, device=ft_line.device)    # (0.., -..)
    phase = torch.exp(-2j * np.pi * k * shift)                # 부호 ‘−’ 유지
    return ft_line * phase.type_as(ft_line)

def _apply_subpixel_shift_np(ft_line: np.ndarray, shift: float) -> np.ndarray:
    L = ft_line.shape[0]
    k = np.fft.fftfreq(L, d=1.0)
    phase = np.exp(-2j * np.pi * k * shift)
    return ft_line * phase


def _nanmin_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 배열의 원소별 NaN-무시 최소값"""
    out = np.where(np.isnan(a), b, np.where(np.isnan(b), a, np.minimum(a, b)))
    return out

def _nanmax_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 배열의 원소별 NaN-무시 최대값"""
    out = np.where(np.isnan(a), b, np.where(np.isnan(b), a, np.maximum(a, b)))
    return out

def _nanstd(x: torch.Tensor, dim: Tuple[int, ...] = (), keepdim: bool = False):
    """torch.nanstd 대체 함수 (PyTorch < 1.11)"""
    mask = torch.isnan(x)
    cnt  = (~mask).float().sum(dim=dim, keepdim=keepdim).clamp(min=1)
    mean = torch.nansum(x, dim=dim, keepdim=True) / cnt
    var  = torch.nansum((x - mean) ** 2, dim=dim, keepdim=keepdim) / cnt
    if not keepdim:
        var = var.squeeze(dim)
    return torch.sqrt(var)

def _nanmin(x: torch.Tensor,
        dim: Optional[int | Tuple[int, ...]] = None,
        keepdim: bool = False):
    """
    torch.nanmin 대체 함수.
    NaN 은 무시하고 최소값을 구한다.
    모든 값이 NaN 이면 결과도 NaN.
    """
    if dim is None:                     # 전체에서 찾기
        mask        = torch.isnan(x)
        if mask.all():
            return torch.tensor(float('nan'), dtype=x.dtype, device=x.device)
        return x[~mask].min()

    # dim 이 하나 또는 다중 tuple 인 경우
    if isinstance(dim, int):
        dim = (dim,)

    # NaN → +∞ 로 바꾼 뒤 최소값
    inf     = torch.tensor(float('inf'), dtype=x.dtype, device=x.device)
    x_safe  = x.clone()
    x_safe[torch.isnan(x_safe)] = inf
    min_val, _ = x_safe.min(dim=dim, keepdim=keepdim)

    # 전부 NaN 이었던 위치 찾아 다시 NaN 삽입
    all_nan = torch.isnan(x).all(dim=dim, keepdim=keepdim)
    min_val[all_nan] = float('nan')
    return min_val

def _nanmax(x: torch.Tensor,
        dim: Optional[int | Tuple[int, ...]] = None,
        keepdim: bool = False):
    """
    torch.nanmax 대체 함수.
    NaN 을 무시하고 최대값을 구한다.
    모든 값이 NaN 이면 결과도 NaN.
    """
    if dim is None:                     # 전체에서 찾기
        mask = torch.isnan(x)
        if mask.all():
            return torch.tensor(float('nan'), dtype=x.dtype, device=x.device)
        return x[~mask].max()

    # dim 이 하나 또는 다중 tuple 인 경우
    if isinstance(dim, int):
        dim = (dim,)

    # NaN → -∞ 로 바꾼 뒤 최대값
    ninf    = torch.tensor(float('-inf'), dtype=x.dtype, device=x.device)
    x_safe  = x.clone()
    x_safe[torch.isnan(x_safe)] = ninf
    max_val, _ = x_safe.max(dim=dim, keepdim=keepdim)

    # 전부 NaN 이었던 위치 찾아 다시 NaN 삽입
    all_nan = torch.isnan(x).all(dim=dim, keepdim=keepdim)
    max_val[all_nan] = float('nan')
    return max_val

def extract_prl_slab(int_img: torch.Tensor, Info: dict):
    L, W, H = int_img.shape

    # 1) valid mask
    valid = torch.any(int_img != 0, dim=0).to(torch.float64)  # shape (W, H)
    valid[valid == 0] = float("nan")

    # 2) masked intensity & axial average
    masked = int_img * valid.unsqueeze(0)                    # broadcast to (L, W, H)
    int_img_avg = torch.nanmean(torch.nanmean(masked, dim=1), dim=1)  # shape (L,)

    # 3) threshold from tail
    tail_start = int(Info["numImgPixels"] * 15 // 16)

    tail_slice = int_img_avg[tail_start:]                      # (L/16,)
    tail_nonan = tail_slice[~torch.isnan(tail_slice)]

    if tail_nonan.numel() == 0:
        raise RuntimeError(
            "tail section (last 1/16 along depth) is entirely NaN.\n"
            "→ FFT/IFFT 스케일링 혹은 입력 값 자체를 확인하세요."
        )

    tail_max = tail_nonan.max()
    if tail_max <= 0:
        warnings.warn(
            f"[extract_prl_slab] tail_max={tail_max:.3g} ≤ 0 — "
            "threshold가 0이 되어 PRL 슬랩이 잘못 잡힐 수 있습니다.",
            RuntimeWarning
        )

    th_val = float(tail_max) * 2.0

    # 4) first/last indices above threshold
    inds = torch.where(int_img_avg >= th_val)[0]
    if inds.numel() == 0:            # threshold crossing이 전혀 없을 때
        raise RuntimeError(
            "No depth samples exceeded th_val (%.3g). "
            "FFT 정규화나 노이즈 레벨을 점검하세요." % th_val
        )

    sidx, eidx = int(inds[0]), int(inds[-1])

    # 5) enforce max thickness <= 180µm
    depth_per_pix = Info["depthPerPixel"]
    max_thick_pix = 180e-3 / depth_per_pix

    pidx      = eidx
    thickness = pidx - sidx + 1

    while thickness > max_thick_pix and pidx > 0:
        pidx -= 1
        if int_img_avg[pidx] > th_val:
            th_val = float(int_img_avg[pidx])
            sidx   = int(torch.where(int_img_avg >= th_val)[0][0])
            eidx   = pidx
        thickness = pidx - sidx + 1

    return sidx, eidx, th_val


def extract_prl_slab_debug(int_img: torch.Tensor, Info: dict, debug: bool = False):
    from typing import Tuple,Optional
    """
    int_img : (L, W, H)  intensity volume (float / double)
    Info    : MATLAB-style Info 구조체(dict)
    debug   : True → 단계별 변수‧값 출력
    """

    L, W, H = int_img.shape
    if debug:
        print(f"[INPUT]  int_img shape = {int_img.shape}, dtype = {int_img.dtype}")

    # 1) valid mask ───────────────────────────────────────────────
    valid = torch.any(int_img != 0, dim=0).to(torch.float64)   # (W, H)
    valid[valid == 0] = float("nan")

    if debug:
        nz = torch.isfinite(valid).sum().item()
        print(f"[1] valid mask  → finite px = {nz}/{W*H}  (NaN = {W*H-nz})")

    # 2) masked intensity & axial average ─────────────────────────
    masked       = int_img * valid.unsqueeze(0)                # (L, W, H)
    int_img_avg = torch.nanmean(int_img * valid.unsqueeze(0), dim=(1, 2))

    if debug:
        print(f"[2] masked stats   mean={torch.nanmean(masked):.4g}  "
              f"std={_nanstd(masked):.4g}")
        print(f"    int_img_avg    shape={int_img_avg.shape}  "
              f"min/max = {_nanmin(int_img_avg):.3g}/{_nanmax(int_img_avg):.3g}")

    # 3) threshold from tail ─────────────────────────────────────
    tail_start = int(np.ceil(Info["numImgPixels"] * 15 / 16))
    tail_max   = _nanmax(int_img_avg[tail_start:])
    th_val     = tail_max * 2.0

    if debug:
        print(f"[3] tail_start={tail_start}, tail_max={tail_max:.3g},  th_val={th_val:.3g}")

    # 4) first/last indices above threshold ───────────────────────
    inds = torch.where(int_img_avg >= th_val)[0]
    if inds.numel() > 0:
        sidx = int(inds[0])
        eidx = int(inds[-1])
    else:
        if debug:
            print("[4] 위-threshold 구간이 없습니다.  →  None 반환")
        return None, None, th_val

    if debug:
        print(f"[4] sidx={sidx}, eidx={eidx},  thickness={eidx-sidx+1}")

    # 5) enforce max thickness ≤ 180 µm ───────────────────────────
    depth_per_pix  = Info["depthPerPixel"]
    L              = Info["numImgPixels"]            # == int_img_avg.size(0)
    max_thick_pix  = 180e-3 / depth_per_pix
    min_thick_px   = 60e-3 / depth_per_pix

    pidx      = eidx
    thickness = pidx - sidx + 1

    if debug:
        print(f"[5-a] 시작 thickness = {thickness:.1f} px,  "
              f"허용 최대 = {max_thick_pix:.1f} px")

    while thickness > min_thick_px and pidx >= 0:
        pidx -= 1
        if int_img_avg[pidx] > th_val:
            th_val = float(int_img_avg[pidx])
            sidx   = int(torch.where(int_img_avg >= th_val)[0][0])
            eidx   = pidx
        thickness = pidx - sidx + 1
        if debug:
            print(f"      shrink: pidx={pidx}, th_val={th_val:.3g}, "
                  f"thickness={thickness:.1f}")

    if debug:
        print(f"[5-b] 최종   sidx={sidx}, eidx={eidx}, thickness={thickness:.1f}")
    

    pidx = eidx                                   # 0-based
    sidx_cand = torch.where(int_img_avg[:eidx + 1] < th_val)[0]
    sidx = int(sidx_cand[-1]) + 1 if sidx_cand.numel() else 0
    thickness = pidx - sidx + 1

    while thickness > min_thick_px and pidx > 0:
        pidx -= 1
        if int_img_avg[pidx] > th_val:            # 새로운 threshold 후보
            th_val = float(int_img_avg[pidx])
            sidx_cand = torch.where(int_img_avg[:sidx] < th_val)[0]
            sidx = int(sidx_cand[-1]) + 1 if sidx_cand.numel() else 0
            eidx = pidx
        thickness = pidx - sidx + 1


    while thickness < min_thick_px and pidx < L - 1:
        pidx += 1
        if int_img_avg[pidx] < th_val:
            th_val = float(int_img_avg[pidx])
            sidx_cand = torch.where(int_img_avg[:sidx] < th_val)[0]
            sidx = int(sidx_cand[-1]) + 1 if sidx_cand.numel() else 0
        eidx = pidx
        thickness = pidx - sidx + 1

    plot_prl_slab(int_img_avg, sidx, eidx, th_val)
    plt.show()

    return sidx, eidx, th_val    

def segment_prl_nfl(
    int_img: torch.Tensor,           # (L,W,H) intensity
    info: dict,
    sub_onh_mask: np.ndarray | None = None,
    debug: bool = False,
):
    """
    파이프라인 개요
      1) PRL slab 경계 추정 (extract_prl_slab_debug)
      2) 정규화 & 3-D Gaussian 필터링 (위 prl_norm_and_filter)
      3) ISOS/RPE 세그멘트      ← TODO  (segmentPRL 에 대응)
      4) ILM/NFL 세그멘트      ← TODO  (segmentNFL3D 에 대응)
    """
    device = int_img.device
    sidx, eidx, th_val = extract_prl_slab_debug(int_img, info, debug=debug)

    W, H = int_img.shape[1:]
    ILM  = torch.full((W, H), float('nan'), device=device)
    NFL  = torch.full((W, H), float('nan'), device=device)
    ISOS = torch.full((W, H), float('nan'), device=device)
    RPE  = torch.full((W, H), float('nan'), device=device)

    if sidx is None or eidx is None or eidx - sidx <= 0:
        if debug:
            print("▶ PRL slab 경계가 유효하지 않아 모든 결과를 NaN 으로 반환합니다.")
        return ILM, NFL, ISOS, RPE

    # ── axial 평균 (intImgAvg) 재계산 ──
    valid = torch.any(int_img != 0, dim=0).to(torch.float32)
    valid[valid == 0] = float('nan')
    int_img_avg = torch.nanmean(torch.nanmean(int_img * valid.unsqueeze(0), dim=1), dim=1)

    if isinstance(int_img_avg, torch.Tensor):
        int_img_avg = int_img_avg.cpu().numpy()

    # ── (1) 정규화 & 필터 ──
    int_img_filt,prl_slab_start, prl_slab_end, int_img_norm = prl_norm_and_filter(
        int_img, int_img_avg, sidx, eidx, info, sub_onh_mask, device
    )
    
    int_img_filt_slab = int_img_filt[prl_slab_start:prl_slab_end+1]
    int_img_norm_slab = int_img_norm[prl_slab_start:prl_slab_end+1]
    
    # ── (2) ISOS / RPE 세그멘트  ← segmentPRL 에 해당 -------
    ISOS, RPE = segment_prl_dbg(
        int_img_filt= int_img_filt_slab,
        int_img_raw = int_img_norm_slab,
        info=info,
        verbose=True,
        debug=True
    )

    ISOS = ISOS + prl_slab_start
    RPE  = RPE  + prl_slab_start

    vol_top = int_img_filt[:sidx] 

    nLines = info["numImgLines"]
    nFrames = info["numImgFrames"]

    if vol_top.shape[1] == nFrames and vol_top.shape[2] == nLines:
        vol_top = vol_top.transpose(0,2,1)

    # ── (3) ILM / NFL 세그멘트  ← segmentNFL3D 에 해당 -------
    ILM, NFL = segment_nfl_3d(
    vol_top,        # PRL slab 윗부분만
    info,
    verbose=debug,
    )

    # ---------- 2-D Gaussian post-smoothing (matlab nanconv) ----------
    W, H = ILM.shape
    sig_x = np.ceil(W / 10)
    sig_y = np.ceil(H / 10)
    depth_max = info["numImgPixels"]        # == L

    ILM  = gaussian_nanblur2d(ILM , sig_x, sig_y, clip=(1, depth_max))
    NFL  = gaussian_nanblur2d(NFL , sig_x, sig_y, clip=(1, depth_max))
    ISOS = gaussian_nanblur2d(ISOS, sig_x, sig_y, clip=(1, depth_max))
    RPE  = gaussian_nanblur2d(RPE , sig_x, sig_y, clip=(1, depth_max))

    return ILM, NFL, ISOS, RPE

def find_edge_layer(mask: np.ndarray,
                    which: str = "last",        # "first" | "last"
                    extrapolate: bool = True) -> np.ndarray:
    """
    Parameters
    ----------
      mask : (L,W) bool
      which: 'first' or 'last' (edge to pick inside each column)
      extrapolate : True → 선형보간+recent-edge 외삽, False → 그대로 NaN
    Returns
    -------
      layer : (W,) float   (1-based depth, NaN 허용)
    """
    L, W = mask.shape
    layer = np.full(W, np.nan, dtype=np.float32)

    # ① 각 column 에서 첫/마지막 1
    for col in range(W):
        rows = np.flatnonzero(mask[:, col])
        if rows.size:
            layer[col] = rows[0] + 1 if which == "first" else rows[-1] + 1

    valid = np.isfinite(layer)
    n_valid = valid.sum()

    if n_valid == 0:
        return np.ones(W, np.float32) if extrapolate else layer

    if n_valid == 1 and extrapolate:
        layer[:] = layer[valid][0]

    elif n_valid < W and extrapolate:
        first, last = np.flatnonzero(valid)[[0, -1]]
        span = np.arange(first, last + 1)
        layer[span] = np.interp(span, np.flatnonzero(valid), layer[valid])
        layer[:first] = layer[first]
        layer[last + 1:] = layer[last]

    # ② round & clip to [1,L]
    ok = np.isfinite(layer)
    layer[ok] = np.clip(np.round(layer[ok]), 1, L)
    return layer

def find_max_layer(img: np.ndarray, cont: bool = True) -> np.ndarray:
    """
    Parameters
    ----------
      img : (L,W) float  – 양수 값 선호
      cont: MATLAB 'cont' 플래그 (연속성 강제)
    Returns
    -------
      layer : (W,) float  (sub-pixel 보정 포함, 1-based)
    """
    L, W = img.shape
    # 중앙 peak
    val_center = img.max(axis=0)
    layer = img.argmax(axis=0).astype(np.float32) + 1        # 1-based

    # ---------------- 연속성(con t) 보정 -----------------------------
    if cont and W > 1 and np.any(np.abs(np.diff(layer)) > 1):
        # diff<=1  True/False 구간 라벨링
        diff_good = np.abs(np.diff(layer)) <= 1
        # label 연속 True 구간
        lbl = np.zeros_like(diff_good, int)
        if diff_good.any():
            lbl[0] = diff_good[0]
            for i in range(1, diff_good.size):
                lbl[i] = lbl[i - 1] if diff_good[i] else lbl[i - 1] + 1
            # 가장 긴 True 구간
            true_lbls, counts = np.unique(lbl[diff_good], return_counts=True)
            best_lbl = true_lbls[counts.argmax()]
            idx_block = np.where(lbl == best_lbl)[0]          # 0-based in diff
            left_end = idx_block.min()          # diff index ⇒ col index same
            right_end = idx_block.max() + 1     # 우측 col

            # left side ↘
            for col in range(left_end - 1, -1, -1):
                peaks, _ = find_peaks(img[:, col])
                if peaks.size == 0:
                    layer[col] = layer[col + 1]
                else:
                    best = peaks[np.argmin((peaks + 1 - layer[col + 1]) ** 2)]
                    layer[col] = best + 1

            # right side ↗
            for col in range(right_end + 1, W):
                peaks, _ = find_peaks(img[:, col])
                if peaks.size == 0:
                    layer[col] = layer[col - 1]
                else:
                    best = peaks[np.argmin((peaks + 1 - layer[col - 1]) ** 2)]
                    layer[col] = best + 1

    # ---------------- sub-pixel refinement ---------------------------
    mid = (layer > 1) & (layer < L)
    idx = np.flatnonzero(mid)
    if idx.size:
        z = layer[idx].astype(int) - 1               # 0-based
        val_minus = img[z - 1, idx]
        val_plus  = img[z + 1, idx]
        denom = 2 * val_center[idx] - val_plus - val_minus
        shift = 0.5 * np.clip((val_plus - val_minus) / denom, -1, 1)
        layer[idx] = layer[idx] + shift

    return layer

def segment_nfl_2d(int_img_filt: np.ndarray,
                   info: dict,
                   show_fig: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    int_img_filt : (L, W) 3-D gaussian filtered B-scan (single frame)
    info : dict   – MATLAB Info 구조체에 대응되는 값들
        필수 key
            sigmaZ, ILMshift, NFLmaxRange, NFLminRange
            segmentFigureOn (bool)
            figurePos22     (무시 가능·plot 용)
    show_fig : True/False → 강제, None → info['segmentFigureOn'] 따라감
    Returns
    -------
    ILM, NFL : (W,) float – 1-based depth 좌표 (NaN 허용, sub-pixel 포함)
    """
    L, W = int_img_filt.shape
    if show_fig is None:
        show = info.get("segmentFigureOn", False)
    else:
        show = show_fig

    # ---------------- ① column-wise 정규화 --------------------------
    col_max = np.maximum(12, int_img_filt.max(axis=0))
    int_img_norm = int_img_filt / col_max * 50.0        # (L,W)

    # ---------------- ② ILM mask & edge -----------------------------
    ilm_mask = (int_img_norm.T < 10)      # (W,L) for ease
    cc_lbl, n_cc = label(ilm_mask, connectivity=2, return_num=True)
    if n_cc:
        ilm_mask[:] = (cc_lbl == 1)

    ILM = find_edge_layer(ilm_mask.T, which="last", extrapolate=True)

    # ---------------- ③ weight & ILM shift -------------------------
    sigma_z = float(info["sigmaZ"])
    sigma_z_int = int(info["sigmaZ"])

    weight = np.empty(W, dtype=np.float32)
    for x in range(W):
        if np.isnan(ILM[x]):
            weight[x] = 0.0
            continue

        z0 = int(np.floor(ILM[x] - 1))                            # 0-based
        z1_incl = min(int(np.floor(ILM[x] - 1 + 3*sigma_z_int)), L - 1)
        weight[x] = int_img_filt[z0 : z1_incl + 1, x].max()

    weight = np.clip((weight - 10), 0, 80) / 80.0
    ILM = ILM + info["ILMshift"] * ((3 + weight) / 4)

    # ---------------- ④ 1-차 미분 이미지 ----------------------------
    diff_img = np.vstack([np.diff(int_img_filt, axis=0) / 2,
                      np.zeros((1, W))])
    diff_img += np.vstack([np.zeros((1, W)),
                       np.diff(int_img_filt, axis=0) / 2])

    # holes fill 후 diff=0
    ilm_mask_filled = binary_fill_holes(ilm_mask)
    diff_img[ilm_mask_filled.T] = 0.0 

    # 아래쪽 (NFLmaxRange 이후) 컷
    nfl_max_rng = int(info["NFLmaxRange"])

    for x in range(W):
        if np.isnan(ILM[x]):
            continue
        
        z_cut_incl = int(np.floor(ILM[x]-1 + nfl_max_rng))
        if z_cut_incl < L - 1:
            diff_img[z_cut_incl + 1 :, x] = 0.0

    # ---------------- ⑤ 시각화(옵션) -------------------------------
    if show:
        plt.figure(103, figsize=(10, 4)); plt.clf()
        plt.subplot(1, 2, 1)
        plt.title("intImgFiltNorm"); plt.imshow(int_img_norm, cmap="gray",
                                                vmin=0, vmax=50, origin="upper")
        plt.subplot(1, 2, 2)
        plt.title("diffImg"); plt.imshow(diff_img, cmap="gray",
                                         vmin=-1, vmax=1, origin="upper")

    # ---------------- ⑥ NFL mask 선택 ------------------------------
    nfl_mask = np.zeros_like(diff_img, bool)
    neg_region = diff_img < -1
    if neg_region.any():
        lbl, n = label(neg_region, connectivity=2, return_num=True)
        if n:
            # MATLAB: sum(diffImg(region)) 최소 → 가장 음수값 큰 CC
            sums = np.array([diff_img[lbl == i + 1].sum() for i in range(n)])
            idx = sums.argmin()
            nfl_mask[lbl == idx + 1] = True

    # ---------------- ⑦ NFL edge -----------------------------------
    nfl_edge = find_edge_layer(nfl_mask, which="last", extrapolate=False)
    nfl_min_rng = info["NFLminRange"]

    NFL = np.maximum(ILM + nfl_min_rng, nfl_edge, where=~np.isnan(nfl_edge))
    # fallback : where nfl_edge is nan
    NFL[np.isnan(NFL)] = ILM[np.isnan(NFL)] + nfl_min_rng

    # blend with weight
    ok = ~np.isnan(NFL)
    NFL[ok] = (ILM[ok] + nfl_min_rng) * (1 - weight[ok]) + NFL[ok] * weight[ok]

    # 컬럼 전체가 NaN 인 경우 propagate NaN
    bad_col = np.isnan(int_img_filt).any(axis=0)
    ILM[bad_col] = np.nan
    NFL[bad_col] = np.nan

    # ---------------- ⑧ 최종 plot ----------------------------------
    if show:
        plt.subplot(1, 2, 1); plt.plot(ILM, "r"); plt.plot(NFL, "c")
        plt.subplot(1, 2, 2); plt.plot(ILM, "r"); plt.plot(NFL, "c")
        plt.pause(0.01)

    return ILM, NFL    

def segment_nfl_3d(
        int_img_filt: np.ndarray,     # (L, W, H)  필터링 intensity
        info: dict,
        *,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    int_img_filt : (L, W, H) ndarray
        가우시안 등으로 미리 필터링된 3-D intensity 볼륨
    info : dict
        MATLAB Info 구조체를 그대로 옮긴 파라미터 딕셔너리
    verbose : bool, optional
        진행 표시 출력 여부 (10 프레임/라인마다)

    Returns
    -------
    ILM, NFL : (W, H) ndarray
        각 B-scan(line)·프레임 좌표계에서의 ILM / NFL depth 인덱스
        (NaN = 미검출)
    """
    nFrames = info["numImgFrames"]
    # -------------- 배열 준비 -----------------
    if int_img_filt.shape[1] == nFrames:
        int_img_filt = int_img_filt.transpose(0, 2, 1)

    _, W, H = int_img_filt.shape
    ILM_X = np.full((W, H), np.nan, dtype=np.float32)
    ILM_Y = np.full((W, H), np.nan, dtype=np.float32)
    NFL_X = np.full((W, H), np.nan, dtype=np.float32)
    NFL_Y = np.full((W, H), np.nan, dtype=np.float32)

    # -------------- B-scan 방향 ----------------
    msg = ""
    for frame in range(H):
        if verbose and frame % 10 == 0:
            print("\b" * len(msg), end="", flush=True)
            msg = f"Segmenting NFL frame {frame+1}/{H}..."
            print(msg, end="\n", flush=True)

        ilm, nfl = segment_nfl_2d(int_img_filt[:, :, frame], info, show_fig=False)
        ILM_X[:, frame] = ilm
        NFL_X[:, frame] = nfl

    if verbose:
        print("\b" * len(msg), end="", flush=True)

    # -------------- 스캔 라인(X) 방향 -----------
    msg = ""
    for line in range(W):
        if verbose and line % 10 == 0:
            print("\b" * len(msg), end="", flush=True)
            msg = f"Segmenting NFL line {line+1}/{W}..."
            print(msg, end="\n", flush=True)

        # (L, 1, H) → (L, H)
        ilm, nfl = segment_nfl_2d(int_img_filt[:, line, :], info, show_fig=False)
        ILM_Y[line, :] = ilm          # 열 방향으로 저장
        NFL_Y[line, :] = nfl

    if verbose:
        print("\b" * len(msg))

    # -------------- 양방향 결과 병합 ------------
    ILM = _nanmin_pair(ILM_X, ILM_Y)  # element-wise min, NaN 무시
    NFL = _nanmax_pair(NFL_X, NFL_Y)  # element-wise max, NaN 무시
    return ILM, NFL    


def prl_norm_and_filter(
    int_img: torch.Tensor,         # (L,W,H)  raw intensity (float32)
    int_img_avg: Union[torch.Tensor, np.ndarray],     # (L,)     axial 평균 (이미 계산)
    sidx: int,
    eidx: int,
    info: dict,
    sub_onh_mask: np.ndarray | None,
    device: torch.device = "cpu",
) -> np.ndarray:
    """
    MATLAB *segmentPRLNFL* 의 ▸ 정규화(normFactor) + 3-D Gaussian 필터 부분
    -------------------------------------------------------------------
    반환값
        int_img_filt : np.ndarray, shape (L,W,H)
    """
    L, W, H = int_img.shape
    depth_per_pix = info["depthPerPixel"]

    # --- PRL slab 범위 -------------------------------------------------

    pad = int(np.ceil(0.1 / depth_per_pix))

    prl_slab_start = max(0, sidx - pad)
    prl_slab_end   = min(L - 1, eidx + pad)
    

    # --- 정규화 인자 ---------------------------------------------------
    int_img_cpu = int_img.cpu().numpy()          # 후처리는 CPU-numpy 로 처리
    prl_part = int_img_cpu[prl_slab_start:prl_slab_end + 1] - 30.0
    prl_part[prl_part < 0] = 0                  # MATLAB 코드상 음수 잘라내기 X 이지만 safety

    if isinstance(int_img_avg, torch.Tensor):
        int_img_avg_np = int_img_avg.cpu().numpy()
    else:
        int_img_avg_np = int_img_avg

    int_img_avg_sum = np.sum(int_img_avg_np[prl_slab_start:prl_slab_end + 1] - 30.0)

    # X-, Y-평균 (Σ_z I)  → shape (W,1) & (1,H)
    depth_sum = np.nansum(prl_part, axis=0) 
    x_avg_sum = np.nanmean(depth_sum, axis=0)      # (H,)  frame-wise
    y_avg_sum = np.nanmean(depth_sum, axis=1)      # (W,)  line-wise

    # NaN 보간 (linear-nearest)  – scipy 대신 numpy 기본으로 충분
    if np.isnan(x_avg_sum).any():
        idx = np.flatnonzero(~np.isnan(x_avg_sum))
        x_avg_sum = np.interp(np.arange(x_avg_sum.size), idx, x_avg_sum[idx])

    if np.isnan(y_avg_sum).any():
        idx = np.flatnonzero(~np.isnan(y_avg_sum))
        y_avg_sum = np.interp(np.arange(y_avg_sum.size), idx, y_avg_sum[idx])
    
    norm_factor = (
    4000.0 * int_img_avg_sum /
    (y_avg_sum[:, None] * x_avg_sum[None, :])   # (W, H) ← 브로드캐스트
    )
    
    # --- intensity 정규화 ---------------------------------------------
    int_img_cpu  = int_img.cpu().numpy()
    int_img_norm = (int_img_cpu - 30.0) * norm_factor[None, :, :]

    # --- 3-D Gaussian filtering ---------------------------------------
    sigma_z  = info["sigmaZ"]
    sigmas   = (sigma_z, 3*info["sigmaX"], 3*info["sigmaY"])

    int_img_filt = ndi.gaussian_filter(int_img_norm, sigma=sigmas, mode="nearest", truncate=2.0)
    # int_img_filt[int_img_filt < 0] = 0.0

    #DEBUG
    p99 = np.nanpercentile(int_img_filt, 99)
    print(f"[PRL-norm] p50={np.nanpercentile(int_img_filt,50):4.1f}"
        f"  p90={np.nanpercentile(int_img_filt,90):4.1f}"
        f"  p99={p99:4.1f}")


    # --- ONH 영역 NaN 마스킹 -------------------------------
    if info.get("excludeONH", False) and (sub_onh_mask is not None):
        mask3d = np.broadcast_to(sub_onh_mask[None, :, :], int_img_filt.shape)
        int_img_filt[1:][mask3d[1:]] = np.nan

    return int_img_filt.astype(np.float32), prl_slab_start, prl_slab_end, int_img_norm.astype(np.float32)
    
def segment_prl_dbg(
    int_img_filt: np.ndarray,     # (L,W,H) 3-D Gaussian-filtered volume
    int_img_raw:  np.ndarray,     # (L,W,H) unclipped raw intensity
    info:         dict,
    *,
    verbose: bool = True,
    debug:   bool = False,        # ★ 새 플래그
):
    """
    MATLAB  segmentPRL  중 tilt removal + PRL-based ISOS/RPE 검출 파트를
    그대로 포팅한 함수에 **디버그 계측/로그** 를 추가한 버전.
    """
    import time
    from textwrap import indent
    tic_total = time.time()

    # ---------- 0. 기본 정보 ----------
    L, W, H = int_img_filt.shape
    if debug:
        print(f"[DBG-0] volume = ({L=}, {W=}, {H=})  "
              f"sigmaZ={info['sigmaZ']}, σx={info['sigmaX']}, σy={info['sigmaY']}")

    # ---------- 1. valid A-scan 마스크 ----------
    valid = ~np.isnan(int_img_filt).any(axis=0)        # (W,H)
    n_valid = valid.sum()
    if debug:
        ratio = 100 * n_valid / (W * H)
        print(f"[DBG-1] valid A-scans : {n_valid}/{W*H}  ({ratio:.1f} %)")

    # ---------- 2. depth-FFT ----------
    ft = torch.fft.fft(torch.as_tensor(int_img_filt, dtype=torch.complex64), dim=0)

    cum_shift = torch.zeros((W, H), dtype=torch.float32, device=ft.device)
    if debug:
        print("[DBG-2] -- tilt-alignment passes (sub-pixel)")

    # ---------- 3. 두 번의 정렬 pass ----------
    for ii in range(2):
        t0 = time.time()
        # pass-1 : frame 기준
        for frm in range(H):
            ref_ft = _complex_nanmean(ft[:, :, frm], dim=1)
            for col in range(W):
                _, _, sh = dftregistration1d(ref_ft, ft[:, col, frm], usfac=4)
                cum_shift[col, frm] += sh              # 부호 그대로
                ft[:, col, frm] = _apply_subpixel_shift(ft[:, col, frm], sh)
        # pass-2 : line 기준
        for col in range(W):
            ref_ft = _complex_nanmean(ft[:, col, :], dim=1)
            for frm in range(H):
                _, _, sh = dftregistration1d(ref_ft, ft[:, col, frm], usfac=4)
                sh = -sh                                # ← MATLAB 과 동일하게 부호 뒤집기
                cum_shift[col, frm] += sh
                ft[:, col, frm] = _apply_subpixel_shift(ft[:, col, frm], sh)
        if debug:
            dt = time.time() - t0
            m, M = cum_shift.min().item(), cum_shift.max().item()
            print(f"  pass-{ii+1} done in {dt:5.1f}s   "
                  f"shift range = [{m:+.2f}, {M:+.2f}] px")

    # ---------- 4. cumShift 결손 보간 ----------
    cum_shift_np = cum_shift.cpu().numpy()
    if (~valid).any():
        good_pts = np.column_stack(np.nonzero(valid))          # (N,2)  (row,col) == (w,h)
        bad_pts  = np.column_stack(np.nonzero(~valid))         # (M,2)

        # meshgrid → X=row(col),Y=frame(h)
        filled = griddata(good_pts, cum_shift_np[valid],
                          bad_pts, method="linear")

        nan_mask = np.isnan(filled)
        
        if nan_mask.any():
            filled[nan_mask] = griddata(good,
                                        cum_shift_np[valid],
                                        bad[nan_mask],
                                        method="nearest")
        cum_shift_np[~valid] = filled

    if debug:
        print(f"[DBG-3] cumShift  mean={cum_shift_np.mean():+.3f}  "
              f"std={cum_shift_np.std():.3f}")
        print("min = ", cum_shift_np.min(), "max = ", cum_shift_np.max())


    # ---------- 5. raw 재-shifting & 3-D Gaussian ----------
    
    int_img_filt_aligned = np.maximum(
        0.0, np.fft.ifft(ft.detach().cpu().numpy(), axis=0).real
    ).astype(np.float32)

    int_base = np.minimum(int_img_raw, 200.0).astype(np.float64)   # clip 200 동일

    k = np.fft.ifftshift((np.arange(np.ceil(-L/2), np.ceil(L/2)) / L).astype(np.float64))
    phase = np.exp(-1j * 2 * np.pi * k[:, None, None] * cum_shift_np[None, :, :])

    int_img_filt_flat = np.fft.ifft(np.fft.fft(int_base, axis=0) * phase, axis=0).real

    sigma3d = [info["sigmaZ"]/4.0, 6*info["sigmaX"], 6*info["sigmaY"]]  # ★ 6배
    int_img_filt_flat = ndi.gaussian_filter(
        int_img_filt_flat, sigma=sigma3d, mode="nearest", truncate=2.0
    )

    too_dim = (int_img_filt_flat <= 50).all(axis=0)
    valid   &= ~too_dim                          # valid = valid & (…>50)

    L, W, H = int_img_filt_aligned.shape
    bad = ~valid                                      # (W,H)
    # --- 방법 A: 평탄화 후 마스크 적용 ---
    flt = int_img_filt_aligned.reshape(L, -1)
    flt[:, bad.ravel()] = np.nan
    int_img_filt_aligned = flt.reshape(L, W, H)

    flt = int_img_filt_flat.reshape(L, -1)
    flt[:, bad.ravel()] = np.nan
    int_img_filt_flat = flt.reshape(L, W, H)
                         

    # ---------- 6. PRL 범위 산정 ----------
    valid_cnt = np.count_nonzero(valid)         # nnz(valid) 와 동일
    int_img_sum = np.nansum(np.nansum(int_img_filt_flat, axis=2), axis=1)  # (L,)
    int_img_avg = int_img_sum / valid_cnt
    
    thr = 25
    thr_idx = np.where(int_img_avg > thr)[0]

    if thr_idx.size == 0:
        raise RuntimeError("PRL 범위를 찾지 못했습니다 – 정규화/스케일 확인")

    prl_start = max(0,                thr_idx[0] - int(round(3*info["sigmaZ"])))
    prl_end   = min(L - 1,            thr_idx[-1] + int(round(3*info["sigmaZ"])))
    int_img_filt_flat_crop = int_img_filt_flat[prl_start:prl_end+1]

    if debug:
        print(f"[DBG-4] PRL-crop  z ∈ [{prl_start}, {prl_end}] "
              f"(len = {prl_end-prl_start+1})")

    if debug:
        frm_dbg = H // 2            # 관심 프레임, 마음대로 바꿔도 됨
        _prl_debug_figs(
            frm_dbg,
            int_img_avg,
            prl_start, prl_end,
            np.nanmean(_central_diff(int_img_filt_flat[prl_start:prl_end+1])[:,:,frm_dbg], axis=1),
            cum_shift_np,
            valid,
            title_prefix="PRL-debug"
        )

    # ---------- 7. central-difference ----------
    crop_filt      = int_img_filt_aligned[prl_start:prl_end+1]
    crop_filt_flat = int_img_filt_flat   [prl_start:prl_end+1]

    diff_img      = _central_diff(crop_filt)         # ▶︎ 원본 대비한 edge 강도
    diff_img_flat = _central_diff(crop_filt_flat)

    s = diff_img_flat[...].ravel()
    
    if debug:
        Lc = crop_filt.shape[0]
        print(f"[DBG-5] PRL-crop  z ∈ [{prl_start}, {prl_end}]  (len = {Lc})")
        print(f"[DBG-6] shapes  crop_filt={crop_filt.shape}  crop_filt_flat={crop_filt_flat.shape}  "
            f"diff_img={diff_img.shape}  diff_img_flat={diff_img_flat.shape}")

        def _stats(tag, a):
            a = a.astype(np.float64, copy=False)
            mn, mx, md = np.nanmin(a), np.nanmax(a), np.nanmedian(a)
            print(f"[DBG-6a] {tag:<16} min={mn: .3f}  med={md: .3f}  max={mx: .3f}")

        _stats("intImgFilt",      crop_filt)
        _stats("intImgFiltFlat",  crop_filt_flat)
        _stats("diffImg",         diff_img)
        _stats("diffImgFlat",     diff_img_flat)

    # shift-에 따라 위/아래 null-out
    for w in range(W):
        for h in range(H):
            if not valid[w, h]:
                continue
            s = int(round(cum_shift_np[w, h]))
            if s < 0:
                diff_img[s:, w, h]      = 0
                diff_img_flat[s:, w, h] = 0
            elif s > 0:
                diff_img[:s+1, w, h]      = 0
                diff_img_flat[:s+1, w, h] = 0

    if debug:
        sh = np.round(cum_shift_np).astype(int)
        print(f"[DBG-7] zeroing applied  shift range = [{sh.min():+d}, {sh.max():+d}] px")

        # 중앙 프레임의 평균 프로파일 간단 확인(탐색 윈도우 판단용)
        frm_dbg = H // 2
        dfa = np.nanmean(diff_img_flat[:, :, frm_dbg], axis=1)
        print(f"[DBG-8] frm {frm_dbg}: diff_flat_avg  min={np.nanmin(dfa): .2f}  "
            f"max={np.nanmax(dfa): .2f}")                

    # ---------- 8. 레이어 탐색 ----------
    ISOS = np.full((W, H), np.nan, np.float32)
    RPE  = np.full((W, H), np.nan, np.float32)

    two_sig  = int(round(2*info["sigmaZ"]))
    four_sig = int(round(4*info["sigmaZ"]))
    L_crop   = diff_img.shape[0]

    hit = 0
    for frm in range(H):
        cols_valid = np.flatnonzero(valid[:, frm])
        if cols_valid.size == 0:
            continue

        diff_flat_avg = np.nanmean(diff_img_flat[:, :, frm], axis=1)

        if debug and frm % 50 == 0:
            print(f"[DBG] frm {frm:4d}: "
                  f"min={diff_flat_avg.min(): .2f}, "
                  f"max={diff_flat_avg.max(): .2f}")        

        neg_idx = np.flatnonzero(diff_flat_avg < -3)
        pos_idx = np.flatnonzero(diff_flat_avg >  +3)
        if neg_idx.size == 0 or pos_idx.size == 0:
            continue

        if debug and (neg_idx.size == 0 or pos_idx.size == 0):
            print(f"[WARN] frame {frm}: "
                  f"neg={neg_idx.size}, pos={pos_idx.size}  → skip")
                          
        rpe_last   = int(neg_idx[-1])       # last < -3
        isos_first = int(pos_idx[0]) 

        rpe_s = max(0, rpe_last - two_sig)
        rpe_e = min(L_crop - 1, rpe_last + two_sig)
        iso_e = min(L_crop - 1, isos_first + four_sig)

        cols       = np.where(valid[:, frm])[0]
        rpe_layer = find_max_layer(np.maximum(0, -diff_img_flat[rpe_s:rpe_e+1, cols_valid, frm]))
        iso_layer = find_max_layer(np.maximum(0,  diff_img_flat[0:iso_e+1,  cols_valid, frm]))
        
        RPE[cols_valid, frm]  = rpe_layer + rpe_s                 # + (PRLstart-1)은 루프 밖에서
        ISOS[cols_valid, frm] = iso_layer - info["sigmaZ"]

        if cols_valid.size < W:
            bad = np.flatnonzero(~valid[:, frm])
            # 선형 보간, 바깥은 최근접으로 채움
            RPE[bad,  frm] = np.interp(bad, cols_valid, RPE[cols_valid,  frm])
            ISOS[bad, frm] = np.interp(bad, cols_valid, ISOS[cols_valid, frm])
        
        hit += len(cols)

    if debug:
        print(f"[DBG-5] frames processed ≈ {hit}/{W*H} columns with valid hits")

    # cumShift 보정 (1-based → 0-based 오프셋 동일)
    ISOS = ISOS - cum_shift_np + prl_start
    RPE  = RPE  - cum_shift_np + prl_start

    if debug:
        toc = time.time() - tic_total
        print(f"[DBG-END] segment_prl_dbg finished in {toc:.1f} s")

    return ISOS.astype(np.float32), RPE.astype(np.float32)

def _prl_debug_figs(frame_idx: int,
                    int_img_avg: np.ndarray,
                    prl_start: int,
                    prl_end: int,
                    diff_flat_avg: np.ndarray,
                    cum_shift_np: np.ndarray,
                    valid: np.ndarray,
                    title_prefix: str = "") -> None:
    """
    한 번 호출에 4개 서브플롯을 그려준다.
        ① axial 평균 + PRL 슬랩
        ② diff_flat_avg (한 프레임)
        ③ cumShift 히트맵
        ④ valid mask
    """
    fig = plt.figure(figsize=(11, 8))
    fig.suptitle(f"{title_prefix} ‖ frame {frame_idx}")

    # ① axial 평균
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("int_img_avg (전체 Z)")
    ax1.plot(int_img_avg, lw=1)
    ax1.axvspan(prl_start, prl_end, color="lime", alpha=0.2,
                label=f"PRL [{prl_start}:{prl_end}]")
    ax1.legend(fontsize=8)

    # ② diff_flat_avg (프레임 하나)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("diff_flat_avg")
    ax2.plot(diff_flat_avg, lw=1)
    ax2.axhline(+3, ls="--", c="tab:blue"); ax2.axhline(-3, ls="--", c="tab:blue")

    # ③ cumShift heat-map
    ax3 = plt.subplot(2, 2, 3)
    im = ax3.imshow(cum_shift_np.T, cmap="seismic", aspect="auto",
                    vmin=-np.nanmax(np.abs(cum_shift_np)),
                    vmax= np.nanmax(np.abs(cum_shift_np)))
    ax3.set_title("cumShift  (col=X, row=frame)")
    plt.colorbar(im, ax=ax3, shrink=.7)

    # ④ valid mask
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title("valid A-scan mask")
    ax4.imshow(valid.T, cmap="gray", aspect="auto")

    plt.tight_layout()
    plt.pause(0.01)    


def plot_prl_slab(int_img_avg: torch.Tensor,
                  sidx: int,
                  eidx: int,
                  th_val: float,
                  *, title: str = "Average Axial Profile",
                  ax=None):
    """
    int_img_avg : (L,) torch.Tensor  –  nanmean (depth-wise) 값
    sidx, eidx  : 슬라브 시작·끝 인덱스  (0-based, eidx 포함)
    th_val      : threshold 값 (float)
    ax          : 미리 만든 matplotlib axis (optional)

    반환값 : axis (편집용)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(int_img_avg.numel())
    y = int_img_avg.detach().cpu().numpy()

    # 전체 프로파일 (빨간색)
    ax.plot(x, y, "r-", lw=1.0, label="Axial mean")

    # 슬라브 부분 (초록색, 두껍게)
    ax.plot(x[sidx:eidx + 1], y[sidx:eidx + 1],
            color="limegreen", lw=2.0, label="PRL slab")

    # 임계값 가로선 (검은색)
    ax.axhline(th_val, color="k", lw=1.0, label="threshold")

    ax.set_xlim(0, len(x) - 1)
    ax.set_xlabel("Depth (pixel)")
    ax.set_ylabel("Mean intensity")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return ax    

