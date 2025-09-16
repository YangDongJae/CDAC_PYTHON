# segment/fastILMISOS.py

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import convolve1d

# ---------- 공통 유틸 (MATLAB 동일 동작) ----------
def mround(x):
    """MATLAB round: half away from zero (vectorized)."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def _gauss_kernel_1d(sigma):
    """imgaussfilt3의 1D 가우시안 커널: 길이 = 2*ceil(2*sigma)+1, 정규화."""
    sigma = float(sigma)
    if sigma < np.finfo(float).eps:
        return np.array([1.0], dtype=np.float64)
    radius = int(np.ceil(2.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k

def _imgaussfilt3_exact(vol, sigma3):
    """
    MATLAB imgaussfilt3 복제:
      - 분리 가우시안(1D) 3회
      - 경계: replicate ('nearest')
      - 커널 길이: 2*ceil(2*sigma)+1
    vol: (L,W,H) float64
    sigma3: (σz, σx, σy)
    """
    zsig, xsig, ysig = map(float, sigma3)
    out = vol
    for axis, sig in enumerate((zsig, xsig, ysig)):
        k = _gauss_kernel_1d(sig)
        # convolve1d는 same-shape, mode='nearest'가 replicate에 해당
        out = convolve1d(out, k, axis=axis, mode="nearest")
    return out

def _fill_missing_smooth(Z, miss):
    """
    MATLAB scatteredInterpolant(x,y,z,'natural','linear') 유사 재현:
    1차: LinearNDInterpolator (natural과 유사한 부드러움)
    2차: Nearest로 남은 NaN 보정
    """
    Z = Z.copy()
    Z[np.isinf(Z)] = np.nan
    W, H = Z.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
    known = ~miss & ~np.isnan(Z)
    if not np.any(known):
        return np.zeros_like(Z, dtype=np.float64)
    pts = np.column_stack((x[known].ravel(), y[known].ravel()))
    vals = Z[known].ravel()
    lin = LinearNDInterpolator(pts, vals, fill_value=np.nan)
    Z_lin = lin(x, y)
    nan_mask = np.isnan(Z_lin)
    if np.any(nan_mask):
        nei = NearestNDInterpolator(pts, vals)
        Z_lin[nan_mask] = nei(x, y)[nan_mask]
    Z[miss] = Z_lin[miss]
    return Z

def _subpixel_edge_crossing(Z, V, thr, sense="up", xy_mask=None):
    """
    MATLAB subpixelEdgeCrossing을 1:1로 재현.
    Z : (W,H) 1-based 실수 인덱스(또는 Inf)
    V : (L,W,H) 데이터 (원본 또는 z미분) — Z는 V의 'z' 좌표계에 맞음
    thr: 임계
    sense: 'up' (v1<thr, v2>=thr) 또는 'down' (v1>=thr, v2<thr)
    xy_mask: (W,H) True 위치만 보정
    """
    Z = Z.copy()
    W, H = Z.shape
    valid = np.isfinite(Z)
    if xy_mask is not None:
        valid &= xy_mask
    if not np.any(valid):
        return Z

    # 1-based → 0-based 인덱싱 (z-1 접근하므로 최소 1)
    z_int = np.floor(Z[valid]).astype(int) - 1
    z_int = np.clip(z_int, 0, V.shape[0] - 2)

    w_idx, h_idx = np.where(valid)
    v1 = V[z_int, w_idx, h_idx]
    v2 = V[z_int + 1, w_idx, h_idx]

    if sense.lower() == "up":
        orient_ok = (v1 < thr) & (v2 >= thr)
    elif sense.lower() == "down":
        orient_ok = (v1 >= thr) & (v2 < thr)
    else:
        orient_ok = np.ones_like(v1, dtype=bool)

    denom = (v2 - v1).astype(np.float64)
    denom[np.abs(denom) < 1e-12] = 1e-12
    corr = (thr - v1) / denom  # 0<=corr<1만 유효
    in_unit = (corr >= 0.0) & (corr < 1.0)

    ok = orient_ok & in_unit
    if np.any(ok):
        subpix = (z_int + 1).astype(np.float64) + corr
        Z_sel = Z[valid]
        Z_sel[ok] = subpix[ok]
        Z[valid] = Z_sel
    return Z

# ---------- fastILMISOS: MATLAB 1:1 포팅 ----------
def fastILMISOS_np(intImg, sigma, thrILM=5.0, thrNFL=15.0, thrISOS=0.015):
    """
    NumPy 참조 구현 (입력/출력 모두 NumPy, dtype=float64, 1-based 인덱스 반환)
    intImg: (L,W,H)
    sigma : 스칼라 또는 (σz,σx,σy)
    """
    intImg = np.asarray(intImg, dtype=np.float64)
    L, W, H = intImg.shape
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if sigma.size == 1:
        sigma = np.repeat(sigma, 3)
    sigZ, sigX, sigY = sigma

    # 1) Denoise (imgaussfilt3 동일)
    img = _imgaussfilt3_exact(intImg, (sigZ, sigX, sigY))

    # 2) ILM: mask & 첫 z (1-based), subpixel 보정
    Z1 = np.arange(1, L + 1, dtype=np.float64).reshape(L, 1, 1)   # 1..L
    maskILM = img > thrILM
    z1 = np.where(maskILM, Z1, np.inf)
    ILM = np.min(z1, axis=0).squeeze()                             # (W,H)
    ILM = _subpixel_edge_crossing(ILM, img, thrILM, sense="up")

    # 3) NFL: 창 내 첫 ≥thr, 그 뒤 첫 <thr, 폴백, subpixel(down)
    dzNFL = int(max(1, mround(3 * sigZ)))
    ILM_i = np.floor(ILM).astype(int)
    ILM_i[~np.isfinite(ILM_i)] = 1
    zStart = np.maximum(2, ILM_i)                 # z-1 써야 하므로 최소 2
    zEnd   = np.minimum(L, ILM_i + dzNFL)

    Z = np.arange(1, L + 1, dtype=np.float64).reshape(L, 1, 1)
    searchNFL = (Z >= zStart.reshape(1, W, H)) & (Z <= zEnd.reshape(1, W, H))

    aboveThr = (img >= thrNFL) & searchNFL
    zMaskA = np.where(aboveThr, Z, np.inf)
    zFirstAbove = np.min(zMaskA, axis=0).squeeze()                 # (W,H)

    afterAbove = (Z >= (zFirstAbove + 1).reshape(1, W, H))
    belowThr   = (img < thrNFL) & afterAbove
    zMaskB = np.where(belowThr, Z, np.inf)
    zFirstBelow = np.min(zMaskB, axis=0).squeeze()                 # (W,H)

    NFL = zFirstBelow.copy()
    NFLexists = np.isfinite(zFirstAbove) & np.isfinite(zFirstBelow)

    NFL[~NFLexists] = ILM[~NFLexists] + sigZ
    NFL = _subpixel_edge_crossing(NFL, img, thrNFL, sense="down", xy_mask=NFLexists)

    # 4) ISOS: NFL+6σ 아래에서 A-scan 정규화 → z미분 → 첫 +edge > thr
    NFLrep = NFL.reshape(1, W, H)  # (1,W,H), 1-based
    searchMask3 = (np.arange(1, L, dtype=np.float64).reshape(L - 1, 1, 1) >= (NFLrep + 6 * sigZ))

    # 각 A-scan의 최대 (masking: L×W×H에서 L-1×W×H를 쓰진 않음; 원본에서 mask)
    mask_full = (np.arange(1, L + 1, dtype=np.float64).reshape(L, 1, 1) >= (NFLrep + 6 * sigZ))
    masked_vals = np.where(mask_full, img, -np.inf)
    maxA = np.max(masked_vals, axis=0, keepdims=True)  # (1,W,H)
    maxA[~np.isfinite(maxA)] = 0.0
    maxA[maxA == 0.0] = np.finfo(float).eps
    img3 = img / maxA  # implicit expansion

    zgrad3 = np.diff(img3, axis=0)  # (L-1,W,H)
    maskISOS = (zgrad3 > thrISOS) & searchMask3

    Z3 = np.arange(1, L, dtype=np.float64).reshape(L - 1, 1, 1)  # 1..L-1
    z3 = np.where(maskISOS, Z3, np.inf)
    ISOS = np.min(z3, axis=0).squeeze()  # (W,H)
    ISOS = _subpixel_edge_crossing(ISOS, zgrad3, thrISOS, sense="up")

    badISOS = ~np.isfinite(ISOS)
    if np.any(badISOS):
        ISOS = _fill_missing_smooth(ISOS, badISOS)

    # 5) small bias (+round(sigma(1)))
    bias = int(mround(sigZ))
    ILM  = ILM  + bias
    NFL  = NFL  + bias
    ISOS = ISOS + bias

    return ILM, NFL, ISOS  # 모두 1-based float64, shape (W,H)

# ---------- (선택) PyTorch 래퍼 ----------
def fastILMISOS(intImg, sigma, thrILM=5.0, thrNFL=15.0, thrISOS=0.015):
    """
    입력이 torch.Tensor라도 받아서 NumPy 참조 구현으로 돌리고,
    결과를 torch.double 텐서(1-based)로 반환.
    """
    try:
        import torch
        is_torch = isinstance(intImg, torch.Tensor)
    except Exception:
        is_torch = False

    if is_torch:
        dev = intImg.device
        np_in = intImg.detach().cpu().numpy().astype(np.float64)
        ILM, NFL, ISOS = fastILMISOS_np(np_in, sigma, thrILM, thrNFL, thrISOS)
        return (torch.from_numpy(ILM).to(dev, dtype=torch.float64),
                torch.from_numpy(NFL).to(dev, dtype=torch.float64),
                torch.from_numpy(ISOS).to(dev, dtype=torch.float64))
    else:
        return fastILMISOS_np(intImg, sigma, thrILM, thrNFL, thrISOS)