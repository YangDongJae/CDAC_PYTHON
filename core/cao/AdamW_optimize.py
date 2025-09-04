# /core/cao/adamw_cao.py
from __future__ import annotations
import math
from typing import Callable, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from utils.utils import fft2c, ifft2c, _center_crop



# ──────────────────────────────────────────────────────────────────────────────
# Objectives (배치 안전)
# ──────────────────────────────────────────────────────────────────────────────


def _resize_fd_tensor(x: torch.Tensor, out_hw: tuple[int,int]) -> torch.Tensor:
    """FD 격자에서의 (H,W,[K]) 텐서를 out_hw로 보간 (bicubic, align_corners=False)."""
    Ht, Wt = out_hw
    if x.dim() == 3:
        # (H,W,K) -> (1,K,H,W) -> 보간 -> (H,W,K)
        x4d = x.permute(2,0,1).unsqueeze(0).to(torch.float32)
        xr  = F.interpolate(x4d, size=(Ht, Wt), mode="bicubic", align_corners=False)
        return xr.squeeze(0).permute(1,2,0).contiguous()
    elif x.dim() == 2:
        # (H,W) -> (1,1,H,W) -> 보간 -> (H,W)
        x4d = x.unsqueeze(0).unsqueeze(0).to(torch.float32)
        xr  = F.interpolate(x4d, size=(Ht, Wt), mode="bicubic", align_corners=False)
        return xr.squeeze(0).squeeze(0).contiguous()
    else:
        raise ValueError("Expected (H,W) or (H,W,K) tensor.")


def _remove_linear_ramp_fd(phase_fd: torch.Tensor) -> torch.Tensor:
    H, W = phase_fd.shape[-2], phase_fd.shape[-1]
    yy = torch.linspace(-1, 1, H, device=phase_fd.device)
    xx = torch.linspace(-1, 1, W, device=phase_fd.device)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    A = torch.stack([X.reshape(-1), Y.reshape(-1), torch.ones(H*W, device=phase_fd.device)], dim=1)
    b = phase_fd.reshape(-1, 1).to(torch.float32)
    coef = torch.linalg.lstsq(A, b).solution.squeeze(1)   # [ax, ay, c]
    ax, ay = coef[0], coef[1]
    return (phase_fd - (ax*X + ay*Y)).to(phase_fd.dtype)


def _phase_fd_to_fullimg(phase_fd: torch.Tensor, out_hw: tuple[int,int]) -> torch.Tensor:
    Him, Wim = out_hw
    # (Hfd,Wfd) -> (1,1,Hfd,Wfd) 보간 -> (Him,Wim)
    phase_img = F.interpolate(
        phase_fd.unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=(Him, Wim), mode="bicubic", align_corners=False
    ).squeeze(0).squeeze(0)                    # (Him,Wim) float32
    # FD좌표 -> 영상좌표로 정렬
    phase_img = torch.fft.ifftshift(phase_img)
    return phase_img


@torch.no_grad()
def entropy_shannon_anyshape(xc: Tensor) -> Tensor:
    """Shannon entropy over the last two dims (H,W) of a complex tensor.
    xc: complex tensor of shape (..., H, W)
    return: real tensor of shape (...,)
    """
    I = (xc.real**2 + xc.imag**2).clamp_min(0)
    # Normalize over spatial dims only
    denom = I.flatten(start_dim=-2).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = I / denom.unsqueeze(-1)
    return -(p * (p + 1e-12).log()).sum(dim=(-2, -1))

@torch.no_grad()
def otf_gain_log(
    xc: Tensor,                      # (...,H,W) complex image (crop된 en-face)
    hf_band=(0.40, 0.90),
    radial_bins=192,
    gamma=1.0,
    roi_mask: Tensor | None = None,  # (H,W) float/bool, 가중 ROI(예: ISOS 근방 해닝창)
) -> Tensor:
    # 진폭
    I = (xc.real**2 + xc.imag**2).clamp_min(0).sqrt()
    if roi_mask is not None:
        I = I * roi_mask  # ROI 가중
    # OTF (실제로는 MTF)
    F = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(I, dim=(-2,-1))), dim=(-2,-1))
    M = F.abs()
    H, W = I.shape[-2:]
    yy = torch.linspace(-1, 1, H, device=I.device)
    xx = torch.linspace(-1, 1, W, device=I.device)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    R = torch.sqrt(X*X + Y*Y).clamp_(0, 1)
    r0, r1 = hf_band
    hf = (R >= r0) & (R <= r1)
    # 고주파 에너지의 로그 합 (스케일 안정화)
    eps = 1e-12
    score = torch.log(M[..., hf].mean(dim=-1) + eps) * gamma
    return score    


@torch.no_grad()
def entropy_shannon_anyshape(xc: Tensor) -> Tensor:
    I = (xc.real**2 + xc.imag**2).clamp_min(0)
    denom = I.flatten(start_dim=-2).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = I / denom.unsqueeze(-1)
    return -(p * (p + 1e-12).log()).sum(dim=(-2, -1))

@torch.no_grad()
def eig_value_anyshape(xc: Tensor) -> Tensor:
    HA = torch.abs(xc).to(torch.float32)
    if HA.ndim < 2:
        raise ValueError("Input must have at least 2 spatial dims (H,W)")
    B = int(torch.prod(torch.tensor(HA.shape[:-2])).item()) if HA.ndim > 2 else 1
    H, W = HA.shape[-2], HA.shape[-1]
    HA2 = HA.reshape(B, H, W)
    denom = torch.sqrt((HA2**2).sum(dim=(1, 2))).clamp_min(1e-20)
    HA2 = HA2 / denom.view(B, 1, 1)
    HA2 = HA2 - HA2.mean(dim=(1, 2), keepdim=True)
    eig_AF = (HA2.pow(2).sum(dim=(1, 2))) / float(H)
    return eig_AF.reshape(HA.shape[:-2])

@torch.no_grad()
def adamw_optimize_cao_gpu(
    F0: Tensor,
    Zfd: Tensor,
    out_hw: Tuple[int, int],
    rs_cs_hw: Tuple[int, int, int, int],
    *,
    mask: Tensor | None = None,
    init_A: Tensor | None = None,
    max_iter: int = 2000,
    n_dirs: int = 8,
    sigma: float | Tensor = 0.01,
    noise: str = "gauss",
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    alpha_min: float | Tensor = 0.004,
    alpha_max: float | Tensor = 0.04,
    patience_window: int = 200,
    freeze_thresh: float = 0.0,
    check_every: int = 20,
    verbose_every: int = 200,
    objective: str | Callable[[Tensor], Tensor] = "eig",
    device: str | torch.device | None = None,
    #OTF 전용
    roi_mask: Tensor | None = None,
    otf_hf_band: tuple[float, float] = (0.40, 0.90),
    otf_radial_bins: int = 192,
    otf_gamma: float = 1.0,
    # >>> 디버깅 옵션
    debug: bool = False,
    debug_every: int = 200,
) -> dict:
    """
    ★변경점★
      - 모든 목적함수/영상 계산을 (Him,Wim)=out_hw '원본 해상도'에서 수행.
      - 전달된 F0(ZP-FFT)는 원본 en-face로 역변환+크롭 → 다시 원본 FFT(F0_native)로 변환.
      - Zfd / mask 역시 원본 해상도로 보간 후 사용.
      - 최종 반환 phase_fd는 메인 코드 호환을 위해 (Hfd,Wfd)로 재보간해서 리턴.
    """
    dev = torch.device(device) if device is not None else F0.device
    F0 = F0.to(dev)
    Zfd = Zfd.to(dev, dtype=torch.float32)

    Him, Wim = out_hw
    Hfd, Wfd, K = int(Zfd.shape[0]), int(Zfd.shape[1]), int(Zfd.shape[2])

    # pupil mask도 전달되면 디바이스/타입 정렬
    if mask is not None:
        mask = mask.to(dev, dtype=F0.dtype)

    # ── (A) '원본 해상도' 복원: F0(ZP)의 ifft → 크롭 → 다시 fft (제로패딩 영향 제거)
    #     (참고) utils.ifft2c/fft2c는 centered FFT라고 가정.
    I0_full  = ifft2c(F0)                                   # (Hfd,Wfd) → 공간
    I0_native = _center_crop(I0_full, (Him, Wim), rs_cs_hw)  # (Him,Wim)
    F0_native = fft2c(I0_native)                             # (Him,Wim) FFT

    # ── (B) Zernike 기저/마스크를 '원본 해상도'로 보간
    Z_native = _resize_fd_tensor(Zfd, (Him, Wim))            # (Him,Wim,K) float32
    if mask is None:
        F0n_in, F0n_out = F0_native, torch.zeros_like(F0_native)
        mask_native = None
    else:
        mask_native = _resize_fd_tensor(mask.real if torch.is_complex(mask) else mask, (Him, Wim))
        mask_native = mask_native.to(dtype=F0_native.dtype)
        F0n_in  = F0_native * mask_native
        F0n_out = F0_native * (1.0 - mask_native)

    # ── init A / Adam states / freeze
    if init_A is None:
        A = torch.zeros(K, dtype=torch.float32, device=dev)
    else:
        A = init_A.to(dev, dtype=torch.float32).clone()
        assert A.numel() == K
    m1 = torch.zeros_like(A)
    v  = torch.zeros_like(A)
    freeze_mask = torch.zeros(K, dtype=torch.bool, device=dev)

    # allow vector params
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, dtype=torch.float32, device=dev)
    if sigma.dim() == 0:
        sigma = sigma.expand(K)
    if not torch.is_tensor(alpha_min):
        alpha_min = torch.tensor(alpha_min, dtype=torch.float32, device=dev)
    if alpha_min.dim() == 0:
        alpha_min = alpha_min.expand(K)
    if not torch.is_tensor(alpha_max):
        alpha_max = torch.tensor(alpha_max, dtype=torch.float32, device=dev)
    if alpha_max.dim() == 0:
        alpha_max = alpha_max.expand(K)

    # objective picker
    if isinstance(objective, str):
        if objective.lower().startswith("ent"):
            obj_fn = entropy_shannon_anyshape
        elif objective.lower().startswith("eig"):
            obj_fn = lambda I: -eig_value_anyshape(I)
        elif objective.lower().startswith("otf"):
            obj_fn = lambda I: -otf_gain_log(
                I, hf_band=otf_hf_band, radial_bins=otf_radial_bins, gamma=otf_gamma, roi_mask=roi_mask
            )
        else:
            raise ValueError("objective must be 'entropy', 'eig', or a callable")
    else:
        obj_fn = objective

    # ── 로그 버퍼
    E_hist = torch.empty(max_iter, dtype=torch.float32, device=dev)
    A_hist = torch.empty(max_iter, K, dtype=torch.float32, device=dev)

    # ── 유틸
    dirs = torch.empty(n_dirs, K, dtype=torch.float32, device=dev)
    def tensordot_Z_A(Z: Tensor, A_like: Tensor) -> Tensor:
        if A_like.dim() == 1:
            return torch.tensordot(Z, A_like, dims=([2], [0]))      # (H,W)
        else:
            return torch.tensordot(Z, A_like.T, dims=([2],[0])).permute(2,0,1).contiguous()  # (N,H,W)

    best_E = torch.tensor(float("inf"), dtype=torch.float32, device=dev)
    no_improve = 0

    # ── baseline
    E0 = obj_fn(I0_native).reshape(()).item()
    if debug:
        print(f"[DEBUG-0] native baseline E0={E0:.6e} | shapes native: F0={tuple(F0_native.shape)} Z={tuple(Z_native.shape)}")

    for it in range(1, max_iter + 1):
        # draw directions
        if noise == "gauss":
            dirs.normal_()
        elif noise == "bernoulli":
            dirs.bernoulli_(0.5).mul_(2.0).add_(-1.0)
        else:
            raise ValueError("noise must be 'gauss' or 'bernoulli'")
        norms = dirs.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        dirs.div_(norms)
        dirs[:, freeze_mask] = 0.0

        # build A_pos/A_neg
        A_pos = A.unsqueeze(0) + sigma.unsqueeze(0) * dirs    # (N,K)
        A_neg = A.unsqueeze(0) - sigma.unsqueeze(0) * dirs

        # phases & images (★ 모두 원본 해상도 ★)
        phase_pos = tensordot_Z_A(Z_native, A_pos)            # (N,H,W)
        phase_neg = tensordot_Z_A(Z_native, A_neg)
        phase_now = tensordot_Z_A(Z_native, A)                # (H,W)

        I_pos_full = ifft2c(F0n_in.unsqueeze(0) * torch.exp(-1j*phase_pos) + F0n_out)  # (N,H,W)
        I_neg_full = ifft2c(F0n_in.unsqueeze(0) * torch.exp(-1j*phase_neg) + F0n_out)
        I_now_full = ifft2c(F0n_in * torch.exp(-1j*phase_now) + F0n_out)               # (H,W)

        I_pos = I_pos_full
        I_neg = I_neg_full
        I_now = I_now_full

        # objectives
        E_pos = obj_fn(I_pos).reshape(-1)
        E_neg = obj_fn(I_neg).reshape(-1)
        E_now = obj_fn(I_now).reshape(())

        # zeroth-order estimate
        scale   = (E_pos - E_neg) / (E_now.abs() + 1e-12)
        delta_E = (scale.view(-1,1) * dirs).mean(dim=0)       # (K,)

        # AdamW update
        t, T = it, max_iter
        alpha_vec = alpha_min + 0.5*(alpha_max - alpha_min)*(1.0 + math.cos(math.pi * t / T))
        m1 = beta1 * m1 + (1.0 - beta1) * delta_E
        v  = beta2 * v  + (1.0 - beta2) * (delta_E * delta_E)
        m_hat = m1 / (1.0 - (beta1 ** t))
        v_hat = v  / (1.0 - (beta2 ** t))
        step = alpha_vec * (m_hat / (v_hat.sqrt() + eps) + weight_decay * A)
        step[freeze_mask] = 0
        A = A - step

        # logs
        E_hist[it-1] = E_now
        A_hist[it-1] = A

        # monitor
        if (it % check_every) == 0:
            if (best_E - E_now) > 0.0:
                best_E = E_now
                no_improve = 0
            else:
                no_improve += check_every

        # freeze 판단
        if it >= patience_window:
            win = A_hist[it - patience_window:it]
            mean_delta = win[1:].diff(dim=0).abs().mean(dim=0)
            freeze_now = mean_delta < freeze_thresh
            freeze_mask = torch.logical_or(freeze_mask, freeze_now)
            if freeze_mask.all():
                E_hist = E_hist[:it]
                A_hist = A_hist[:it]
                break

        if (verbose_every > 0) and (it % verbose_every == 0):
            print(f"[ZO-AdamW] iter={it:4d}  E={E_now.item():.6f}  frozen={int(freeze_mask.sum().item())}/{K}")

    # ── 최종 보정 (원본 해상도)
    phase_native = tensordot_Z_A(Z_native, A)                 # (Him,Wim)
    I_corr_full  = ifft2c(F0n_in * torch.exp(-1j*phase_native) + F0n_out)
    I_corr       = I_corr_full                                # (Him,Wim) 바로 사용 (크롭 불필요)

    # 메인 코드 호환: phase_fd는 (Hfd,Wfd)로 되돌려서 반환
    phase_fd_pad = _resize_fd_tensor(phase_native, (Hfd, Wfd))  # (Hfd,Wfd)
    phase_img    = _phase_fd_to_fullimg(phase_fd_pad, out_hw)   # (Him,Wim) real
    CAOfilter_img = torch.exp(-1j * phase_img).to(dtype=F0.dtype)

    return {
        "A": A,
        "phase_fd": phase_fd_pad,          # (Hfd,Wfd)  ← 메인에서 그대로 사용 가능
        "I_corr": I_corr,                  # (Him,Wim) complex
        "E_hist": E_hist.detach().cpu(),
        "A_hist": A_hist.detach().cpu(),
        "freeze_mask": freeze_mask.detach().cpu(),
        "CAOfilter": CAOfilter_img.detach().cpu(),
    }


def distance_weighted_cao_filter(info, vol, centers_xy, loaded_filters, dev, dtype):
    """
    info: Info dict (subdivFactors, partialVols, zfMap, centerX/Y 포함)
    vol:  선형 인덱스 (Fortran order)
    centers_xy: (centerX, centerY) 1D arrays (길이=rows,cols 각각)
    loaded_filters: dict {volp(int): CAOfilter(torch.complex)}  already computed/loaded
    dev, dtype: torch device/dtype for output
    """
    rows, cols = tuple(np.array(info["subdivFactors"], int).ravel())
    m, n = np.unravel_index(vol, (rows, cols), order="F")
    centerX, centerY = centers_xy

    volDist = np.full((rows, cols), np.inf, dtype=float)
    pv = np.asarray(info["partialVols"], bool)
    for volp in np.flatnonzero(pv.ravel(order="F")):
        mp, np_ = np.unravel_index(volp, (rows, cols), order="F")
        d = np.hypot(centerX[m] - centerX[mp], centerY[n] - centerY[np_])
        volDist[mp, np_] = d

    wsum = None
    wden = 0.0
    for volp, filt in loaded_filters.items():   # filt: (Him,Wim) complex torch
        mp, np_ = np.unravel_index(volp, (rows, cols), order="F")
        d = volDist[mp, np_]
        if np.isfinite(d) and (d > 0):
            w = 1.0 / d
            wden += w
            wsum = (filt * w) if (wsum is None) else (wsum + filt * w)

    if (wsum is None) or (wden <= 0):
        # fallback: identity phase
        CAOfilter = torch.ones((int(info["numImgLines"]), int(info["numImgFrames"])),
                               device=dev, dtype=dtype)
    else:
        CAOfilter = (wsum / wden).to(device=dev, dtype=dtype)

    # 크기 정규화 → 단위복소
    return CAOfilter / (CAOfilter.abs() + 1e-12)

