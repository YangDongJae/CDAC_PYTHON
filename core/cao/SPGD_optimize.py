import torch
import torch.nn.functional as F
from utils.utils import fft2c, ifft2c, _center_crop,_center_embed, load_layers

def entropy_no_norm(xc: torch.Tensor) -> torch.Tensor:
    I = (xc.real**2 + xc.imag**2).clamp_min(1e-20)
    return -(I * torch.log(I)).sum()

def entropy_shannon(xc: torch.Tensor) -> torch.Tensor:
    # xc: complex image (H,W)
    I = (xc.real**2 + xc.imag**2).clamp_min(0)
    p = I / (I.sum() + 1e-12)      # 반드시 정규화!
    return -(p * (p + 1e-12).log()).sum()
    
def Eig_value(X: torch.Tensor) -> torch.Tensor:
    # 2D로 보장 (HxW), 복소 → 실수 크기
    HA = torch.abs(X).to(torch.float32)
    if HA.ndim != 2:
        HA = HA.reshape(HA.shape[-2], HA.shape[-1])

    oydim = HA.shape[0]

    # L2 정규화 (전체 합), 평균 제거
    denom = torch.sqrt(torch.sum(HA**2)).clamp_min(1e-20)
    HA = HA / denom
    HA = HA - torch.mean(HA)

    # === 빠른 계산(고유값 합 == trace) ===
    # trace(HA*HA' / oydim) = (1/oydim) * ||HA||_F^2
    eig_AF = (HA.pow(2).sum()) / float(oydim)
    return eig_AF    


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
def spgd_optimize_cao(
    F0: torch.Tensor,
    Zfd: torch.Tensor,
    out_hw: tuple[int, int],
    rs_cs_hw: tuple[int, int, int, int],
    *,
    mask: torch.Tensor | None = None,
    init_A: torch.Tensor | None = None,
    max_iter: int = 10000,
    p_pick: float = 0.5,
    sigma: float = 0.5,
    gamma: float = -0.5,
    early_stop_patience: int = 1000,
    min_delta: float = 1e-6,
    freeze_mask: torch.Tensor | None = None,
    verbose_every: int = 1000,
    objective_fn = None,
):
    """
    F0     : 기준 en-face의 FFT (centered)  — (Hfd,Wfd) complex
    Zfd    : Zernike 기저 (FD)               — (Hfd,Wfd,K) float
    out_hw : 원래 이미지 크기 (Him,Wim)
    rs_cs_hw: _center_embed에서 받은 (rs,cs,h,w)
    mask   : pupil 마스크 (Hfd,Wfd) float/byte, None이면 전체 보정
    init_A : 초기 계수 (K,)  — 없으면 0 벡터
    반환   : dict(A, phase_fd, I_corr, E_hist, A_hist)
    """
    dev = F0.device
    Zfd = Zfd.to(device=dev, dtype=torch.float32)

    if objective_fn is None:
        objective_fn = entropy_shannon  # 상단에 이미 정의되어 있음

    Him, Wim = out_hw
    Hfd, Wfd, K = int(Zfd.shape[0]), int(Zfd.shape[1]), int(Zfd.shape[2])

    if mask is not None:
        m = mask.to(device=dev, dtype=F0.dtype)

    # pupil 분리
    if mask is None:
        F0_in  = F0
        F0_out = torch.zeros_like(F0)
    else:
        F0_in  = F0 * m
        F0_out = F0 * (1.0 - m)

    # 초기 A, freeze 마스크
    if init_A is None:
        A = torch.zeros(K, dtype=torch.float32, device=dev)
    else:
        A = init_A.to(device=dev, dtype=torch.float32).clone()
        assert A.numel() == K
    if freeze_mask is None:
        freeze_mask = torch.zeros(K, dtype=torch.bool, device=dev)
    else:
        freeze_mask = freeze_mask.to(device=dev, dtype=torch.bool)

    # 로그/히스토리
    E_hist = torch.empty(max_iter, dtype=torch.float64, device="cpu")
    A_hist = torch.empty(max_iter, K, dtype=torch.float32, device="cpu")

    best_E = float("inf")
    no_improve = 0

    for it in range(1, max_iter + 1):
        # 0/σ perturbation
        pick   = (torch.rand(K, device=dev) < p_pick).to(torch.float32)
        deltaA = sigma * pick
        deltaA[freeze_mask] = 0.0

        A_pos = A + deltaA
        A_neg = A - deltaA

        # φ = Σ a_k Z_k
        phase_pos = torch.tensordot(Zfd, A_pos, dims=([2],[0]))  # (Hfd,Wfd)
        phase_neg = torch.tensordot(Zfd, A_neg, dims=([2],[0]))
        phase_now = torch.tensordot(Zfd, A,     dims=([2],[0]))

        # pupil 안만 보정, 바깥 유지
        I_pos_full = ifft2c(F0_in * torch.exp(-1j * phase_pos) + F0_out)
        I_neg_full = ifft2c(F0_in * torch.exp(-1j * phase_neg) + F0_out)
        I_now_full = ifft2c(F0_in * torch.exp(-1j * phase_now) + F0_out)

        # 원 크기 복원
        I_pos = _center_crop(I_pos_full, (Him, Wim), rs_cs_hw)
        I_neg = _center_crop(I_neg_full, (Him, Wim), rs_cs_hw)
        I_now = _center_crop(I_now_full, (Him, Wim), rs_cs_hw)

        # 목적함수 (정규화 Shannon entropy)
        E_pos = objective_fn(I_pos)
        E_neg = objective_fn(I_neg)
        deltaE = (E_pos - E_neg).item()

        # 업데이트
        A = A + gamma * deltaE * deltaA

        # 모니터링/조기종료
        E_now = objective_fn(I_now).item()
        E_hist[it-1] = E_now
        A_hist[it-1] = A.detach().cpu()

        if (best_E - E_now) > min_delta:
            best_E = E_now
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"[SPGD] Early stop at iter {it}  best={best_E:.6f}  curr={E_now:.6f}")
                E_hist = E_hist[:it]
                A_hist = A_hist[:it]
                break

        if (it % verbose_every) == 0:
            print(f"[SPGD] iter={it:4d}  H={E_now:.6f}")

    # 최종 보정
    phase_fd = torch.tensordot(Zfd, A, dims=([2],[0]))
    I_corr_full = ifft2c(F0_in * torch.exp(-1j * phase_fd) + F0_out)
    I_corr = _center_crop(I_corr_full, (Him, Wim), rs_cs_hw)

    return {
        "A": A,                         # (K,)
        "phase_fd": phase_fd,           # (Hfd,Wfd)
        "I_corr": I_corr,               # (Him,Wim) complex
        "E_hist": E_hist,               # (T,)
        "A_hist": A_hist,               # (T,K)
    }

import torch
from utils.utils import fft2c, ifft2c, _center_crop

@torch.no_grad()
def spgd_optimize_cao_gpu(
    F0: torch.Tensor,
    Zfd: torch.Tensor,
    out_hw: tuple[int, int],
    rs_cs_hw: tuple[int, int, int, int],
    *,
    mask: torch.Tensor | None = None,
    init_A: torch.Tensor | None = None,
    max_iter: int = 10000,
    p_pick: float = 0.5,
    sigma: float = 0.5,
    gamma: float = -0.5,
    early_stop_patience: int = 1000,
    min_delta: float = 1e-6,
    freeze_mask: torch.Tensor | None = None,
    verbose_every: int = 1000,
    objective_fn = None,
    device: str | torch.device | None = None,   # GPU: explicit device
    n_dirs: int = 8,                             # GPU: evaluate N perturbations in parallel
    check_every: int = 20,                       # GPU: reduce sync frequency
):
    """
    Returns dict with: A, phase_fd, I_corr, E_hist, A_hist (all stay on device; move to .cpu() when you plot).
    """
    if objective_fn is None:
        objective_fn = entropy_shannon  # 상단에 이미 정의되어 있음

    # ---- device / dtypes ----
    dev = torch.device(device) if device is not None else F0.device
    print("Now Using : " , dev)
    F0  = F0.to(dev)                               # complex64
    Zfd = Zfd.to(dev, dtype=torch.float32)         # (Hfd,Wfd,K)
    if mask is not None:
        mask = mask.to(dev, dtype=F0.dtype)        # complex dtype broadcast-ok

    Him, Wim = out_hw
    Hfd, Wfd, K = Zfd.shape

    # pupil split (stay on GPU)
    if mask is None:
        F0_in, F0_out = F0, torch.zeros_like(F0)
    else:
        F0_in  = F0 * mask
        F0_out = F0 * (1.0 - mask)

    # coefficients
    if init_A is None:
        A = torch.zeros(K, dtype=torch.float32, device=dev)
    else:
        A = init_A.to(dev, dtype=torch.float32).clone()
        assert A.numel() == K
    if freeze_mask is None:
        freeze_mask = torch.zeros(K, dtype=torch.bool, device=dev)
    else:
        freeze_mask = freeze_mask.to(dev, dtype=torch.bool)

    gamma_t = torch.tensor(gamma, dtype=torch.float32, device=dev)  # GPU: avoid host scalar

    # logs (keep on GPU during loop; move to CPU after)
    E_hist = torch.empty(max_iter, dtype=torch.float32, device=dev)
    A_hist = torch.empty(max_iter, K, dtype=torch.float32, device=dev)

    best_E = torch.tensor(float("inf"), dtype=torch.float32, device=dev)
    no_improve = 0

    # buffers for batching
    # deltaA: (n_dirs, K)
    # we’ll reuse the same tensor each iter to reduce allocs
    deltaA = torch.empty(n_dirs, K, dtype=torch.float32, device=dev)

    for it in range(1, max_iter + 1):
        # ---- sample n_dirs perturbations in parallel ----
        # 0/σ Bernoulli(p_pick)
        deltaA.bernoulli_(p_pick).mul_(sigma)
        deltaA[:, freeze_mask] = 0.0

        A_pos = A.unsqueeze(0) + deltaA         # (N,K)
        A_neg = A.unsqueeze(0) - deltaA         # (N,K)

        # φ = Σ a_k Z_k → (Hfd,Wfd,K) · (K,N) -> (Hfd,Wfd,N) -> (N,Hfd,Wfd)
        phase_pos = torch.tensordot(Zfd, A_pos.T, dims=([2],[0])).permute(2,0,1).contiguous()
        phase_neg = torch.tensordot(Zfd, A_neg.T, dims=([2],[0])).permute(2,0,1).contiguous()

        # apply in FD (broadcast F0_in)
        # I_full: (N,Hfd,Wfd) complex
        I_pos_full = ifft2c(F0_in.unsqueeze(0) * torch.exp(-1j*phase_pos) + F0_out)
        I_neg_full = ifft2c(F0_in.unsqueeze(0) * torch.exp(-1j*phase_neg) + F0_out)

        # crop back to image size → stack (N,H,W)
        I_pos = torch.stack([_center_crop(I_pos_full[n], (Him, Wim), rs_cs_hw) for n in range(n_dirs)], dim=0)
        I_neg = torch.stack([_center_crop(I_neg_full[n], (Him, Wim), rs_cs_hw) for n in range(n_dirs)], dim=0)

        # vectorized objective for N samples → (N,)
        E_pos = objective_fn(I_pos).reshape(-1)
        E_neg = objective_fn(I_neg).reshape(-1)

        # SPGD update: average over directions (reduces variance, better GPU usage)
        deltaE = (E_pos - E_neg)                          # (N,)
        # ΔA = mean_n [ gamma * δE_n * δA_n ]
        update = (gamma_t * deltaE.view(-1,1) * deltaA).mean(dim=0)   # (K,)
        A = A + update

        # monitor using current A (single forward)
        phase_now = torch.tensordot(Zfd, A, dims=([2],[0]))   # (Hfd,Wfd)
        I_now_full = ifft2c(F0_in * torch.exp(-1j*phase_now) + F0_out)
        I_now = _center_crop(I_now_full, (Him, Wim), rs_cs_hw)

        E_now = objective_fn(I_now).reshape(())               # scalar tensor
        E_hist[it-1] = E_now
        A_hist[it-1] = A

        # ---- early stop & verbose (do these sparsely to avoid sync) ----
        if (it % check_every) == 0:
            # this sync point happens only every check_every iters
            if (best_E - E_now) > min_delta:
                best_E = E_now
                no_improve = 0
            else:
                no_improve += check_every
                if no_improve >= early_stop_patience:
                    print(f"[SPGD] Early stop at iter {it}  best={best_E.item():.6f}  curr={E_now.item():.6f}")
                    E_hist = E_hist[:it]
                    A_hist = A_hist[:it]
                    break

        if (verbose_every > 0) and ((it % verbose_every) == 0):
            # one sync print per verbose interval
            print(f"[SPGD] iter={it:4d}  H={E_now.item():.6f}")

    # final correction
    phase_fd = torch.tensordot(Zfd, A, dims=([2],[0]))   # (Hfd,Wfd)
    I_corr_full = ifft2c(F0_in * torch.exp(-1j*phase_fd) + F0_out)
    I_corr = _center_crop(I_corr_full, (Him, Wim), rs_cs_hw)

    phase_img = _phase_fd_to_fullimg(phase_fd, out_hw)           # (Him,Wim) real
    CAOfilter_img = torch.exp(-1j * phase_img).to(dtype=F0.dtype)
    
    # move logs to CPU only once (keep tensors otherwise on device)
    return {
        "A": A,                                  # (K,) on dev
        "phase_fd": phase_fd,                    # (Hfd,Wfd) on dev
        "I_corr": I_corr,                        # (Him,Wim) complex on dev
        "E_hist": E_hist.detach().cpu(),         # (T,)
        "A_hist": A_hist.detach().cpu(),         # (T,K)    "phase_img": phase_img.detach().cpu(),          # <<< 추가
        "phase_img": phase_img.detach().cpu(),          # <<< 추가
        "CAOfilter": CAOfilter_img.detach().cpu(), 
    }    