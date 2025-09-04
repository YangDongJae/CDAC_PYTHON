from __future__ import annotations
import math, torch
from torch import Tensor
from typing import Optional, Sequence, Dict, Any

# util들은 네 프로젝트의 모듈을 그대로 사용
from utils.utils import fft2c, ifft2c, _center_crop


# ── Utilities ─────────────────────────────────────────────────────────
@torch.no_grad()
def _hann2d(h:int, w:int, dtype=torch.float64, device=None)->Tensor:
    wy = torch.hann_window(h, dtype=dtype, device=device)
    wx = torch.hann_window(w, dtype=dtype, device=device)
    return wy[:, None] * wx[None, :]

@torch.no_grad()
def _intensity(xc: Tensor) -> Tensor:
    return (xc.real**2 + xc.imag**2).clamp_min_(0)

@torch.no_grad()
def _fft_mag2_centered(img: Tensor) -> Tensor:
    F = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
    return (F.real**2 + F.imag**2).clamp_min_(0)

@torch.no_grad()
def _radial_profile(power: Tensor, nbins: int = 192) -> tuple[Tensor, Tensor]:
    H, W = power.shape[-2:]
    yy = torch.linspace(-1, 1, H, device=power.device, dtype=power.dtype)
    xx = torch.linspace(-1, 1, W, device=power.device, dtype=power.dtype)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    R = torch.sqrt(X*X + Y*Y)                 # 0..sqrt(2)
    Rn = (R / math.sqrt(2)).clamp_max(1.0)    # 0..1
    bins = torch.linspace(0, 1.0, nbins+1, device=power.device, dtype=power.dtype)
    idx = torch.bucketize(Rn.reshape(-1), bins) - 1
    idx = idx.clamp(0, nbins-1)

    rb  = torch.zeros(nbins, dtype=power.dtype, device=power.device)
    cnt = torch.zeros(nbins, dtype=power.dtype, device=power.device)
    flat= power.reshape(-1)

    rb.index_add_(0, idx, flat)
    cnt.index_add_(0, idx, torch.ones_like(flat))
    r_centers = bins[:-1] + (bins[1]-bins[0]) * 0.5
    return rb/(cnt + 1e-18), r_centers

@torch.no_grad()
def _apply_roi(I: Tensor, roi: Optional[Tensor]) -> Tensor:
    if roi is None:
        return I
    return I * roi.to(I.dtype).to(I.device)


# ── Metrics (↑가 좋음) ───────────────────────────────────────────────
@torch.no_grad()
def metric_otf_gain_log(
    I_now_cpx: Tensor,
    I_ref_cpx: Tensor,
    *,
    hf_band: tuple[float,float] = (0.40, 0.90),
    radial_bins: int = 192,
    gamma: float = 1.0,
    use_hann: bool = True,
    roi_mask: Optional[Tensor] = None,
) -> Tensor:
    In = _intensity(I_now_cpx).to(torch.float64)
    Ir = _intensity(I_ref_cpx).to(torch.float64)
    In = _apply_roi(In, roi_mask); Ir = _apply_roi(Ir, roi_mask)

    In = In/(In.sum() + 1e-18); Ir = Ir/(Ir.sum() + 1e-18)
    if use_hann:
        win = _hann2d(*In.shape[-2:], dtype=In.dtype, device=In.device)
        In *= win; Ir *= win

    Pn = _fft_mag2_centered(In)
    Pr = _fft_mag2_centered(Ir)
    rn, r = _radial_profile(Pn, nbins=radial_bins)
    rr, _ = _radial_profile(Pr, nbins=radial_bins)

    rmin, rmax = hf_band
    mask = (r>=rmin) & (r<=rmax)
    if not torch.any(mask):
        mask = slice(None)
        w = r
    else:
        w = (r**gamma)[mask]

    ratio = (rn[mask] + 1e-18) / (rr[mask] + 1e-18)
    return (w * ratio.log()).sum() / (w.sum() + 1e-18)

@torch.no_grad()
def metric_strehl_proxy(I_now_cpx: Tensor, I_ref_cpx: Tensor, *, roi_mask: Optional[Tensor]=None) -> Tensor:
    In = _intensity(I_now_cpx); Ir = _intensity(I_ref_cpx)
    In = _apply_roi(In, roi_mask); Ir = _apply_roi(Ir, roi_mask)
    In = In/(In.sum()+1e-18); Ir = Ir/(Ir.sum()+1e-18)
    return In.max()/(Ir.max()+1e-18)

@torch.no_grad()
def strehl_marechal_from_phi(phi_res: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        m = torch.ones_like(phi_res, dtype=torch.bool)
    else:
        m = (mask > 0.5)
    var = phi_res[m].float().var(unbiased=False).clamp_min(0)
    return torch.exp(-var)


# ── Bases & steps ────────────────────────────────────────────────────
@torch.no_grad()
def normalize_zernike_by_pupil_L2(Zfd: Tensor, mask: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    if mask is None:
        m = torch.ones_like(Zfd[...,0], dtype=torch.float32)
    else:
        m = (mask.real if torch.is_complex(mask) else mask).to(torch.float32)
    num = ((Zfd**2) * m.unsqueeze(-1)).sum(dim=(0,1))
    den = m.sum() + 1e-18
    norms = torch.sqrt(num/den).clamp_min(1e-12)  # (K,)
    Zn = Zfd / norms.view(1,1,-1)
    return Zn, norms

@torch.no_grad()
def gen_orthogonal_dirs_qr(n_dirs:int, K:int, freeze_mask: Tensor, device=None, dtype=torch.float32) -> Tensor:
    if n_dirs <= 0:
        raise ValueError("n_dirs must be >=1")
    M = torch.randn(K, max(n_dirs,1), device=device, dtype=dtype)
    if torch.any(freeze_mask):
        M[freeze_mask, :] *= 1e-6
    Q,_ = torch.linalg.qr(M, mode='reduced')   # (K, n_dirs)
    dirs = Q.T.contiguous()
    return dirs/dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)

@torch.no_grad()
def trust_region_clip(step: Tensor, Delta: float, per_mode_clip: float) -> Tensor:
    step = step.clamp(min=-per_mode_clip, max=per_mode_clip)  # per-mode
    nrm = step.norm()
    if nrm > Delta:
        step = step * (Delta/(nrm + 1e-18))                   # global
    return step


# ── Main optimizer ───────────────────────────────────────────────────
@torch.no_grad()
def cao_optimize_selfsupervised(
    *,
    F0: Tensor,                   # (Hfd,Wfd) complex
    Zfd: Tensor,                  # (Hfd,Wfd,K) float
    out_hw: tuple[int,int],       # (Him,Wim)
    rs_cs_hw: tuple[int,int,int,int],
    mask: Optional[Tensor] = None,     # (Hfd,Wfd) in {0,1}
    roi_mask: Optional[Tensor] = None, # (Him,Wim)
    init_A: Optional[Tensor] = None,   # (K,)

    # iterations / perturbations
    max_iter: int = 6000,
    n_dirs: int = 8,
    sigma0: float | Tensor = 5e-3,
    sigma_decay: bool = True,
    sigma_floor: float = 5e-5,

    # AdamW
    beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
    weight_decay: float = 1e-4,
    alpha_min: float | Tensor = 2e-3,
    alpha_max: float | Tensor = 2e-2,

    # accept / backtracking
    compare_to: str = "best",         # "best"|"init"
    accept_mad_window: int = 50,
    accept_k: float = 1.0,
    backtracking_steps: int = 2,
    backtracking_shrink: float = 0.5,

    # trust region
    trust_region_delta: float = 0.5,  # rad (global)
    per_mode_clip: float = 0.5,       # rad (per mode)

    # freeze(옵션)
    patience_window: int = 200,
    freeze_thresh: float = 0.0,

    # metrics
    metric: str = "otf_gain_log",     # "otf_gain_log"|"strehl_proxy"|"strehl_marechal"
    hf_band: tuple[float,float] = (0.40, 0.90),
    radial_bins: int = 192,
    gamma: float = 1.0,
    strehl_from_phi_fn = None,

    # coarse-to-fine schedule
    mode_schedule: Optional[Sequence[Sequence[int]]] = None,
    stage_iters: Optional[Sequence[int]] = None,

    # logs
    verbose_every: int = 200,
    debug_every: int = 400,

    device: Optional[torch.device | str] = None,
) -> Dict[str,Any]:

    dev = torch.device(device) if device is not None else F0.device
    F0  = F0.to(dev)
    Zfd = Zfd.to(dev, dtype=torch.float32)
    Him, Wim = out_hw
    if mask is not None:
        mask = mask.to(dev, dtype=F0.dtype)
    if roi_mask is not None:
        roi_mask = roi_mask.to(dev, dtype=torch.float32)

    Hfd, Wfd, K_full = Zfd.shape
    # Zernike normalization on pupil
    Zfd_norm, z_norms = normalize_zernike_by_pupil_L2(Zfd, mask)

    # pupil split
    if mask is None:
        F0_in, F0_out = F0, torch.zeros_like(F0)
    else:
        F0_in  = F0 * mask
        F0_out = F0 * (1.0 - mask)

    # baseline image
    I_init = _center_crop(ifft2c(F0_in + F0_out), (Him, Wim), rs_cs_hw)

    # metric selection
    use_img_metric = metric in ("otf_gain_log", "strehl_proxy")
    if metric == "otf_gain_log":
        def metric_img(I_now, I_ref):
            return metric_otf_gain_log(
                I_now, I_ref, hf_band=hf_band,
                radial_bins=radial_bins, gamma=gamma,
                use_hann=True, roi_mask=roi_mask
            )
        metric_name = f"otf_gain_log(hf={hf_band},γ={gamma})"
    elif metric == "strehl_proxy":
        def metric_img(I_now, I_ref):
            return metric_strehl_proxy(I_now, I_ref, roi_mask=roi_mask)
        metric_name = "strehl_proxy"
    elif metric == "strehl_marechal":
        if strehl_from_phi_fn is None:
            def metric_phi(phi):
                return strehl_marechal_from_phi(phi, mask)
        else:
            def metric_phi(phi):
                return strehl_from_phi_fn(phi, mask)
        metric_name = "strehl_marechal"
    else:
        raise ValueError("metric must be one of {'otf_gain_log','strehl_proxy','strehl_marechal'}")

    # compare_to
    compare_to = compare_to.lower()
    if compare_to not in ("best", "init"):
        raise ValueError("compare_to must be 'best' or 'init'")

    # coarse→fine schedule
    if mode_schedule is None:
        if K_full <= 8:
            mode_schedule = [list(range(K_full))]
        else:
            mode_schedule = [list(range(min(4, K_full))),
                             list(range(min(8, K_full))),
                             list(range(K_full))]
    n_stages = len(mode_schedule)
    if stage_iters is None:
        it_per = max(1, max_iter // n_stages)
        stage_iters = [it_per]*(n_stages-1) + [max_iter - it_per*(n_stages-1)]
    assert len(stage_iters) == n_stages

    # states (full)
    A_full  = torch.zeros(K_full, dtype=torch.float32, device=dev) if init_A is None \
              else init_A.to(dev, dtype=torch.float32).clone()
    m1_full = torch.zeros_like(A_full)
    v_full  = torch.zeros_like(A_full)
    freeze_mask_full = torch.zeros(K_full, dtype=torch.bool, device=dev)

    # vectorize params
    def _as_vec(x, K):
        if torch.is_tensor(x):
            t = x.to(dev, dtype=torch.float32)
        else:
            t = torch.tensor(x, dtype=torch.float32, device=dev)
        if t.dim()==0:
            t = t.expand(K)
        return t
    sigma0_full  = _as_vec(sigma0, K_full).clamp_min(1e-6)
    alpha_minV_f = _as_vec(alpha_min, K_full)
    alpha_maxV_f = _as_vec(alpha_max, K_full)

    # initial reference and Q0
    I_best = I_init.clone()
    if use_img_metric:
        Q0 = metric_img(I_best, I_init if compare_to=="init" else I_best)
    else:
        Q0 = metric_phi(torch.zeros(Hfd, Wfd, dtype=torch.float32, device=dev))
    print(f"[SS-CAO] init Q={float(Q0):.6e}  (metric={metric_name}, compare_to={compare_to})")

    # logs
    T_total = sum(stage_iters)
    Q_hist  = torch.empty(T_total, dtype=torch.float32, device=dev)
    A_hist  = torch.empty(T_total, K_full, dtype=torch.float32, device=dev)
    Q_best  = Q0
    best_snapshot = {"A": A_full.clone(), "I": I_best.clone(), "Q": float(Q_best)}

    # handy crop indices
    rs, cs, h, w = rs_cs_hw

    # ── Stage loop ────────────────────────────────────────────────────
    it_glob = 0
    for stage_idx, modes in enumerate(mode_schedule):
        modes = list(modes)
        stage_K   = len(modes)
        it_stage  = stage_iters[stage_idx]
        if it_stage <= 0 or stage_K == 0:
            continue

        # activate only current modes
        active_mask = torch.zeros(K_full, dtype=torch.bool, device=dev)
        active_mask[modes] = True
        # freeze others permanently (coarse→fine)
        freeze_mask_full = torch.logical_or(freeze_mask_full, ~active_mask)

        # views into active slice
        def view_active(x_full: Tensor) -> Tensor:
            return x_full[modes]

        A   = view_active(A_full)
        m1  = view_active(m1_full)
        v   = view_active(v_full)
        aMin= view_active(alpha_minV_f)
        aMax= view_active(alpha_maxV_f)
        s0V = view_active(sigma0_full)

        # local helper: tensordot on active subspace
        Z_act = Zfd_norm[..., modes]  # (Hfd,Wfd,K_act)

        def tensordot_Z_A_active(A_like: torch.Tensor) -> torch.Tensor:
            Z_act = Zfd_norm[..., modes]                    # (Hfd,Wfd,K_act) float32
            A_like = A_like.to(Z_act.dtype)                 # <<< dtype 정렬 (float32)
            if A_like.dim() == 1:
                return torch.tensordot(Z_act, A_like, dims=([2], [0]))            # (Hfd,Wfd)
            else:  # (N,K_act)
                out = torch.tensordot(Z_act, A_like.T, dims=([2], [0]))           # (Hfd,Wfd,N)
                return out.permute(2, 0, 1).contiguous()                           # (N,Hfd,Wfd)

        # stage iterations
        for it in range(1, it_stage+1):
            t_global = it_glob + it

            # 1) sigma schedule
            if sigma_decay:
                cosw = 0.5 * (1.0 + math.cos(math.pi * (t_global-1) / max(1, T_total)))
                sigma_t = (s0V * (0.1 + 0.9 * cosw)).clamp_min(sigma_floor)
            else:
                sigma_t = s0V

            # 2) orthogonal directions (in active space)
            dirs = gen_orthogonal_dirs_qr(n_dirs, stage_K, freeze_mask_full[modes], device=dev, dtype=torch.float32)

            # 3) A±
            A_pos = A.unsqueeze(0) + sigma_t.unsqueeze(0) * dirs
            A_neg = A.unsqueeze(0) - sigma_t.unsqueeze(0) * dirs

            # 4) φ(now/±) – 반드시 active basis 사용!
            phase_now = tensordot_Z_A_active(A)        # (Hfd, Wfd) float32
            phase_pos = tensordot_Z_A_active(A_pos)    # (N, Hfd, Wfd) float32
            phase_neg = tensordot_Z_A_active(A_neg)    # (N, Hfd, Wfd) float32
            
            # 5) Q(now/±): metric path 고정
            if not use_img_metric:
                Q_now = metric_phi(phase_now)
                Q_pos = torch.stack([metric_phi(phase_pos[n]) for n in range(phase_pos.shape[0])], dim=0)
                Q_neg = torch.stack([metric_phi(phase_neg[n]) for n in range(phase_neg.shape[0])], dim=0)

                Q_now = Q_now.float()
                Q_pos = Q_pos.float()
                Q_neg = Q_neg.float()
                
            else:
                I_now_full = ifft2c(F0_in * torch.exp(-1j*phase_now) + F0_out)
                I_now      = I_now_full[rs:rs+h, cs:cs+w]

                # batch FFT for pos/neg
                Npos = phase_pos.shape[0]
                Nneg = phase_neg.shape[0]
                F0_in_pos  = F0_in.unsqueeze(0).expand(Npos, -1, -1)
                F0_out_pos = F0_out.unsqueeze(0).expand(Npos, -1, -1)
                F0_in_neg  = F0_in.unsqueeze(0).expand(Nneg, -1, -1)
                F0_out_neg = F0_out.unsqueeze(0).expand(Nneg, -1, -1)

                I_pos_full = ifft2c(F0_in_pos * torch.exp(-1j*phase_pos) + F0_out_pos)
                I_neg_full = ifft2c(F0_in_neg * torch.exp(-1j*phase_neg) + F0_out_neg)
                I_pos = I_pos_full[:, rs:rs+h, cs:cs+w]
                I_neg = I_neg_full[:, rs:rs+h, cs:cs+w]

                I_ref = I_init if compare_to == "init" else I_best
                Q_now = metric_img(I_now, I_ref)
                Q_pos = torch.stack([metric_img(I_pos[n], I_ref) for n in range(I_pos.shape[0])], dim=0)
                Q_neg = torch.stack([metric_img(I_neg[n], I_ref) for n in range(I_neg.shape[0])], dim=0)

                Q_now = Q_now.float()
                Q_pos = Q_pos.float()
                Q_neg = Q_neg.float()

            # 6) ZO-gradient (SPSA)
            c = float(sigma_t.mean().item())
            g_est = ((Q_pos - Q_neg).view(-1,1) * dirs) / (2.0 * max(c, 1e-12))
            g = g_est.mean(dim=0)  # (K_act,)

            # 7) AdamW + cosine lr
            alpha_vec = aMin + 0.5*(aMax - aMin) * (1.0 + math.cos(math.pi * t_global / T_total))
            m1 = beta1*m1 + (1.0-beta1)*g
            v  = beta2*v  + (1.0-beta2)*(g*g)
            m_hat = m1 / (1.0 - (beta1 ** t_global))
            v_hat = v  / (1.0 - (beta2 ** t_global))
            step = alpha_vec * (m_hat / (v_hat.sqrt() + eps) + weight_decay * A)

            # 8) trust region + freeze
            step = trust_region_clip(step, trust_region_delta, per_mode_clip)
            step[freeze_mask_full[modes]] = 0.0

            # 9) candidate eval + MAD-accept + backtracking
            def eval_Q_of(A_cand: Tensor) -> tuple[Tensor, Optional[Tensor]]:
                phi_cand = tensordot_Z_A_active(A_cand)
                if use_img_metric:
                    I_cand_full = ifft2c(F0_in * torch.exp(-1j*phi_cand) + F0_out)
                    I_cand = _center_crop(I_cand_full, (Him, Wim), rs_cs_hw)
                    I_ref  = I_init if compare_to=="init" else I_best
                    Q_cand = metric_img(I_cand, I_ref)
                    return Q_cand, I_cand
                else:
                    Q_cand = metric_phi(phi_cand)
                    return Q_cand, None

            A_candidate = A + step
            Q_cand, I_cand = eval_Q_of(A_candidate)

            # noise-aware accept using recent MAD
            window = min(accept_mad_window, t_global-1)
            eps_mad = 0.0
            if window > 10:
                qtail = Q_hist[t_global-window:t_global]
                med = qtail.median()
                mad = (qtail - med).abs().median()
                eps_mad = float(1.4826 * mad * accept_k)

            accepted = torch.isfinite(Q_cand) and (Q_cand >= Q_now - eps_mad)
            tried = 0
            while (not accepted) and (tried < backtracking_steps):
                tried += 1
                step = trust_region_clip(step * backtracking_shrink, trust_region_delta, per_mode_clip)
                A_candidate = A + step
                Q_cand, I_cand = eval_Q_of(A_candidate)
                accepted = torch.isfinite(Q_cand) and (Q_cand >= Q_now - eps_mad)

            if accepted:
                # apply to active slice
                A = A_candidate
                Q_now = Q_cand
                # sync to full state
                A_full[modes]  = A
                m1_full[modes] = m1
                v_full[modes]  = v

                # update best (and I_best only when image metric & compare_to="best")
                if Q_now > Q_best:
                    Q_best = Q_now
                    if use_img_metric:
                        I_best = I_cand.detach().clone()
                    best_snapshot["A"] = A_full.clone()
                    best_snapshot["I"] = I_best.clone() if use_img_metric else best_snapshot.get("I", None)
                    best_snapshot["Q"] = float(Q_best)

            # 10) freeze bookkeeping & logs
            Q_hist[t_global-1] = Q_now
            A_hist[t_global-1] = A_full

            if (t_global >= patience_window) and (freeze_thresh > 0.0):
                win = A_hist[t_global - patience_window:t_global]  # (W,K_full)
                mean_delta = win[1:].diff(dim=0).abs().mean(dim=0)
                freeze_now = (mean_delta < freeze_thresh)
                freeze_mask_full = torch.logical_or(freeze_mask_full, freeze_now)

            if (verbose_every>0) and (t_global % verbose_every == 0):
                print(f"[SS-CAO][S{stage_idx+1}/{n_stages}] it={t_global:5d}  "
                      f"Q={float(Q_now):.6e}  bestQ={float(Q_best):.6e}  "
                      f"||A||={A_full.norm():.3e}  accepted={'Y' if accepted else 'N'}  "
                      f"frozen={int(freeze_mask_full.sum().item())}/{K_full}")
            if (debug_every>0) and (t_global % debug_every == 0):
                dq0 = float((Q_pos[0]-Q_neg[0]).item()) if (Q_pos.numel()>0 and Q_neg.numel()>0) else float('nan')
                print(f"[SS-CAO][DBG] it={t_global:5d}  probe ΔQ0={dq0:.3e}  "
                      f"step.mean={float(step.mean().item()):.3e}  epsMAD={eps_mad:.3e}  "
                      f"accepted={'Y' if accepted else 'N'}")

        print(f"[SS-CAO] stage {stage_idx+1}/{n_stages} done. modes={modes}  "
              f"Q_best={float(Q_best):.6e}")
        it_glob += it_stage

    # final phase & corrected image
    phase_fd = torch.tensordot(Zfd_norm, A_full.to(torch.float32), dims=([2],[0]))
    I_corr_full= ifft2c(F0_in * torch.exp(-1j*phase_fd) + F0_out)
    I_corr     = _center_crop(I_corr_full, (Him, Wim), rs_cs_hw)

    meta = {
        "metric": metric_name,
        "compare_to": compare_to,
        "mode_schedule": [list(s) for s in mode_schedule],
        "stage_iters": list(stage_iters),
        "zernike_norms": z_norms.detach().cpu(),
    }
    return {
        "A": A_full.detach().cpu(),
        "phase_fd": phase_fd.detach().cpu(),
        "I_corr": I_corr.detach().cpu(),
        "Q_hist": Q_hist.detach().cpu(),
        "A_hist": A_hist.detach().cpu(),
        "freeze_mask": freeze_mask_full.detach().cpu(),
        "I_best": best_snapshot["I"].detach().cpu() if use_img_metric else None,
        "Q_best": float(Q_best),
        "meta": meta,
    }