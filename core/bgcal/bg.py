# ────────────────────────────────────────────────────────
#  core/bgcal/bg.py     (MATLAB calcBG / calcNoise 대응)
# ────────────────────────────────────────────────────────
import numpy as np, torch
from pathlib import Path
from scipy.signal import lombscargle
from types import SimpleNamespace
from core.registration.dftregistration import dftregistration1d

# ───────── util ─────────
def _get(info, key):
    """dict  또는  SimpleNamespace  모두 지원"""
    return info[key] if isinstance(info, dict) else getattr(info, key)


# ========================================================
# 1)  BG mean / bgOsc  + trigger-delay alignment
# ========================================================

@torch.no_grad()
def calc_bg(bin_path, info, *, device="cuda"):
    dev = torch.device(device)

    ns        = _get(info, "numSamples")          # 768
    K         = _get(info, "numUsedSamples")      # 280
    Lbg       = _get(info, "bgLines")             # 8704
    trig_del  = int(_get(info, "trigDelay"))
    bg_shift  = _get(info, "bgLineShift")

    used_idx  = torch.as_tensor(_get(info, "usedSamples"),
                                dtype=torch.long, device=dev) - 1
    bg_ref    = torch.as_tensor(_get(info, "bgRef"),
                                dtype=torch.float32, device=dev)

    # ── BG frame load ───────────────────────────────────
    f = Path(bin_path)
    with open(f, "rb") as fp:
        fp.seek(2 * (trig_del + ns * bg_shift))
        raw = np.fromfile(fp, count=ns * Lbg, dtype=np.uint16)
    bg_frame = torch.as_tensor(raw, dtype=torch.float32, device=dev)\
                   .view(ns, Lbg)

    # ── A) trigger-delay alignment (dftregistration1d) ──
    if torch.any(bg_ref):
        buf1 = torch.cat([bg_ref - 2**15,
                          torch.ones(ns - K, device=dev)])
        buf2 = bg_frame.mean(1) - 2**15
        _, _, delay = dftregistration1d(torch.fft.fft(buf1),
                                        torch.fft.fft(buf2), 1)
        trig_del -= int(delay)

        with open(f, "rb") as fp:
            fp.seek(2 * (trig_del + ns * bg_shift))
            raw = np.fromfile(fp, count=ns * Lbg, dtype=np.uint16)
        bg_frame = torch.as_tensor(raw, dtype=torch.float32,
                                   device=dev).view(ns, Lbg)

    # ── B) slice K-samples & bgMean ─────────────────
    bg_frame = bg_frame[used_idx]            # (K, Lbg)
    bg_mean  = bg_frame.mean(1)              # (K,)

    # ── C) 501-tap mean filter (replicate pad) ─────
    resid = torch.nn.functional.pad(
                bg_frame - bg_mean[:, None],
                (250, 250), mode="replicate"
            )                                # (K, Lbg+500)
    ker   = torch.ones(1, 1, 501, device=dev) / 501
    bg_res = torch.nn.functional.conv1d(
                resid.unsqueeze(1), ker, padding=0
             )[:, 0]                         # (K, Lbg)

    bg_res_sum = bg_res.abs().sum(0).cpu().numpy()
    var = float(np.dot(bg_res_sum, bg_res_sum))
    if var < 1e-12:
        return bg_mean, torch.zeros_like(bg_mean), trig_del

    # ── D) Lomb-Scargle (DC 제외) ───────────────────
    t     = np.arange(1, Lbg, dtype=float)
    freqs = 2*np.pi*t/Lbg
    pxx   = lombscargle(t, bg_res_sum[1:], freqs, normalize=False) * 2/var
    if pxx.max() <= 2*pxx[0]:
        return bg_mean, torch.zeros_like(bg_mean), trig_del

    # ── E) dominant period → bgOsc ─────────────────
    f_peak = int(np.argmax(pxx) + 1)
    T      = int(round(Lbg / f_peak))
    Lcrop  = (Lbg // T) * T
    osc_fft = torch.fft.fft(bg_res[:, :Lcrop], dim=1)  # (K, Lcrop)
    bin_idx = Lcrop // T
    bg_osc  = osc_fft[:, bin_idx].real                 # (K,)

    val   = bg_osc.abs().max().clamp(min=1e-12)
    theta = bg_osc.sum() / bg_osc.abs().sum().clamp(min=1e-12)
    bg_osc = (bg_osc / val / theta).real

    return bg_mean, bg_osc.to(bg_mean.dtype), trig_del


@torch.no_grad()
def calc_bg_kaist(bin_path, info, *, device="cuda"):
    dev = torch.device(device)

    ns        = int(_get(info, "numSamples"))          # 768
    K         = int(_get(info, "numUsedSamples"))      # 280
    Lbg       = int(_get(info, "bgLines"))             # 8704
    trig_del  = int(_get(info, "trigDelay"))
    bg_shift  = int(_get(info, "bgLineShift"))

    used_idx  = torch.as_tensor(_get(info, "usedSamples"),
                                dtype=torch.long, device=dev) - 1
    bg_ref    = torch.as_tensor(_get(info, "bgRef"),
                                dtype=torch.float32, device=dev)

    # ── BG frame load ───────────────────────────────────
    f = Path(bin_path)
    with open(f, "rb") as fp:
        fp.seek(2 * (trig_del + ns * bg_shift))
        raw = np.fromfile(fp, count=ns * Lbg, dtype=np.uint16)
    bg_frame = torch.as_tensor(raw, dtype=torch.float32, device=dev)\
                   .view(ns, Lbg)

    # ── A) trigger-delay alignment (dftregistration1d) ──
    if torch.any(bg_ref):
        buf1 = torch.cat([bg_ref - 2**15,
                          torch.ones(ns - K, device=dev)])
        buf2 = bg_frame.mean(1) - 2**15
        _, _, delay = dftregistration1d(torch.fft.fft(buf1),
                                        torch.fft.fft(buf2), 1)
        trig_del -= int(delay)

        with open(f, "rb") as fp:
            fp.seek(2 * (trig_del + ns * bg_shift))
            raw = np.fromfile(fp, count=ns * Lbg, dtype=np.uint16)
        bg_frame = torch.as_tensor(raw, dtype=torch.float32,
                                   device=dev).view(ns, Lbg)

    # ── B) slice K-samples & bgMean ─────────────────
    bg_frame = bg_frame[used_idx]            # (K, Lbg)
    bg_mean  = bg_frame.mean(1)              # (K,)

    # ── C) 501-tap mean filter (replicate pad) ─────
    resid = torch.nn.functional.pad(
                bg_frame - bg_mean[:, None],
                (250, 250), mode="replicate"
            )                                # (K, Lbg+500)
    ker   = torch.ones(1, 1, 501, device=dev) / 501
    bg_res = torch.nn.functional.conv1d(
                resid.unsqueeze(1), ker, padding=0
             )[:, 0]                         # (K, Lbg)

    bg_res_sum = bg_res.abs().sum(0).cpu().numpy()
    var = float(np.dot(bg_res_sum, bg_res_sum))
    if var < 1e-12:
        return bg_mean, torch.zeros_like(bg_mean), trig_del

    # ── D) Lomb-Scargle (DC 제외) ───────────────────
    t     = np.arange(1, Lbg, dtype=float)
    freqs = 2*np.pi*t/Lbg
    pxx   = lombscargle(t, bg_res_sum[1:], freqs, normalize=False) * 2/var
    if pxx.max() <= 2*pxx[0]:
        return bg_mean, torch.zeros_like(bg_mean), trig_del

    # ── E) dominant period → bgOsc ─────────────────
    f_peak = int(np.argmax(pxx) + 1)
    T      = int(round(Lbg / f_peak))
    Lcrop  = (Lbg // T) * T
    osc_fft = torch.fft.fft(bg_res[:, :Lcrop], dim=1)  # (K, Lcrop)
    bin_idx = Lcrop // T
    bg_osc  = osc_fft[:, bin_idx].real                 # (K,)

    val   = bg_osc.abs().max().clamp(min=1e-12)
    theta = bg_osc.sum() / bg_osc.abs().sum().clamp(min=1e-12)
    bg_osc = (bg_osc / val / theta).real

    return bg_mean, bg_osc.to(bg_mean.dtype), trig_del


# ========================================================
# 2)  Noise profile   (MATLAB calcNoise)
# ========================================================
@torch.no_grad()
def calc_noise(bin_path, info, *, device="cuda"):
    dev   = torch.device(device)
    ns    = _get(info, "numSamples")
    Lbg   = _get(info, "bgLines")
    idx   = torch.as_tensor(_get(info, "usedSamples"),
                            dtype=torch.long, device=dev) - 1

    with open(bin_path, "rb") as fp:
        fp.seek(2 * (_get(info, "trigDelay") + ns * _get(info, "bgLineShift")))
        raw = np.fromfile(fp, count=ns * Lbg, dtype=np.uint16)
    bg = torch.as_tensor(raw, dtype=torch.float32, device=dev)\
         .view(ns, Lbg)[idx]                             # (K,Lbg)

    bg_mean = torch.as_tensor(_get(info, "bgMean"), device=dev)
    coef = ((bg-2**15)*(bg_mean-2**15)[:,None]).sum(0) / \
           ((bg_mean-2**15)**2).sum()
    bg = (bg-2**15)/coef - bg_mean[:,None] + 2**15

    bg_osc = torch.as_tensor(_get(info, "bgOsc"), device=dev)
    if torch.any(bg_osc):
        a  = (bg*bg_osc[:,None]).sum(0)/(bg_osc**2).sum()
        bg = bg - bg_osc[:,None]*a

    disp   = torch.as_tensor(_get(info, "dispComp"),      device=dev)
    window = torch.as_tensor(_get(info, "spectralWindow"), device=dev)
    noise  = torch.fft.ifft(bg*disp[:,None]*window[:,None], dim=0)\
                 .abs().mean(1)
    return noise / noise.mean() * 0.7

@torch.no_grad()
def calc_noise_kaist(bin_path, info, *, device="cuda"):
    dev = torch.device(device)
    ns       = int(_get(info, "numSamples"))
    Lbg      = int(_get(info, "bgLines"))
    trig_del = int(_get(info, "trigDelay"))
    bg_shift = int(_get(info, "bgLineShift"))
    
    idx = torch.as_tensor(_get(info, "usedSamples"), dtype=torch.long, device=dev) - 1

    with open(bin_path, "rb") as fp:
        # ✅ 정수형으로 변환된 변수 사용
        fp.seek(2 * (trig_del + ns * bg_shift))
        raw = np.fromfile(fp, count=ns * Lbg, dtype=np.uint16)
    
    bg = torch.as_tensor(raw, dtype=torch.float32, device=dev).view(ns, Lbg)[idx]

    bg_mean = torch.as_tensor(_get(info, "bgMean"), device=dev, dtype=torch.float32)
    center_val = 2**15
    
    coef = ((bg - center_val) * (bg_mean - center_val)[:, None]).sum(0) / ((bg_mean - center_val)**2).sum()
    bg = (bg - center_val) / coef - bg_mean[:, None]

    bg_osc = torch.as_tensor(_get(info, "bgOsc"), device=dev, dtype=torch.float32)
    if torch.any(bg_osc.abs() > 1e-9):
        a = (bg * bg_osc[:, None]).sum(0) / (bg_osc**2).sum()
        bg = bg - bg_osc[:, None] * a

    disp   = torch.as_tensor(_get(info, "dispComp"), device=dev, dtype=torch.complex64)
    window = torch.as_tensor(_get(info, "spectralWindow"), device=dev, dtype=torch.float32)
    
    noise = torch.fft.ifft(bg * disp[:, None] * window[:, None], dim=0).abs().mean(1)
    
    return noise / noise.mean() * 0.7