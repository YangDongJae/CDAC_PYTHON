# core/registration/dftregistration.py
import torch
import math
from typing import Tuple

def _fft_pad_1d(x: torch.Tensor, out_len: int) -> torch.Tensor:
    """1-D FFT계열 zero-pad·crop (DC는 0번 인덱스)."""
    x = torch.fft.fftshift(x)
    pad = torch.zeros(out_len, dtype=x.dtype, device=x.device)
    c_in  = x.shape[0] // 2           # 입력 중앙
    c_out = out_len // 2              # 출력 중앙
    pad[c_out - c_in : c_out + math.ceil(x.shape[0] / 2)] = x
    return torch.fft.ifftshift(pad) * (out_len / x.shape[0])

def _dft_upsample_1d(f: torch.Tensor, n_out: int, usfac: int, roff: int) -> torch.Tensor:
    """행렬-곱 DFT(Guizar 방식) – 작은 구간만 Upsample."""
    n_in  = f.shape[0]
    dev   = f.device
    real_dtype = f.real.dtype

    t = torch.arange(n_out, device=dev, dtype=real_dtype)[:, None]   # (n_out,1)
    n = torch.fft.ifftshift(torch.arange(n_in, device=dev, dtype=real_dtype)) - n_in // 2
    kern  = torch.exp(-1j * 2 * math.pi / (n_in * usfac) * (t - roff).to(f.dtype) @ n[None, :].to(f.dtype))
    return kern @ f

# ---------- 메인 함수 ----------
def dftregistration1d(buf1ft: torch.Tensor,
                      buf2ft: torch.Tensor,
                      usfac: int = 1) -> Tuple[float, float, int]:
    """
    1-D 서브픽셀 정합(Guizar 알고리즘) – Torch/GPU 버전.
    buf1ft, buf2ft : DC 성분이 0번 인덱스(fftshift X).
    반환             : (error, diffphase, net_shift)
    """
    assert buf1ft.shape == buf2ft.shape, "FFT 길이가 달라요."
    N   = buf1ft.shape[0]
    dev, dtype = buf1ft.device, buf1ft.dtype
    Nr  = torch.fft.ifftshift(torch.arange(-N//2, N - N//2, device=dev).to(dtype))

    # -- ① coarse 정합  --------------------------------------------------------
    CC      = torch.fft.ifft(buf1ft.conj() * buf2ft)          # Cross-corr
    rloc    = torch.argmax(torch.abs(CC))                     # 피크 위치
    CCmax   = CC[rloc]
    shift_0 = Nr[rloc]                                        # coarse shift

    # -- ② upsample(refine) ----------------------------------------------------
    if usfac > 1:
        shift_0_r = (shift_0.real * usfac).round() / usfac
        win       = int(math.ceil(usfac * 1.5))               # 작은 창 크기
        roff      = int(win//2)                               # 창 offset
        CC_us     = _dft_upsample_1d(buf2ft.conj()*buf1ft, win, usfac, roff=int(shift_0_r*usfac))
        rloc_us   = torch.argmax(torch.abs(CC_us))
        shift_ref = (shift_0_r + (rloc_us - roff) / usfac).item()
    else:
        shift_ref = shift_0.item()

    # -- ③ 품질 지표 -----------------------------------------------------------
    err   = 1.0 - torch.abs(CCmax) / (torch.linalg.norm(buf1ft) * torch.linalg.norm(buf2ft))
    phase = torch.angle(CCmax).item()

    return float(err), phase, float(shift_ref.real)