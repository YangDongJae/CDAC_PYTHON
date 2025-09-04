# core/preprocessing/cao_zernike.py
import numpy as np
import torch
from math import factorial
from typing import Tuple

# ------------------------------------------------------------------
# single-mode generator ------------------------------------------------
def _calc_zernike_torch(n: int, m: int,
                        R: torch.Tensor, TH: torch.Tensor) -> torch.Tensor:
    """
    Return real-valued Zernike polynomial Z_nm on the unit disk
    using the same normalization as MATLAB zernfun(...,'norm').
    Worked out on GPU or CPU depending on R device.
    """
    m_abs = abs(m)
    radial = torch.zeros_like(R)
    # Radial polynomial R_n^{|m|}
    for s in range((n - m_abs)//2 + 1):
        c = (-1)**s * factorial(n-s) / (
            factorial(s) *
            factorial((n + m_abs)//2 - s) *
            factorial((n - m_abs)//2 - s)
        )
        radial += c * R ** (n - 2*s)
    radial[R > 1] = 0.0  # mask outside unit circle

    if m == 0:
        z = radial
    elif m > 0:
        z = radial * torch.cos(m * TH)
    else:                 # m < 0 (odd)
        z = radial * torch.sin(-m * TH)
    return z


# ------------------------------------------------------------------
# public API -----------------------------------------------------------
def generate_zernike_functions_from_info(info: dict, useCUDA: bool = False) -> Tuple[torch.Tensor, int]:
    """
    Generate Zernike basis cube based on info dictionary.

    Returns
    -------
    z_funcs : torch.Tensor  (N_modes, H_FD, W_FD)
        Zernike functions stored along the first axis (mode-index).
    num_modes : int
        Total number of modes (same as info["CAOzCoeffsN"]).
    """
    if not info.get("CAOenable", 0):
        raise ValueError("CAO module disabled (info['CAOenable'] = 0).")

    # ---------- derived sizes ----------
    H = info["numImgLines"]
    W = info["numImgFrames"]
    pad_factor = info["CAOzeroPadFactor"]
    FD_H, FD_W = H * pad_factor, W * pad_factor
    max_order = info["CAOzMaxOrder"]

    num_modes = (max_order + 4) * (max_order - 1) // 2
    device = torch.device("cuda" if useCUDA and torch.cuda.is_available() else "cpu")

    # ---------- normalized grid ----------
    yy = torch.linspace(-1, 1, FD_H, device=device)
    xx = torch.linspace(-1, 1, FD_W, device=device)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)
    TH = torch.atan2(Y, X)

    # ---------- allocate and fill cube ----------
    z_cube = torch.zeros((num_modes, FD_H, FD_W), device=device, dtype=torch.float32)
    k = 0
    for n in range(2, max_order + 1):
        for m in range(-n, n + 1, 2):
            z_cube[k] = _calc_zernike_torch(n, m, R, TH)
            k += 1

    # ---------- store derived values back to info (optional) ----------
    info["CAOzCoeffsN"] = num_modes
    info["CAOimSize"] = np.array([H, W])
    info["CAOFDsize"] = np.array([FD_H, FD_W])
    info["zFuncs"] = z_cube.cpu().numpy()  # or save torch.Tensor if needed later

    return z_cube, num_modes

def calcZernike_N(n: int, m: int, Info: dict, device=None) -> torch.Tensor:
    """
    MATLAB calcZernike_N(n,m,Info) 대응:
      - 격자: x,y ∈ [-1,1], 크기 = Info.CAOFDsize
      - 단위원판(ρ<=1) 밖은 0
      - zernfun(...,'norm') 동일 정규화(OSA/ANSI)
    반환: (Hfd, Wfd) torch.float32
    """
    Hfd, Wfd = int(Info["CAOFDsize"][0]), int(Info["CAOFDsize"][1])
    device = device or ("cuda" if Info.get("CAOuseCUDA", False) and torch.cuda.is_available() else "cpu")

    # grid (MATLAB meshgrid(x,x)와 동일: y가 행, x가 열)
    y = torch.linspace(-1.0, 1.0, Hfd, device=device)
    x = torch.linspace(-1.0, 1.0, Wfd, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    R = torch.sqrt(X**2 + Y**2)
    TH = torch.atan2(Y, X)

    # ---- radial polynomial R_n^{|m|} ----
    m_abs = abs(int(m))
    if ((n - m_abs) % 2) != 0 or m_abs > n:
        return torch.zeros((Hfd, Wfd), dtype=torch.float32, device=device)

    radial = torch.zeros_like(R, dtype=torch.float32)
    kmax = (n - m_abs) // 2
    # factorial 기반 합 (n<=12 정도면 충분)
    from math import factorial
    for s in range(kmax + 1):
        c = ((-1)**s) * factorial(n - s) / (
            factorial(s) * factorial((n + m_abs)//2 - s) * factorial((n - m_abs)//2 - s)
        )
        radial = radial + c * (R ** (n - 2*s))

    # 각도 부분 (실수 Zernike)
    if m == 0:
        Z = radial
    elif m > 0:
        Z = radial * torch.cos(m * TH)
    else:
        Z = radial * torch.sin(-m * TH)

    # OSA/ANSI 'norm' 정규화: m=0 → √(n+1), m≠0 → √(2(n+1))
    if m == 0:
        Z = torch.sqrt(torch.tensor(n + 1.0, device=device)) * Z
    else:
        Z = torch.sqrt(torch.tensor(2.0 * (n + 1.0), device=device)) * Z

    # 단위원판 밖 0
    Z = torch.where(R <= 1.0, Z, torch.zeros_like(Z))
    return Z.to(torch.float32)    