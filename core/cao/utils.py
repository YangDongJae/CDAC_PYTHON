# matlab_prl_block.py
from __future__ import annotations
import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.utils import nanmin_t, nanmax_t, nanmean_t
from torch import Tensor

# 필요시: from core.enface import getSegmentedEnFace2
# getSegmentedEnFace2(img3d, top_surface, bottom_surface) -> 2D en-face

def prl_block_make_isos_projection(
    img: torch.Tensor,            # (Z, X, Y) complex torch tensor  (MATLAB: img)
    ISOS: np.ndarray,             # (X, Y)
    RPE:  np.ndarray,             # (X, Y)
    Info: Dict[str, Any],
    vol: int,
    dFileName: str,
    getSegmentedEnFace2,          # 함수 핸들 주입
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    MATLAB 코드:

    PRLstart = floor(min(ISOS(:)));
    PRLend   = ceil(max(RPE(:)));
    imgSlab  = permute(img(PRLstart:PRLend, :, :), [2 3 1]);
    ISOSslab = ISOS-PRLstart+1;   RPEslab = RPE-PRLstart+1;

    FACz = imgSlab(:,:,2:end).*conj(imgSlab(:,:,1:end-1));
    phaseSlope = angle(mean(FACz,'all'));
    N = PRLend-PRLstart+1;
    imgSlabComp = imgSlab .* repmat(exp(-1i*phaseSlope * permute(ceil(-N/2):ceil(N/2)-1,[3 1 2])),
                                    [Info.numImgLines, Info.numImgFrames]);
    imgSlabProj = getSegmentedEnFace2(imgSlabComp, ISOSslab, ISOSslab+3*Info.sigmaZ);

    (옵션) PRE en-face 저장 블록 포함
    """
    # ── 조건 체크 (Info.CAOenable & Info.validVols(vol)) ───────────────────
    if not bool(Info.get("CAOenable", False)):
        return None, None

    valid = Info.get("validVols", True)
    is_valid = True
    if isinstance(valid, (list, tuple, np.ndarray)):
        is_valid = bool(valid[vol - 1])    # MATLAB 1-based → Python 0-based
    elif torch.is_tensor(valid):
        is_valid = bool(valid[vol - 1].item())
    else:
        is_valid = bool(valid)

    if not is_valid:
        return None, None

    device = img.device

    # ── 1) PRL 범위 산출 (1:1) ────────────────────────────────────────────
    PRLstart = int(np.floor(np.nanmin(ISOS)))  # MATLAB floor(min(ISOS(:)))
    PRLend   = int(np.ceil(np.nanmax(RPE)))    # MATLAB ceil(max(RPE(:)))
    N = PRLend - PRLstart + 1                  # 포함 범위 길이

    # ── 2) 슬랩 추출 + 차원 재배열 (permute [2 3 1]) ─────────────────────
    # MATLAB: img(PRLstart:PRLend, :, :)  (1-based, end 포함)
    # Python: [start-1 : end]  (0-based, end 미포함)
    slab = img[PRLstart - 1:PRLend, :, :]                # (N, X, Y)
    imgSlab = slab.permute(1, 2, 0).contiguous()         # (X, Y, N) == (numImgLines, numImgFrames, N)

    ISOSslab = ISOS - PRLstart + 1   # 그대로 1-based 좌표계(연산 유지)
    RPEslab  = RPE  - PRLstart + 1

    # ── 3) 위상 경사 보정 (phaseSlope) ────────────────────────────────────
    # FACz = imgSlab(:,:,2:end) .* conj(imgSlab(:,:,1:end-1));
    FACz = imgSlab[:, :, 1:] * torch.conj(imgSlab[:, :, :-1])
    phaseSlope = torch.angle(FACz.mean())  # mean(FACz,'all') → .mean()

    # k = ceil(-N/2) : ceil(N/2)-1  (길이 N), 3축에 배치 → (1,1,N)
    k_start = int(np.ceil(-N / 2.0))
    k_end   = int(np.ceil( N / 2.0))  # 파이썬 arange 상한 제외 → 길이 N 보장
    k = torch.arange(k_start, k_end, dtype=torch.float32, device=device).view(1, 1, -1)

    # imgSlabComp = imgSlab .* repmat(exp(-1i*phaseSlope*k), [numImgLines, numImgFrames])
    phase_ramp = torch.exp(-1j * phaseSlope * k)         # (1,1,N) broadcasting
    imgSlabComp = imgSlab * phase_ramp                   # (X, Y, N), complex

    # ── 4) ISOS 주변 복소 프로젝션 생성 ──────────────────────────────────
    # imgSlabProj = getSegmentedEnFace2(imgSlabComp, ISOSslab, ISOSslab + 3*Info.sigmaZ);
    top    = ISOSslab
    bottom = ISOSslab + 3 * float(Info["sigmaZ"])
    # getSegmentedEnFace2가 numpy를 기대한다면 변환
    imgSlabProj = getSegmentedEnFace2(
        imgSlabComp.detach().cpu().numpy(),
        np.asarray(top, dtype=float),
        np.asarray(bottom, dtype=float),
    )  # (X, Y) 복소 or 실수 (함수 정의에 따름)

    # ── 5) PRL en-face 사전 저장 (옵션) ──────────────────────────────────
    if bool(Info.get("CAOsaveProgress", False)):
        out_path = f"{dFileName}_Int_{vol:02d}_PREnFace.tif"
        if os.path.exists(out_path):
            os.remove(out_path)

        # intImgSlab = real(imgSlab).^2 + imag(imgSlab).^2;
        intImgSlab = (imgSlab.real**2 + imgSlab.imag**2).detach().cpu().numpy()

        # enFace = getSegmentedEnFace2(intImgSlab, ISOSslab, RPEslab);
        enFace = getSegmentedEnFace2(
            intImgSlab.astype(np.float32),
            np.asarray(ISOSslab, dtype=float),
            np.asarray(RPEslab,  dtype=float),
        )  # (X, Y) 실수

        # enFace_uint16 = uint16(nthroot(enFace,4)/10^((Info.dBRange+Info.noiseFloor)/40)*65535);
        scale = 10.0 ** ((float(Info["dBRange"]) + float(Info["noiseFloor"])) / 40.0)
        enFace_u16 = np.clip((np.power(enFace, 0.25) / scale) * 65535.0, 0, 65535).astype(np.uint16)

        # imwrite(enFace_uint16.', ..., 'writemode','append')  → 전치 후 append
        try:
            import tifffile as tiff
            tiff.imwrite(out_path, enFace_u16.T, append=True)
        except Exception:
            # tifffile 미설치 시 단일 페이지 저장 (append 불가)
            from PIL import Image
            Image.fromarray(enFace_u16.T).save(out_path)

    return imgSlabProj, (PRLstart, PRLend)

try:
    from scipy.optimize import differential_evolution
    from scipy.io import loadmat, savemat
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ── helpers: matlab과 동일한 center-embed / crop / fftshift ──────────────
def _fftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(x, dim=(-2, -1))

def _ifftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftshift(x, dim=(-2, -1))

def _mlab_center_range(L_big: int, L_small: int) -> Tuple[int, int]:
    """
    MATLAB 인덱스:
      start1 = floor(L_big/2 - L_small/2 + 1)
      end1   = floor(L_big/2 + L_small/2)      % 포함
    Python slice:
      [start0 : end_excl]  with start0 = start1-1, end_excl = end1
    """
    start1 = int(np.floor(L_big/2 - L_small/2 + 1))
    end1   = int(np.floor(L_big/2 + L_small/2))
    start0 = start1 - 1
    end_excl = end1
    return start0, end_excl

def _center_embed_mlab(small: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    Hfd, Wfd = int(out_hw[0]), int(out_hw[1])
    Him, Wim = int(small.shape[-2]), int(small.shape[-1])
    rs, re = _mlab_center_range(Hfd, Him)
    cs, ce = _mlab_center_range(Wfd, Wim)
    out = torch.zeros((*small.shape[:-2], Hfd, Wfd), dtype=small.dtype, device=small.device)
    out[..., rs:re, cs:ce] = small
    return out

def _center_crop_mlab(big: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    Hfd, Wfd = big.shape[-2], big.shape[-1]
    Him, Wim = int(out_hw[0]), int(out_hw[1])
    rs, re = _mlab_center_range(Hfd, Him)
    cs, ce = _mlab_center_range(Wfd, Wim)
    return big[..., rs:re, cs:ce]

def _resize_2d(x: torch.Tensor, scale: float) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    th, tw = max(1, int(round(h*scale))), max(1, int(round(w*scale)))
    return F.interpolate(x.unsqueeze(0).unsqueeze(0), size=(th, tw),
                         mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

# ── 엔트로피/코스트 ───────────────────────────────────────────────────────
def _entropy_from_complex(img: torch.Tensor) -> torch.Tensor:
    I = (img.real**2 + img.imag**2).clamp_min_(1e-20)
    return -(I * I.log()).sum()

# ── applyZernike (MATLAB 1:1) ─────────────────────────────────────────────
def applyZernike(ImgZP: torch.Tensor, zCoeffs: np.ndarray, Info: Dict[str, Any]) -> torch.Tensor:
    """
    MATLAB applyZernike:
      - φ = Σ zCoeffs(k) * zFuncs(:,:,k)
      - centerGrad 제거 (shift 제거)
      - imgZP = fftshift(ifft2(ifftshift( ImgZP .* exp(-1i*φ) )))
      - 중앙 crop
    """
    device = ImgZP.device
    Hfd, Wfd = int(Info["CAOFDsize"][0]), int(Info["CAOFDsize"][1])
    Him, Wim = int(Info["CAOimSize"][0]), int(Info["CAOimSize"][1])

    zfuncs = Info["zFuncs"]
    if not isinstance(zfuncs, torch.Tensor):
        zfuncs = torch.as_tensor(zfuncs, dtype=torch.float32, device=device)
    else:
        zfuncs = zfuncs.to(device=device, dtype=torch.float32)

    zc = torch.as_tensor(zCoeffs, dtype=torch.float32, device=device)  # (K,)
    # φ = Σ a_k Z_k
    zPhase = torch.tensordot(zc, zfuncs.permute(2, 0, 1), dims=([0], [0]))  # (Hfd,Wfd)

    # centerGrad 제거 (MATLAB 인덱스 정확 대응)
    r0, c0 = Hfd//2, Wfd//2   # 0-based center (floor(H/2))
    center = zPhase[r0, c0]
    gx = center - zPhase[r0-1, c0]
    gy = center - zPhase[r0, c0-1]

    rows = torch.arange(np.ceil(-Hfd/2), np.ceil(Hfd/2), dtype=torch.float32, device=device).view(-1, 1)
    cols = torch.arange(np.ceil(-Wfd/2), np.ceil(Wfd/2), dtype=torch.float32, device=device).view(1, -1)
    zPhase = zPhase - rows * gx - cols * gy

    # 역위상 적용 후 IFFT (두 축에 ifftshift/fftshift)
    imgZP = _fftshift2(torch.fft.ifft2(_ifftshift2(ImgZP * torch.exp(-1j * zPhase))))

    # 중심 crop (MATLAB 인덱스와 동일한 경계식)
    img = _center_crop_mlab(imgZP, (Him, Wim))
    if Info.get("CAOuseCUDA", False):
        # MATLAB gather() 대응이 필요하면 이 단계에서 CPU로 이동
        pass
    return img

# ── surrogateopt 대응 최적화 (SciPy DE로 근사) ───────────────────────────
def evalEntrp(zCoeffs: np.ndarray, ImgSlabProjZP: torch.Tensor, Info: Dict[str, Any]) -> float:
    img = applyZernike(ImgSlabProjZP, zCoeffs, Info)
    H = _entropy_from_complex(img)  # Parseval 가정: 재정규화 불필요
    return float(H.detach().cpu().item())

def optimizeZernike_surrogate(imgSlabProj: torch.Tensor, Info: Dict[str, Any]) -> np.ndarray:
    """
    MATLAB optimizeZernike_surrogate의 파이썬 포팅.
    - 정규화: img / sqrt(mean(|img|^2))
    - 중앙 임베드(0-padding)
    - 2D FFT → fftshift
    - surrogateopt ≈ SciPy differential_evolution로 예산(≈60*K evals) 맞춰 근사
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy(required) not available. Install scipy.")
    device = imgSlabProj.device

    # 정규화
    Imean = (imgSlabProj.real**2 + imgSlabProj.imag**2).mean()
    img_n = imgSlabProj / torch.sqrt(Imean.clamp_min(1e-20))

    Hfd, Wfd = int(Info["CAOFDsize"][0]), int(Info["CAOFDsize"][1])
    Him, Wim = int(Info["CAOimSize"][0]), int(Info["CAOimSize"][1])

    # 중앙 임베드 (MATLAB 인덱스와 동일식)
    imgZP = _center_embed_mlab(img_n, (Hfd, Wfd))

    # 2x oversampled FD image (두 축 ifftshift/fftshift)
    ImgSlabProjZP = _fftshift2(torch.fft.fft2(_ifftshift2(imgZP)))
    if bool(Info.get("CAOuseCUDA", False)):
        ImgSlabProjZP = ImgSlabProjZP.to(device)

    # 경계/예산
    K  = int(Info["CAOzCoeffsN"])
    lb = float(Info["CAOzCoeffsLB"])
    ub = float(Info["CAOzCoeffsUB"])
    bounds = [(lb, ub)] * K

    target_evals = max(60 * K, 200)
    popsize = 10
    maxiter = max(1, int(np.ceil(target_evals / (popsize * K))))

    def _obj(zc_np: np.ndarray) -> float:
        return evalEntrp(zc_np.astype(np.float32), ImgSlabProjZP, Info)

    result = differential_evolution(
        _obj, bounds=bounds, maxiter=maxiter, popsize=popsize,
        tol=0.0, polish=True, updating="deferred",
        workers=1,  # GPU 자원 보호 (필요시 -1로)
        seed=Info.get("seed", None),
        disp=False
    )
    zCoeffs = result.x.astype(np.float32)

    # 출력 포맷 (MATLAB printf와 동일 순회)
    if "CAOzMaxOrder" in Info:
        k = 0
        print("Zernike Coefficients:\n[", end="")
        for n in range(2, int(Info["CAOzMaxOrder"]) + 1):
            for m in range(-n, n - 1, 2):
                print(f"{zCoeffs[k]:.3f}, " if m < n - 1 else f"{zCoeffs[k]:.3f}", end="")
                k += 1
            print("" if n == int(Info["CAOzMaxOrder"]) else "\n", end="")
        print("]")
    return zCoeffs

# ── 최종 CAO 필터 로드/생성 (MATLAB 1:1) ─────────────────────────────────
def build_or_load_cao_filter_surrogate(
    dFileName: str,
    vol: int,
    imgSlabProj: torch.Tensor,     # (Him,Wim) complex
    Info: Dict[str, Any],
) -> torch.Tensor:
    """
    if exist(..._CAO.mat) → load CAOfilter
    else:
      zCoeffs = optimizeZernike_surrogate(...)
      zPhase = Σ zc*Z
      CAOfilter = exp(-1i * ifftshift( imresize(zPhase, 1/zeroPadFactor) ))
      save MAT('CAOfilter')
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy (savemat/loadmat) required.")

    mat_path = f"{dFileName}_Int_{vol:02d}_CAO.mat"
    try:
        md = loadmat(mat_path)
        if "CAOfilter" in md:
            arr = md["CAOfilter"]
            return torch.as_tensor(arr, dtype=torch.complex64)
    except FileNotFoundError:
        pass

    # 1) zCoeffs 최적화
    zCoeffs = optimizeZernike_surrogate(imgSlabProj, Info)

    # 2) zPhase 생성
    device = imgSlabProj.device
    Hfd, Wfd = int(Info["CAOFDsize"][0]), int(Info["CAOFDsize"][1])
    zfuncs = Info["zFuncs"]
    if not isinstance(zfuncs, torch.Tensor):
        zfuncs = torch.as_tensor(zfuncs, dtype=torch.float32, device=device)
    else:
        zfuncs = zfuncs.to(device=device, dtype=torch.float32)
    zc = torch.as_tensor(zCoeffs, dtype=torch.float32, device=device)
    zPhase = torch.tensordot(zc, zfuncs.permute(2, 0, 1), dims=([0], [0]))  # (Hfd,Wfd)

    # 3) imresize(zPhase, 1/zeroPadFactor) + ifftshift + exp(-i·)
    scale = 1.0 / float(Info.get("CAOzeroPadFactor", 1))
    zPhase_small = _resize_2d(zPhase, scale=scale)
    CAOfilter = torch.exp(-1j * _ifftshift2(zPhase_small)).to(torch.complex64)

    # 4) 저장 (변수명 'CAOfilter' 동일)
    savemat(mat_path, {"CAOfilter": CAOfilter.detach().cpu().numpy()})
    return CAOfilter    

def save_prl_projection_after_cao_png(
    img: torch.Tensor,                  # (Z, X, Y) complex torch tensor (MATLAB: img)
    CAOfilter: torch.Tensor,            # (X, Y) complex, ifftshift-domain (MATLAB 저장 형식)
    ISOSslab: np.ndarray,               # (X, Y) 1-based
    RPEslab:  np.ndarray,               # (X, Y) 1-based
    PRLstart: int,
    PRLend:   int,
    Info: Dict[str, Any],
    vol: int,
    dFileName: str,                     # 베이스 이름 (확장자 제외)
    getSegmentedEnFace2: Callable,      # 함수 핸들
    out_dir: Optional[str | Path] = None,
    png_name: str = "PRL_AfterCAO.png",
) -> Path:
    """
    MATLAB:
      imgSlab = ifft2( fft2( permute(img(PRLstart:PRLend,:,:),[2 3 1]) ) .* repmat(CAOfilter,[1 1 N]) );
      intImgSlab = real(imgSlab).^2 + imag(imgSlab).^2;
      enFace = getSegmentedEnFace2(intImgSlab, ISOSslab, RPEslab);
      enFace_uint16 = uint16(nthroot(enFace,4)/10^((Info.dBRange+Info.noiseFloor)/40)*65535);
      imwrite(enFace_uint16.', [dFileName, sprintf('_Int_%02d_PREnFace.tif',vol)], 'writemode','append');

    변경점:
      - TIF append 대신, 폴더를 만들고 .png로 저장
    """
    if not bool(Info.get("CAOsaveProgress", False)):
        # 저장 비활성화면 그냥 빠져나감
        return Path("")

    device = img.device
    X = int(Info["numImgLines"])
    Y = int(Info["numImgFrames"])
    # 1) 슬랩 + permute (Z,X,Y) → (X,Y,N)  (MATLAB [2 3 1])
    #    MATLAB 1-based / end 포함 → 파이썬 0-based / end 미포함
    slab = img[PRLstart - 1:PRLend, :, :]                      # (N, X, Y)
    imgSlab = slab.permute(1, 2, 0).contiguous()               # (X, Y, N)

    # 2) FFT2 * CAOfilter (ifftshift-domain) → IFFT2
    #    PyTorch 기본은 비-시프트 도메인 → MATLAB과 동일
    if CAOfilter.device != device:
        CAOfilter = CAOfilter.to(device)
    if CAOfilter.dtype != imgSlab.dtype:
        CAOfilter = CAOfilter.to(imgSlab.dtype)

    F_slab = torch.fft.fft2(imgSlab, dim=(0, 1))               # (X, Y, N)
    F_slab_corr = F_slab * CAOfilter[..., None]                # (X, Y, N)
    imgSlab_corr = torch.fft.ifft2(F_slab_corr, dim=(0, 1))    # (X, Y, N)

    # 3) 강도 계산
    intImgSlab = (imgSlab_corr.real**2 + imgSlab_corr.imag**2).detach().cpu().numpy()

    # 4) PRL en-face (ISOS~RPE)
    enFace = getSegmentedEnFace2(
        intImgSlab.astype(np.float32),
        np.asarray(ISOSslab, dtype=float),
        np.asarray(RPEslab,  dtype=float),
    )  # (X, Y) real

    # 5) 스케일링 (nthroot(.,4) / 10^((dBRange+noiseFloor)/40) * 65535)
    scale = 10.0 ** ((float(Info["dBRange"]) + float(Info["noiseFloor"])) / 40.0)
    enFace_u16 = np.clip((np.power(enFace, 0.25) / scale) * 65535.0, 0, 65535).astype(np.uint16)

    # 6) 저장 경로 (폴더 + PNG)
    if out_dir is None:
        # 기본: dFileName_Int_##_PRLEnFace/
        base = f"{dFileName}_Int_{vol:02d}_PRLEnFace"
        out_dir = Path(base)
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / png_name
    # MATLAB은 전치 저장(enFace_uint16.'): 시각화 방향을 동일하게 유지하려면 transpose
    Image.fromarray(enFace_u16.T).save(out_png)

    return out_png    

@torch.no_grad()
def make_isos_enface_complex(
    img: torch.Tensor,             # [Z, X, Y] complex64/complex128 (B-scan stack already reconstructed)
    ISOS: np.ndarray | torch.Tensor, # [X, Y] (float; axial indices in 'img' coordinates)
    Info: dict,                    # contains sizes & sigmaZ if available
    z_thickness: int | None = None,# number of z-pixels to project (if None, uses 3*Info["sigmaZ"])
    remove_phase_slope: bool = True
):
    """
    Returns
    -------
    enface_cpx : torch.Tensor [X, Y] complex
        Complex en-face projection around the IS/OS layer.
    debug : dict
        Slab bounds, slope (rad/plane), etc.
    """

    # --- sanitize inputs ---
    assert img.ndim == 3, "img must be [Z,X,Y]"
    if not torch.is_complex(img):
        raise TypeError("img must be a complex tensor [Z,X,Y].")
    device = img.device
    Ztot, X, Y = img.shape

    if isinstance(ISOS, torch.Tensor):
        isos = ISOS.detach().cpu().numpy().astype(np.float32)
    else:
        isos = np.asarray(ISOS, dtype=np.float32)
    assert isos.shape == (X, Y), f"ISOS shape must be ({X},{Y})"

    # thickness (in pixels) to project above ISOS
    if z_thickness is None:
        sigZ = int(Info.get("sigmaZ", 1))
        z_thickness = max(1, int(round(3 * sigZ)))

    # slab bounds in absolute z
    PRLstart = int(np.floor(np.nanmin(isos)))
    PRLend   = int(np.ceil(np.nanmax(isos + z_thickness)))
    PRLstart = max(0, PRLstart)
    PRLend   = min(Ztot, PRLend)
    Nz = PRLend - PRLstart
    if Nz <= 1:
        raise ValueError("Computed slab thickness Nz<=1. Check ISOS and sigmaZ.")

    # slice slab and permute to [X,Y,Z]
    imgSlab = img[PRLstart:PRLend, ...].permute(1, 2, 0).contiguous()  # [X,Y,Nz]

    # local ISOS indices inside the slab
    ISOSslab = torch.from_numpy(isos - PRLstart + 0.0).to(device=device)

    # -------- small inter-slice phase slope removal (FACz estimate) --------
    if remove_phase_slope:
        FACz = imgSlab[..., 1:] * torch.conj(imgSlab[..., :-1])       # [X,Y,Nz-1]
        phaseSlope = torch.angle(FACz.mean())                          # scalar (rad/Δz)
        k = torch.arange(-(Nz//2), -(Nz//2)+Nz, device=device)         # centered ramp
        ramp = torch.exp(-1j * phaseSlope * k)                         # [Nz]
        imgSlab = imgSlab * ramp                                      # broadcast [X,Y,Nz]
    else:
        phaseSlope = torch.tensor(0.0, device=device)

    # -------- segmentation-guided complex projection around ISOS --------
    # Build top/bottom per (x,y) in slab coordinates
    top    = ISOSslab
    bottom = ISOSslab + float(z_thickness)

    zidx = torch.arange(Nz, device=device)[None, None, :]              # [1,1,Nz]
    top_b   = top[..., None].floor()                                   # [X,Y,1]
    bot_b   = bottom[..., None].ceil()
    mask = (zidx >= top_b) & (zidx < bot_b)                            # [X,Y,Nz], boolean

    # complex mean inside [top, bottom)
    w = mask.to(imgSlab.dtype)
    denom = w.sum(dim=-1).clamp_min(1.0)                               # [X,Y]
    enface_cpx = (imgSlab * w).sum(dim=-1) / denom                     # [X,Y] complex

    debug = dict(
        PRLstart=PRLstart, PRLend=PRLend, Nz=Nz,
        z_thickness=z_thickness, phaseSlope=float(phaseSlope.item())
    )
    return enface_cpx, debug@torch.no_grad()
    
def make_isos_enface_complex(
    img: torch.Tensor,             # [Z, X, Y] complex64/complex128 (B-scan stack already reconstructed)
    ISOS: np.ndarray | torch.Tensor, # [X, Y] (float; axial indices in 'img' coordinates)
    Info: dict,                    # contains sizes & sigmaZ if available
    z_thickness: int | None = None,# number of z-pixels to project (if None, uses 3*Info["sigmaZ"])
    remove_phase_slope: bool = True
):
    """
    Returns
    -------
    enface_cpx : torch.Tensor [X, Y] complex
        Complex en-face projection around the IS/OS layer.
    debug : dict
        Slab bounds, slope (rad/plane), etc.
    """

    # --- sanitize inputs ---
    assert img.ndim == 3, "img must be [Z,X,Y]"
    if not torch.is_complex(img):
        raise TypeError("img must be a complex tensor [Z,X,Y].")
    device = img.device
    Ztot, X, Y = img.shape

    if isinstance(ISOS, torch.Tensor):
        isos = ISOS.detach().cpu().numpy().astype(np.float32)
    else:
        isos = np.asarray(ISOS, dtype=np.float32)
    assert isos.shape == (X, Y), f"ISOS shape must be ({X},{Y})"

    # thickness (in pixels) to project above ISOS
    if z_thickness is None:
        sigZ = int(Info.get("sigmaZ", 1))
        z_thickness = max(1, int(round(3 * sigZ)))

    # slab bounds in absolute z
    PRLstart = int(np.floor(np.nanmin(isos)))
    PRLend   = int(np.ceil(np.nanmax(isos + z_thickness)))
    PRLstart = max(0, PRLstart)
    PRLend   = min(Ztot, PRLend)
    Nz = PRLend - PRLstart
    if Nz <= 1:
        raise ValueError("Computed slab thickness Nz<=1. Check ISOS and sigmaZ.")

    # slice slab and permute to [X,Y,Z]
    imgSlab = img[PRLstart:PRLend, ...].permute(1, 2, 0).contiguous()  # [X,Y,Nz]

    # local ISOS indices inside the slab
    ISOSslab = torch.from_numpy(isos - PRLstart + 0.0).to(device=device)

    # -------- small inter-slice phase slope removal (FACz estimate) --------
    if remove_phase_slope:
        FACz = imgSlab[..., 1:] * torch.conj(imgSlab[..., :-1])       # [X,Y,Nz-1]
        phaseSlope = torch.angle(nanmean_t(FACz))                          # scalar (rad/Δz)
        k = torch.arange(-(Nz//2), -(Nz//2)+Nz, device=device)         # centered ramp
        ramp = torch.exp(-1j * phaseSlope * k)                         # [Nz]
        imgSlab = imgSlab * ramp                                      # broadcast [X,Y,Nz]
    else:
        phaseSlope = torch.tensor(0.0, device=device)

    # -------- segmentation-guided complex projection around ISOS --------
    # Build top/bottom per (x,y) in slab coordinates
    top    = ISOSslab
    bottom = ISOSslab + float(z_thickness)

    zidx = torch.arange(Nz, device=device)[None, None, :]              # [1,1,Nz]
    top_b   = top[..., None].floor()                                   # [X,Y,1]
    bot_b   = bottom[..., None].ceil()
    mask = (zidx >= top_b) & (zidx < bot_b)                            # [X,Y,Nz], boolean

    # complex mean inside [top, bottom)
    w = mask.to(imgSlab.dtype)
    denom = w.sum(dim=-1).clamp_min(1.0)                               # [X,Y]
    enface_cpx = (imgSlab * w).sum(dim=-1) / denom                     # [X,Y] complex

    debug = dict(
        PRLstart=PRLstart, PRLend=PRLend, Nz=Nz,
        z_thickness=z_thickness, phaseSlope=float(phaseSlope.item())
    )
    return enface_cpx, debug

@torch.no_grad()
def segmented_enface_with_rephase(
    img_vol: torch.Tensor,                   # (Z,X,Y) complex
    ISOS: np.ndarray | torch.Tensor,         # (X,Y)
    RPE:  np.ndarray | torch.Tensor,         # (X,Y)
    *,
    sigmaZ: float = 1.0,
    layer2_mode: str = "isos_plus_k_sigma",  # "isos_plus_k_sigma"|"to_rpe"|"isos_to_mid"
    k: float = 3.0,
    slab_bounds: tuple[int,int] | None = None,
    clamp_to_slab: bool = True,
    return_meta: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex64,
    # MATLAB 정합/옵션
    strict_matlab: bool = True,              # True: MATLAB getSegmentedEnFace2와 동일한 두께 정규화
    t_override: float | None = None,         # <-- 이름 통일 (기존 nargin 제거)
    # 추가 옵션(전처리 강화)
    roi_mask: torch.Tensor | None = None,    # (X,Y) float 가중 마스크 (SNR/ROI)
    apply_apodize: bool = True,              # lateral apodization(Hann) 적용
    fac_mag_thresh: float = 1e-6,            # FAC 저진폭 픽셀 제외 임계치
):
    assert img_vol.dtype.is_complex, "img_vol must be complex (Z,X,Y)"
    dev = torch.device(device) if device is not None else img_vol.device
    img_vol = img_vol.to(dev, dtype=dtype)

    # to tensor
    ISOS_t = torch.as_tensor(ISOS, device=dev, dtype=torch.float32)
    RPE_t  = torch.as_tensor(RPE,  device=dev, dtype=torch.float32)

    # slab bounds
    Z, X, Y = img_vol.shape
    if slab_bounds is None:
        PRLstart = int(torch.floor(nanmin_t(ISOS_t)).item())
        PRLend   = int(torch.ceil (nanmax_t(RPE_t )).item())
        PRLstart = max(PRLstart, 0)
        PRLend   = min(PRLend,   Z-1)
    else:
        PRLstart, PRLend = slab_bounds
        PRLstart = max(int(PRLstart), 0)
        PRLend   = min(int(PRLend),   Z-1)
    assert PRLend >= PRLstart + 2, f"PRL slab too thin: [{PRLstart},{PRLend}]"

    # --- slab & robust rephase ----------------------------------------------
    imgSlab = img_vol[PRLstart:PRLend+1]          # (Zs,X,Y)
    Zs = imgSlab.shape[0]

    FACz = imgSlab[1:] * torch.conj(imgSlab[:-1])  # (Zs-1,X,Y)

    mag = FACz.abs()
    valid = mag > fac_mag_thresh
    if roi_mask is not None:
        w = roi_mask.to(mag.dtype).clamp_min(0).clamp_max(1).unsqueeze(0).expand_as(mag)
        weighted = torch.where(valid, FACz * w, torch.zeros_like(FACz))
    else:
        weighted = torch.where(valid, FACz, torch.zeros_like(FACz))

    num = weighted.sum()
    if num.abs() == 0:
        phaseSlope = torch.tensor(0.0, device=dev, dtype=imgSlab.real.dtype)
    else:
        phaseSlope = torch.atan2(num.imag, num.real)

    # 일관된 z 인덱스 (짝/홀 무관)
    z = torch.arange(-(Zs//2), Zs-(Zs//2), device=dev, dtype=torch.float32)
    ramp  = torch.exp(-1j * phaseSlope * z).view(Zs,1,1)
    imgSlabComp = imgSlab * ramp

    if apply_apodize:
        wx = torch.hann_window(X, device=dev).view(1, X, 1)
        wy = torch.hann_window(Y, device=dev).view(1, 1, Y)
        imgSlabComp = imgSlabComp * (wx * wy)

    # --- layers in slab coords ----------------------------------------------
    ISOS_raw = ISOS_t - float(PRLstart)
    RPE_raw  = RPE_t  - float(PRLstart)

    if layer2_mode == "isos_plus_k_sigma":
        l1_raw = ISOS_raw
        l2_raw = ISOS_raw + float(k) * float(sigmaZ)
    elif layer2_mode == "to_rpe":
        l1_raw = ISOS_raw
        l2_raw = RPE_raw
    elif layer2_mode == "isos_to_mid":
        l1_raw = ISOS_raw
        l2_raw = 0.5 * (ISOS_raw + RPE_raw)
    else:
        raise ValueError("layer2_mode must be one of {'isos_plus_k_sigma','to_rpe','isos_to_mid'}")

    nan_mask = torch.isnan(l1_raw) | torch.isnan(l2_raw)

    l1 = torch.nan_to_num(l1_raw, nan=0.0)
    l2 = torch.nan_to_num(l2_raw, nan=0.0)
    l2 = torch.maximum(l2, l1)

    if clamp_to_slab:
        l1 = l1.clamp(0, Zs-1)
        l2 = l2.clamp(0, Zs-1)

    # --- variable-thickness complex projection -------------------------------
    cum = torch.cumsum(imgSlabComp, dim=0)                       # (Zs,X,Y)
    cum_pad = torch.cat([torch.zeros_like(imgSlabComp[:1]), cum], dim=0)  # (Zs+1,X,Y)

    start_idx = torch.ceil (l1)
    end_idx   = torch.floor(l2)
    start_idx_cl = start_idx.clamp(0, Zs-1).to(torch.long)
    end_idx_cl   = end_idx  .clamp(0, Zs-1).to(torch.long)

    g_end   = (end_idx_cl + 1).unsqueeze(0)                      # (1,X,Y)
    g_start = (start_idx_cl).unsqueeze(0)                        # (1,X,Y)
    S_int = torch.gather(cum_pad, 0, g_end).squeeze(0) - torch.gather(cum_pad, 0, g_start).squeeze(0)

    w_low  = (torch.ceil(l1) - l1)
    w_high = (l2 - torch.floor(l2))

    l1_floor_cl = torch.floor(l1).clamp(0, Zs-1).to(torch.long)
    l2_ceil_cl  = torch.ceil (l2).clamp(0, Zs-1).to(torch.long)
    I_low  = torch.gather(imgSlabComp, 0, l1_floor_cl.unsqueeze(0)).squeeze(0)
    I_high = torch.gather(imgSlabComp, 0, l2_ceil_cl .unsqueeze(0)).squeeze(0)

    enface_sum = S_int + I_low * w_low + I_high * w_high

    if strict_matlab:
        enface_sum = torch.where(nan_mask, torch.zeros_like(enface_sum), enface_sum)

    # --- thickness normalization --------------------------------------------
    if t_override is not None:
        t_norm = float(t_override)
        enface_cpx = enface_sum / (t_norm + 1e-12)
    else:
        if strict_matlab:
            t_thick_raw = (l2_raw - l1_raw + 1.0)
            valid = ~nan_mask
            if valid.any():
                t_med = torch.median(t_thick_raw[valid])
                t_norm = float(t_med.item())
            else:
                t_norm = 1.0
            enface_cpx = enface_sum / (t_norm + 1e-12)
        else:
            t_exact = (end_idx_cl - start_idx_cl + 1).to(torch.float32) + w_low + w_high
            t_exact = torch.where(nan_mask, torch.ones_like(t_exact), t_exact)
            enface_cpx = enface_sum / (t_exact + 1e-12)

    if return_meta:
        return enface_cpx, {
            "PRLstart": PRLstart, "PRLend": PRLend,
            "phaseSlope": float(phaseSlope.item()),
            "Zs": Zs,
            "layer2_mode": layer2_mode, "k": float(k), "sigmaZ": float(sigmaZ),
            "strict_matlab": bool(strict_matlab),
        }
    return enface_cpx
