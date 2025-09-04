#/core/onh/onh_detect.py

from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import tifffile as tiff  # ✅ 이것이 필요
from skimage.morphology import binary_closing, binary_dilation, disk

def detect_onh_and_volumes_from_info(info: dict,
                                     vols: List[int],
                                     dset_root: Path,
                                     sample_name: str,
                                     useCUDA: bool = False) -> None:
    """
    ONH 마스크 및 valid/partial volume 계산 (info 기반)

    Parameters
    ----------
    info : dict
        OCT 설정 정보 딕셔너리
    vols : List[int]
        Subvolume 번호 목록 (MATLAB 기준 1-based)
    dset_root : Path
        데이터가 위치한 디렉토리
    sample_name : str
        "eye1" 등 샘플 이름
    useCUDA : bool
        GPU 사용 여부
    """

    if not info.get("excludeONH", False):
        m, n = info["subdivFactors"]
        info["ONHmask"] = np.zeros((info["numResampLines"], info["numScanFrames"]), dtype=bool)
        info["validVols"] = np.ones((m, n), dtype=bool)
        info["partialVols"] = np.zeros((m, n), dtype=bool)
        info["zfMap"] = np.full((m, n), np.nan, dtype=np.float32)
        return

    # 1. 3D intensity stack 로드
    tiff_path = dset_root / f"{sample_name}_IntFull.tif"
    stack = tiff.imread(tiff_path)  # (D, H, W)
    stack = (stack.astype(np.float32) / 65535 * info["dBRange"])
    stack = 10 ** (stack / 10)

    # 2. 3D Gaussian blur
    sigmas = (
        info["segmentSigma"][2] / info["depthPerPixel"],
        3 * info["segmentSigma"][0] / info["scanSize"][0] * info["numResampLines"],
        3 * info["segmentSigma"][1] / info["scanSize"][1] * info["numScanFrames"],
    )
    dev = torch.device('cuda' if useCUDA and torch.cuda.is_available() else 'cpu')
    vol_t = torch.from_numpy(stack[None, None]).to(dev)
    vol_blur = _gaussian_blur3d(vol_t, sigmas).squeeze().cpu().numpy()

    # 3. ONH mask 생성
    H, W = vol_blur.shape[1:]
    ONHmask = np.zeros((H, W), dtype=bool)
    thresh_high = 15

    for f in range(vol_blur.shape[0]):
        img = vol_blur[f]
        ILM = _find_edge(img > thresh_high, first=True)
        tmp = img.copy()
        for x in range(W):
            tmp[:ILM[x], x] = thresh_high
        ONL = _find_edge(tmp < thresh_high, first=True)
        for x in range(W):
            tmp[:ONL[x], x] = thresh_high
        ISOS = _find_edge(tmp > thresh_high, first=True)
        ONHmask[:, f] = ISOS == H

    se = disk(int(max(sigmas[1], sigmas[2])))
    ONHmask = binary_dilation(binary_closing(ONHmask, se), se)
    info["ONHmask"] = ONHmask

    # 4. valid / partial volume 계산
    m_fac, n_fac = info["subdivFactors"]
    validVols = torch.zeros((m_fac, n_fac), dtype=torch.bool)
    for vol in vols:
        m, n = np.unravel_index(vol-1, info["subdivFactors"])
        lineShift = int(info["centerX"][m] - info["numImgLines"]/2)
        frameShift = int(info["centerY"][n] - info["numImgFrames"]/2)
        sub = ONHmask[lineShift:lineShift+info["numImgLines"],
                      frameShift:frameShift+info["numImgFrames"]]
        if np.count_nonzero(~sub) / sub.size > 1/8:
            validVols[m, n] = True

    kernel = torch.tensor([[0,1,0],[1,1,1],[0,1,0]], dtype=torch.float32)
    adj = (
        F.conv2d(validVols[None,None].float(), kernel[None,None], padding=1)
        .squeeze()
        .bool()
    )
    partialVols = adj & (~validVols)

    info["validVols"] = validVols.numpy()
    info["partialVols"] = partialVols.numpy()
    info["zfMap"] = np.full((m_fac, n_fac), np.nan, dtype=np.float32)