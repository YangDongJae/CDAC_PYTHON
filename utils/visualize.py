# /src/utils/visualize.py
# (기존 파일에 아래 함수 추가)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch 



def segmented_enface(
    slab: np.ndarray,                 # (W, F, Z) intensity
    start_map: np.ndarray,            # (W, F) float indices (1-based 유입 허용)
    end_map: np.ndarray,              # (W, F) float indices
    max_range: float,                 # 최대 누적 범위(픽셀)
    reduce: str = "sum",              # "sum" | "mean" | "max"
) -> np.ndarray:
    """
    MATLAB getSegmentedEnFace2와 동일 목적:
      각 (w,f)에서 start~end 구간(최대 max_range)만 Z축으로 누적/평균/최대.

    - start/end는 실수여도 OK: round 후 clip.
    - 1-based 인덱스가 들어와도 자동 보정.
    """
    W, F, Z = slab.shape
    out = np.zeros((W, F), dtype=np.float32)

    s = np.rint(start_map).astype(np.int32) - 1  # to 0-based
    e = np.rint(end_map).astype(np.int32) - 1

    # max_range 적용
    rng = (e - s + 1)
    rng = np.clip(rng, 1, int(max_range))
    e = s + rng - 1

    # clip to valid Z
    s = np.clip(s, 0, Z - 1)
    e = np.clip(e, 0, Z - 1)

    for w in range(W):
        for f in range(F):
            a, b = int(s[w, f]), int(e[w, f])
            if a > b:  # swap if needed
                a, b = b, a
            seg = slab[w, f, a:b+1]
            if seg.size == 0:
                val = 0.0
            elif reduce == "mean":
                val = float(seg.mean())
            elif reduce == "max":
                val = float(seg.max())
            else:
                val = float(seg.sum())
            out[w, f] = val
    return out


def intensity_to_db(intensity, noise_floor=0.0, eps=1e-12):
    return 10.0 * np.log10(np.clip(intensity, eps, None)) - float(noise_floor)

def db_to_uint16(db_img, dB_range):
    db = np.clip(db_img, 0.0, float(dB_range))
    return (db / float(dB_range) * (2**16 - 1)).astype(np.uint16)

def show_bscan(db_img, dB_range, title=None):
    plt.figure(2, figsize=(8,4)); plt.clf()
    plt.imshow(db_img, cmap="gray", vmin=0, vmax=float(dB_range),
               origin="lower", aspect="auto")
    if title: plt.title(title)
    plt.xlabel("A-line"); plt.ylabel("Depth (px)")
    plt.pause(0.01)        


def save_multipage_tiff(pages_uint16, out_path: str | Path):
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()
    with tiff.TiffWriter(str(out_path), bigtiff=True) as tfile:
        for page in pages_uint16:
            tfile.write(page, contiguous=True)
    return out_path    

def get_segmented_enface2(img: np.ndarray,
                          layer1: np.ndarray,
                          layer2: np.ndarray,
                          t: float | None = None) -> np.ndarray:
    """
    Python port of MATLAB getSegmentedEnFace2 (1:1 동작).
    입력:
      - img: (I, J, K) 실수 영상 (슬랩 스택; 마지막 축이 깊이 K)
      - layer1, layer2: (I, J) 실수 맵 (MATLAB 1-based 인덱스, 소수 포함)
      - t: (옵션) 정규화 스칼라. None이면 median(layer2-layer1+1, omitnan)
    출력:
      - enFace: (I, J) 실수

    MATLAB 코드의 경계/보간 로직과 동일:
      sum(img(ii,jj,ceil(l1):floor(l2)))
      + img(ii,jj,floor(l1))*(ceil(l1)-l1)
      + img(ii,jj,ceil(l2))*(l2-floor(l2))
      후, enFace = enFace / t
    """

    # 검증 및 dtype 통일
    if img.ndim != 3:
        raise ValueError("img must be 3D array (I,J,K).")
    I, J, K = img.shape

    layer1 = np.asarray(layer1, dtype=np.float64)
    layer2 = np.asarray(layer2, dtype=np.float64)

    if layer1.shape != (I, J) or layer2.shape != (I, J):
        raise ValueError("layer1/layer2 must have shape (I,J) matching img[:2].")

    # --- MATLAB 전처리와 동일한 클램프 ---
    # layer1 < 1 -> 1, layer1 > K -> K
    l1 = layer1.copy()
    l2 = layer2.copy()

    # np.nan 비교는 False이므로 NaN은 그대로 유지됨 (MATLAB과 동일한 효과)
    l1[l1 < 1] = 1
    l1[l1 > K] = K

    # layer2 < layer1 -> layer1 값으로 끌어올림
    mask_lt = (l2 < l1)
    l2[mask_lt] = l1[mask_lt]
    # layer2 > K -> K
    l2[l2 > K] = K

    # --- t 계산 (median(layer2 - layer1 + 1, 'omitnan')) ---
    if t is None:
        t = np.nanmedian(l2 - l1 + 1.0)

    enFace = np.zeros((I, J), dtype=np.float64)

    # --- 픽셀별 계산 (MATLAB 루프 그대로) ---
    # 주의: 아래에서 쓰는 인덱스는 1-based(l1/l2) -> 0-based로 변환하여 numpy 접근
    for jj in range(J):
        for ii in range(I):
            a = l1[ii, jj]
            b = l2[ii, jj]

            # NaN이면 0
            if np.isnan(a) or np.isnan(b):
                enFace[ii, jj] = 0.0
                continue

            # 경계 인덱스
            a_floor = int(np.floor(a))           # 1..K
            a_ceil  = int(np.ceil(a))            # 1..K
            b_floor = int(np.floor(b))           # 1..K
            b_ceil  = int(np.ceil(b))            # 1..K

            # 내부 합: k = ceil(a) .. floor(b)  (빈 구간이면 0)
            s = 0.0
            if a_ceil <= b_floor:
                # numpy는 0-based이므로 -1
                s = float(np.sum(img[ii, jj, (a_ceil-1):(b_floor)]))

            # 경계 보간 항
            # img(ii,jj,floor(a)) * (ceil(a) - a)
            s += float(img[ii, jj, a_floor-1]) * (a_ceil - a)

            # img(ii,jj,ceil(b)) * (b - floor(b))
            s += float(img[ii, jj, b_ceil-1]) * (b - b_floor)

            enFace[ii, jj] = s

    # 정규화 (MATLAB: enFace = enFace / t)
    enFace = enFace / t
    return enFace        

def complex_enface_isos_to_mid(
    img_vol: torch.Tensor,            # complex (Z,X,Y)
    ISOS: np.ndarray, RPE: np.ndarray,
    mode: str = "isos_to_mid",
    extra_thick: int = 0
) -> torch.Tensor:
    Z, X, Y = img_vol.shape
    isos = np.clip(np.rint(ISOS).astype(np.int32), 0, Z-1)
    mid  = np.clip(np.rint(0.5*(ISOS + RPE)).astype(np.int32), 0, Z-1)

    if mode != "isos_to_mid":
        raise ValueError("mode must be 'isos_to_mid'")

    z0 = np.minimum(isos, mid) - extra_thick
    z1 = np.maximum(isos, mid) + extra_thick
    z0 = np.clip(z0, 0, Z-1); z1 = np.clip(z1, 0, Z-1)

    span = (z1 - z0 + 1).astype(np.int32)  # (X,Y)
    max_span = int(span.max())

    acc_real = torch.zeros((X, Y), dtype=torch.float32, device=img_vol.device)
    acc_imag = torch.zeros((X, Y), dtype=torch.float32, device=img_vol.device)

    xs = torch.arange(X, device=img_vol.device).view(-1,1).expand(X,Y)
    ys = torch.arange(Y, device=img_vol.device).view(1,-1).expand(X,Y)
    z0_t = torch.from_numpy(z0).to(img_vol.device)
    span_t = torch.from_numpy(span).to(img_vol.device)

    for k in range(max_span):
        zk = z0_t + k
        mask = (k < span_t)
        if not mask.any():
            continue
        vals = img_vol[zk.clamp_max(Z-1), xs, ys]
        acc_real[mask] += vals.real[mask].to(torch.float32)
        acc_imag[mask] += vals.imag[mask].to(torch.float32)

    enface = (acc_real + 1j*acc_imag) / span_t.clamp_min(1)
    return enface    