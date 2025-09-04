# /core/cao/prep.py
from __future__ import annotations
import numpy as np
import torch,  math

def _soft_pupil(H: int, W: int, device, r0=0.98, feather=0.02):
    yy = torch.linspace(-1, 1, H, device=device)
    xx = torch.linspace(-1, 1, W, device=device)
    Yg, Xg = torch.meshgrid(yy, xx, indexing='ij')
    r = torch.sqrt(Xg*Xg + Yg*Yg)
    return torch.clamp((1 - (r - r0)/feather), min=0.0, max=1.0)  # 0..1
    
def center_embed_hwz(x: torch.Tensor, out_hw: tuple[int,int]):
    """
    x: (X, Y, Zs), out_hw: (Hfd, Wfd)
    return: (Hfd, Wfd, Zs), (rs, cs, X, Y)
    """
    assert x.dim() == 3, "x must be (X,Y,Zs)"
    X, Y, Zs = x.shape
    H, W = out_hw
    rs, cs = H//2 - X//2, W//2 - Y//2
    out = torch.zeros((H, W, Zs), dtype=x.dtype, device=x.device)
    out[rs:rs+X, cs:cs+Y, :] = x
    return out, (rs, cs, X, Y)

def center_crop_hwz(x: torch.Tensor, rs_cs_hw: tuple[int,int,int,int]):
    """
    x: (Hfd, Wfd, Zs), rs_cs_hw=(rs,cs,X,Y)
    return: (X, Y, Zs)
    """
    rs, cs, X, Y = rs_cs_hw
    return x[rs:rs+X, cs:cs+Y, :]

def _infer_grid_shape(info: dict) -> tuple[int,int]:
    # 우선 subdivFactors가 있으면 사용
    if "subdivFactors" in info:
        rc = tuple(np.array(info["subdivFactors"], int).ravel())
        assert len(rc) == 2, f"subdivFactors must be 2D, got {info['subdivFactors']}"
        return rc
    # 없으면 validVols나 numVolumes로 추정
    if "validVols" in info:
        vv = np.asarray(info["validVols"])
        if vv.ndim == 2:
            return vv.shape
        return (vv.size, 1)
    return (int(info.get("numVolumes", 1)), 1)

def _ensure_zf_valid_partial(info: dict) -> tuple[int,int]:
    """zfMap/validVols/partialVols를 MATLAB과 동일한 격자 모양으로 정리하고 없으면 초기화."""
    rows, cols = _infer_grid_shape(info)
    shp = (rows, cols)

    # zfMap 준비 (없으면 NaN 초기화)
    if "zfMap" not in info:
        info["zfMap"] = np.full(shp, np.nan, dtype=float)
    else:
        zf = np.asarray(info["zfMap"], dtype=float)
        if zf.size == rows*cols and zf.shape != shp:
            zf = zf.reshape(shp, order="F")
        elif zf.shape != shp:
            zf = np.full(shp, np.nan, dtype=float)
        info["zfMap"] = zf

    # validVols/partialVols 모양 맞추기
    for key, default in (("validVols", True), ("partialVols", False)):
        arr = np.asarray(info.get(key, default)).astype(bool)
        if arr.size == rows*cols and arr.shape != shp:
            arr = arr.reshape(shp, order="F")
        elif arr.shape != shp:
            arr = np.full(shp, default, dtype=bool)
        info[key] = arr

    return shp

def _f_flat(x) -> np.ndarray:
    return np.asarray(x).ravel(order="F")

def _nan_stats_safe(name, arr):
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        print(f"{name:5s}: all-NaN or non-finite")
    else:
        mn = float(np.nanmin(a))
        mx = float(np.nanmax(a))
        nn = int(np.isnan(a).sum())
        print(f"{name:5s}: min={mn:.2f}  max={mx:.2f}  NaN={nn}")

def _ensure_cao_grids(Info: dict):
    """MATLAB과 동일한 격자 크기/형태 보장: zfMap, validVols, partialVols"""
    # subdivFactors = [rows, cols]
    rows, cols = tuple(np.array(Info["subdivFactors"], int).ravel())
    shp = (rows, cols)

    # zfMap
    zf = np.asarray(Info.get("zfMap", np.nan), dtype=float)
    if zf.shape != shp:
        Info["zfMap"] = np.full(shp, np.nan, dtype=float)

    # validVols / partialVols
    for key, default in (("validVols", True), ("partialVols", False)):
        arr = np.asarray(Info.get(key, default), dtype=bool)
        if arr.size == rows*cols and arr.shape != shp:
            Info[key] = arr.reshape(shp, order="F")
        elif arr.shape != shp:
            Info[key] = np.full(shp, default, dtype=bool)
        else:
            Info[key] = arr
    return shp  # (rows, cols)

def _float_pos_inf(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).max) if dtype.is_floating_point else float("inf")

def _float_neg_inf(dtype: torch.dtype) -> float:
    return -float(torch.finfo(dtype).max) if dtype.is_floating_point else -float("inf")

def nanmin_t(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    # PyTorch가 지원하면 그걸 사용
    if hasattr(torch, "nanmin"):
        return torch.nanmin(x, dim=dim, keepdim=keepdim) if dim is not None else torch.nanmin(x)
    # 수동 마스킹
    if dim is None:
        m = torch.isfinite(x)
        return x[m].min() if m.any() else torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
    m = torch.isfinite(x)
    y = x.clone()
    y[~m] = _float_pos_inf(x.dtype)
    v, _ = y.min(dim=dim, keepdim=keepdim)
    all_bad = ~m.any(dim=dim, keepdim=keepdim)
    v = v.masked_fill(all_bad, float("nan"))
    return v

def nanmax_t(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    if hasattr(torch, "nanmax"):
        return torch.nanmax(x, dim=dim, keepdim=keepdim) if dim is not None else torch.nanmax(x)
    if dim is None:
        m = torch.isfinite(x)
        return x[m].max() if m.any() else torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
    m = torch.isfinite(x)
    y = x.clone()
    y[~m] = _float_neg_inf(x.dtype)
    v, _ = y.max(dim=dim, keepdim=keepdim)
    all_bad = ~m.any(dim=dim, keepdim=keepdim)
    v = v.masked_fill(all_bad, float("nan"))
    return v

def nanmean_t(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    if hasattr(torch, "nanmean"):
        return torch.nanmean(x, dim=dim, keepdim=keepdim) if dim is not None else torch.nanmean(x)
    # 수동 평균: 유한값 합 / 유한값 개수
    m = torch.isfinite(x)
    if dim is None:
        denom = m.sum()
        return (x[m].sum() / denom) if denom > 0 else torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
    x0 = torch.where(m, x, torch.zeros((), dtype=x.dtype, device=x.device))
    num = x0.sum(dim=dim, keepdim=keepdim)
    den = m.sum(dim=dim, keepdim=keepdim)
    out = num / den.clamp_min(1)
    out = out.masked_fill(den == 0, float("nan"))
    return out    

def compute_zfmap_and_bounds(
    ILM: np.ndarray, NFL: np.ndarray, ISOS: np.ndarray, RPE: np.ndarray,
    Info: dict, vol: int
):
    """
    MATLAB 블록을 그대로 반영:
    - 유효성 검사(모두 NaN RPE → invalid)
    - valid면: ILMstart/ILMend 계산, zfMap(m,n) = (mean(ISOS)+mean(RPE))/2
    - invalid면: 레이어 NaN + partialVols 거리 가중 평균으로 zfMap(m,n) 보정
    """
    rows, cols = _ensure_cao_grids(Info)
    # MATLAB ind2sub 대응: vol은 선형 인덱스(0-based) 가정
    m, n = np.unravel_index(vol, (rows, cols), order="F")

    # 유효성 체크: 해당 볼륨이 valid인데 RPE가 전부 NaN이면 invalid 처리
    valid_flat = Info["validVols"].ravel(order="F")
    if valid_flat[vol] and np.all(np.isnan(RPE)):
        # 2D 위치로 false 설정
        Info["validVols"][m, n] = False
        valid_flat = Info["validVols"].ravel(order="F")  # 갱신 반영

    if valid_flat[vol]:
        ILMstart = int(np.floor(np.nanmin(ILM)))
        gcl_w = float(Info["GCLprct"][0])
        mix = (NFL * gcl_w + ISOS * (100.0 - gcl_w)) / 100.0
        ILMend = int(np.nanmax(np.ceil(mix)))

        ISOSmean = int(np.round(np.nanmean(ISOS)))
        RPEmean  = int(np.round(np.nanmean(RPE)))
        Info["zfMap"][m, n] = (ISOSmean + RPEmean) / 2.0

        print(f"[prep] vol {vol}: valid, ILM [{ILMstart},{ILMend}], zf={Info['zfMap'][m, n]:.2f}")
        return ILM, NFL, ISOS, RPE, ILMstart, ILMend, Info

    # ---- invalid 분기 ----
    ILM[:] = np.nan; NFL[:] = np.nan; ISOS[:] = np.nan; RPE[:] = np.nan
    ILMstart = int(Info["numImgPixels"])
    ILMend   = int(Info["numImgPixels"])

    volDist = np.full((rows, cols), np.inf, dtype=float)
    pv = Info["partialVols"]
    for volp in np.flatnonzero(pv.ravel(order="F")):
        mp, np_ = np.unravel_index(volp, (rows, cols), order="F")
        dx = Info["centerX"][m] - Info["centerX"][mp]
        dy = Info["centerY"][n] - Info["centerY"][np_]
        volDist[mp, np_] = np.hypot(dx, dy)

    mask = pv & np.isfinite(volDist)
    num = np.nansum( Info["zfMap"][mask] / volDist[mask] )
    den = np.nansum( 1.0 / volDist[mask] )
    Info["zfMap"][m, n] = num / den if den > 0 else np.nan

    print(f"[prep] vol {vol}: invalid → zf interpolated = {Info['zfMap'][m, n]:.2f}")
    return ILM, NFL, ISOS, RPE, ILMstart, ILMend, Info

def _center_crop(x: torch.Tensor, crop_hw, rs_cs_hw):
    rs, cs, h, w = rs_cs_hw
    return x[rs:rs+h, cs:cs+w]

def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))

def ifft2c(X: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X)))

def _center_embed(x: torch.Tensor, out_hw):
    H, W = out_hw
    h, w = x.shape
    rs, cs = H//2 - h//2, W//2 - w//2
    out = torch.zeros((H, W), dtype=x.dtype, device=x.device)
    out[rs:rs+h, cs:cs+w] = x
    return out, (rs, cs, h, w)


def load_layers(pt_path: Path, mat_path: Path|None=None):
    if pt_path.exists():
        d = torch.load(pt_path, map_location="cpu")
        def to_np(x): 
            if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
            return np.asarray(x)
        return to_np(d["ILM"]), to_np(d["NFL"]), to_np(d["ISOS"]), to_np(d["RPE"])
    if mat_path and mat_path.exists():
        md = loadmat(mat_path)
        return md["ILM"], md["NFL"], md["ISOS"], md["RPE"]
    raise FileNotFoundError("Cannot find layer file.")    

@torch.no_grad()
def get_segmented_enface2_intensity(img_xyz: torch.Tensor,
                                     layer1: np.ndarray,
                                     layer2: np.ndarray,
                                     t_override: float | None = None) -> torch.Tensor:
    """
    MATLAB getSegmentedEnFace2와 동일 동작 (intensity 입력, (X,Y,Z) 순서).
    - img_xyz : (X,Y,Zs) 실수(강도) or 복소(강도는 내부에서 처리)
    - layer1/layer2 : (X,Y) float, 1-based 기준을 그대로 넣어도 되고(아래에서 보정),
                      우리는 MATLAB과 동일하게 ceil/floor, 경계분수 가중 포함
    - t_override : 4번째 인자(t)와 동일 (없으면 median(layer2-layer1+1,'omitnan'))
    반환: (X,Y) float
    """
    # 강도로 변환 (복소면 |.|^2)
    if img_xyz.is_complex():
        Ixyz = (img_xyz.real**2 + img_xyz.imag**2).to(torch.float32)
    else:
        Ixyz = img_xyz.to(torch.float32)

    X, Y, Zs = Ixyz.shape
    dev = Ixyz.device

    l1 = torch.as_tensor(layer1, device=dev, dtype=torch.float32)
    l2 = torch.as_tensor(layer2, device=dev, dtype=torch.float32)

    # MATLAB: 1-based 인덱스, 경계 클램프
    l1 = l1.clamp(1, Zs)
    l2 = torch.maximum(l2, l1).clamp(1, Zs)

    # 정수 구간합 + 경계 분수
    # 누적합 계산을 위해 z축 앞에 패딩(0)
    cum = torch.cumsum(Ixyz, dim=2)                             # (X,Y,Zs)
    cum_pad = torch.cat([torch.zeros_like(Ixyz[..., :1]), cum], dim=2)  # (X,Y,Zs+1)

    # 정수 인덱스 (0-based gather용으로 변환)
    l1c = torch.ceil(l1)
    l2f = torch.floor(l2)
    l1c0 = (l1c-1).clamp(0, Zs-1).to(torch.long)                # [0..Zs-1]
    l2f0 = (l2f-1).clamp(0, Zs-1).to(torch.long)

    # 구간합: sum_{ceil(l1)}^{floor(l2)}
    # cum_pad[..., k+1]-cum_pad[..., j] 이므로 gather 인덱스에 +1 필요
    idx_end   = (l2f0 + 1)
    idx_start = (l1c0)
    S_int = cum_pad.gather(2, idx_end.unsqueeze(-1)).squeeze(-1) \
          - cum_pad.gather(2, idx_start.unsqueeze(-1)).squeeze(-1)  # (X,Y)

    # 경계 분수 가중
    w_low  = (l1c - l1)                    # [0,1)
    w_high = (l2  - l2f)                   # [0,1)

    I_low  = Ixyz.gather(2, l1c0.unsqueeze(-1)).squeeze(-1)
    I_high = Ixyz.gather(2, l2f0.unsqueeze(-1)).squeeze(-1)

    enface = S_int + I_low * w_low + I_high * w_high

    # NaN 레이어 픽셀 0 처리
    nan_mask = torch.isnan(l1) | torch.isnan(l2)
    enface = torch.where(nan_mask, torch.zeros_like(enface), enface)

    # 두께 정규화 t
    if t_override is not None:
        t = float(t_override)
    else:
        t_raw = (l2 - l1 + 1.0)
        valid = ~nan_mask
        if valid.any():
            t = float(torch.median(t_raw[valid]).item())
        else:
            t = 1.0
    return enface / (t + 1e-12)                                 # (X,Y) float        

def mat_to_dataframes(file_path):
    """
    Extract data from MATLAB v7.3 file and convert suitable structures to DataFrames
    """
    dataframes = {}  # Dictionary to store all extracted dataframes
    
    try:
        with h5py.File(file_path, 'r') as file:
            # Print file structure first to understand what we're working with
            print("File structure:")
            file.visititems(lambda name, obj: print(f"{'Dataset' if isinstance(obj, h5py.Dataset) else 'Group'}: {name}, "
                                                   f"{'Shape: ' + str(obj.shape) + ', Type: ' + str(obj.dtype) if isinstance(obj, h5py.Dataset) else ''}"))
            
            # Process each top-level element
            print("\nExtracting data to DataFrames:")
            for key in file.keys():
                # Check if it's a dataset directly
                if isinstance(file[key], h5py.Dataset):
                    df = dataset_to_dataframe(file[key], key)
                    if df is not None:
                        dataframes[key] = df
                # If it's a group, try to extract structured data
                else:
                    group_dfs = extract_group_dataframes(file[key], parent_name=key)
                    dataframes.update(group_dfs)
                    
            return dataframes
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def dataset_to_dataframe(dataset, name):
    """Convert an h5py dataset to a pandas DataFrame if possible"""
    try:
        data = dataset[()]
        
        # Handle string data
        if data.dtype.kind in ('S', 'O', 'U'):
            try:
                # Try to convert character arrays to strings
                if len(data.shape) == 2 and min(data.shape) == 1:
                    data = ''.join(chr(c) for c in data.flat)
                    print(f"  Extracted string from {name}: {data}")
                    return None  # Single strings don't need a DataFrame
            except:
                pass  # If conversion fails, continue with normal processing
        
        # For 1D or 2D numeric arrays
        if len(data.shape) <= 2:
            # Convert to DataFrame (handles both 1D and 2D arrays)
            df = pd.DataFrame(data)
            print(f"  Created DataFrame from {name}: {df.shape}")
            return df
            
        else:
            print(f"  Skipping {name}: {len(data.shape)}-dimensional data")
            return None
            
    except Exception as e:
        print(f"  Error converting {name} to DataFrame: {e}")
        return None

def extract_group_dataframes(group, parent_name=""):
    """
    Recursively extract DataFrames from group structure
    Handles common MATLAB structure patterns in HDF5
    """
    dataframes = {}
    
    # For each item in the group
    for key in group.keys():
        full_key = f"{parent_name}/{key}"
        
        # If it's a dataset, try to convert to DataFrame
        if isinstance(group[key], h5py.Dataset):
            df = dataset_to_dataframe(group[key], full_key)
            if df is not None:
                dataframes[full_key] = df
        
        # If it's a group, process recursively
        else:
            sub_dataframes = extract_group_dataframes(group[key], full_key)
            dataframes.update(sub_dataframes)
    
    # Handle special case: MATLAB structure arrays
    # These typically have fields like 'field1', 'field2' that should be combined
    if all(k.startswith(parent_name + "/") for k in dataframes.keys()):
        # Check if we can combine fields into a single DataFrame
        try:
            # Get all direct children fields
            fields = [k.split('/')[-1] for k in dataframes.keys() 
                     if len(k.split('/')) == len(parent_name.split('/')) + 2]
            
            if fields:
                print(f"  Detected potential structure in {parent_name} with fields: {fields}")
                
                # Future enhancement: combine structure fields into a single DataFrame
                # This would require additional logic to align the data properly
        except:
            pass
    
    return dataframes
