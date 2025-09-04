# /core/phase_stabilize/stabilize_phase.py

import numpy as np
import torch
import math 
from skimage.registration import phase_cross_correlation
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from pathlib import Path

DEBUG = True
def dbg(msg:str):
    if DEBUG:
        print(msg)

def stabilize_phase(fringes: torch.Tensor, d_file_name: str, vol: int, info: dict):
    """
    Equivalent of MATLAB:
    [fringes, cumPhaseX, cumPhaseY] = stabilizePhase(fringes, dFileName, vol, Info)

    Args:
        fringes (torch.Tensor): Complex tensor of shape (num_samples, num_lines, num_frames)
        d_file_name (str): path to data file (not used here, but passed for consistency)
        vol (int): subvolume index
        info (dict): metadata including:
            - subdivFactors: tuple (Mx, My)
            - centerX, centerY: list or 1D np.ndarray
            - numImgLines, numImgFrames: int
            - ONHmask: 2D numpy array or torch.Tensor

    Returns:
        cum_phase_x (torch.Tensor): (num_lines,)
        cum_phase_y (torch.Tensor): (num_frames,)
        sub_onh_mask (torch.Tensor): (numImgLines, numImgFrames)
    """
    _, num_lines, num_frames = fringes.shape

    # (1) unravel volume index to (m, n) *만약* `vol` 이 1-based(=MATLAB과 동일) 로 넘어온다면 → -1 ,(vol 이 이미 0-based 라면 `vol - 1` 부분을 빼야함)
    m, n = np.unravel_index(vol - 1, info["subdivFactors"], order = 'F')

    # (2) compute line/frame shifts
    line_shift = info["centerX"][m] - round(info["numImgLines"] / 2)
    frame_shift = info["centerY"][n] - round(info["numImgFrames"] / 2)

    # (3) extract sub-region of ONH mask
    onh_mask = info["ONHmask"]
    if isinstance(onh_mask, torch.Tensor):
        sub_onh_mask = onh_mask[
            line_shift : line_shift + int(info["numImgLines"]),
            frame_shift : frame_shift + int(info["numImgFrames"])
        ]
    else:
        sub_onh_mask = torch.from_numpy(onh_mask[
            line_shift : line_shift + int(info["numImgLines"]),
            frame_shift : frame_shift + int(info["numImgFrames"])
        ])

    # (4) initialize cumulative phase containers
    cum_phase_x = torch.zeros(num_lines, dtype=torch.float32)
    cum_phase_y = torch.zeros(num_frames, dtype=torch.float32)

    #DEBUG-------------------------------------------------------
    # dbg(f"line_shift = {line_shift}")
    # dbg(f"frame_shift = {frame_shift}")
    # dbg(f"sub_onh_mask shape = {sub_onh_mask.shape}")
    # dbg(f"cum_phase_x shape = {cum_phase_x.shape}")
    # dbg(f"cum_phase_y shape = {cum_phase_y.shape}")

    # # 확인용 내용 출력 (샘플 값)
    # dbg(f"cum_phase_x sample: {cum_phase_x[:5]}")
    # dbg(f"cum_phase_y sample: {cum_phase_y[:5]}")
    #-------------------------------------------------------CLEAR

    n_refer = info["n"]
    wl_high = info["wlhigh"]
    wl_low = info["wllow"]
    num_used = int(info["numUsedSamples"])
    num_ft   = int(info["numFTSamples"])

    kl = n_refer * 2 * np.pi / wl_high
    kh = n_refer * 2 * np.pi / wl_low
    ke = (kh - kl) / (num_used - 1) * (num_ft - num_used)
    kc = (kh + kl) / 2

    n_lines  = fringes.shape[1]
    n_frames = fringes.shape[2]
    device   = fringes.device
    dtype    = fringes.real.dtype

    if info["FDFlip"]:
        # MATLAB: linspace(-kh, -kl + ke, numFTSamples).'
        kvec = torch.linspace(-kh, -kl + ke, steps=num_ft,
                            device=device, dtype=dtype)
        sign = -1
    else:
        # MATLAB: linspace(kl - ke, kh, numFTSamples).'
        kvec = torch.linspace(kl - ke, kh, steps=num_ft,
                            device=device, dtype=dtype)
        sign = 1

    # MATLAB repmat → PyTorch broadcast
    # (num_ft, 1, 1)  → expand to (num_ft, n_lines, n_frames)
    k0 = kvec[:, None, None].expand(num_ft, n_lines, n_frames)

    #DEBUG-------------------------------------------------------
    # dbg(f"kl = {kl:.10f}")
    # dbg(f"kh = {kh:.10f}")
    # dbg(f"ke = {ke:.10f}")
    # dbg(f"kc = {kc:.10f}")
    # dbg(f"FDFlip = {info.get('FDFlip', False)}")
    # dbg(f"sign = {sign}")
    # dbg(f"k0 shape = {k0.shape}")
    # dbg("k0 sample values (first 5):")
    # dbg(k0[:5, 0, 0])
    #-------------------------------------------------------CLEAR

    mask_has_full_column = sub_onh_mask.bool().all(dim=0).any().item()

    if mask_has_full_column:
        # --------------------- 누적 위상 로드 ---------------------
        cum_phase_x = torch.as_tensor(info["cumPhaseX"], dtype=torch.float32, device=fringes.device)
        cum_phase_y = torch.as_tensor(info["cumPhaseY"], dtype=torch.float32, device=fringes.device)

        # (line_shift, frame_shift 는 앞에서 계산되어 있음)
        x_slice = cum_phase_x[line_shift : line_shift + info["numImgLines"]]            # (L,)
        y_slice = cum_phase_y[frame_shift : frame_shift + info["numImgFrames"]]         # (F,)

        # ------------------- cumPhaseArray 생성 -------------------
        # phase_x: (1, L, 1)  ,  phase_y: (1, 1, F)
        phase_x = x_slice[None, :, None]
        phase_y = y_slice[None, None, :]

        # broadcast → (numFTSamples, L, F)
        cum_phase_array = phase_x + phase_y
        cum_phase_array = cum_phase_array.expand(num_ft, -1, -1)

        # ---------------------- 위상 보정 -------------------------
        # k0   : (numFTSamples, L, F)  (이전 단계에서 생성)
        # sign : ±1                    (이전 단계에서 결정)
        # kc   : 스칼라                (이전 단계에서 계산)
        fringes = fringes * torch.exp((-sign * 1j / kc) * k0 * cum_phase_array)

        # --------------------- .pt 파일 저장 ---------------------
        CACHE_DIR = Path("/home/work/OCT_DL/CDAC_OCT/CDAC_PYTHON/cache")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        pt_path = CACHE_DIR / f"{Path(d_file_name).stem}_vol{vol:02d}_cumPhase.pt"

        torch.save(
            {
                "cumPhaseX": cum_phase_x.cpu(),   # 텐서를 그대로 저장
                "cumPhaseY": cum_phase_y.cpu()
            },
            pt_path
        )

        print(f"⇢ cumPhase saved → {pt_path}")     

    else:
        num_repeats   = int(info["stabilizePhaseRepeats"])
        num_img_pix   = int(info["numImgPixels"])
        noise_factor  = 10 ** (info["noiseFloor"] / 10)

        for rep in range(1, num_repeats + 1):
            print(f"\n Stabilizing Phase {rep}/{num_repeats}", flush=True)

            # --- make intensity image -------------------------------------------
            img     = torch.fft.ifft(fringes, dim=0)          # (k, x, y) → complex
            # dbg(f"ifft image shape: {img.shape}")
            img     = img[:num_img_pix,:,:]                  # (z, x, y) crop
            # dbg(f"cropped img shape: {img.shape}")
            int_img = (img.real**2 + img.imag**2) / noise_factor
            # dbg(f"int_img stats before masking: min={int_img.min().item():.4f}, max={int_img.max().item():.4f}, mean={int_img.mean().item():.4f}, std = {int_img.std().item():.4f}")
            
            # put right after int_img creation, BEFORE NaN mask
            # dbg("\n── INT_IMG DEBUG ─────────────────────────")
            # dbg(f"int_img dtype           : {int_img.dtype}")
            # dbg(f"int_img shape           : {tuple(int_img.shape)}")
            # dbg(f"min / max after scaling : {int_img.min().item():.4e} / {int_img.max().item():.4e}")
            # dbg(f"num zeros (==0)         : {(int_img == 0).sum().item()}")
            # dbg(f"num NaNs                : {torch.isnan(int_img).sum().item()}")

            if sub_onh_mask is not None:          # ⟺  MATLAB’s  nargin == 4
                int_img[:, sub_onh_mask.bool()] = torch.nan
                # dbg("NaN mask applied to sub_ONH_mask region.")

            # dbg(f"int_img shape: {int_img.shape}")
            # dbg("int_img sample slice at z=0:")      

            valid_pixels = int_img[~torch.isnan(int_img)]
            
            mean_val = valid_pixels.mean().item()
            std_val = valid_pixels.std().item()

            filled = torch.nan_to_num(int_img, nan=1.0)
            valid = (filled != 0).any(dim=0)        # → (x,y) bool
            valid = valid.unsqueeze(0).to(torch.float64)

            # after NaN mask has been applied
            # dbg(f"after ONH mask → zeros: {(int_img == 0).sum().item()},  NaNs: {torch.isnan(int_img).sum().item()}")

            # k-축이 전부 0 인 (x,y) column 개수
            all_zero_cols = (~torch.any(int_img != 0, dim=0)).sum().item()
            # dbg(f"columns with ALL-ZERO along k = {all_zero_cols}")

            # ── 3. intImg .* repmat(valid, [num_img_pix 1 1]) ─────────────────────
            masked = int_img * valid.expand_as(int_img)                # broadcast

            # ── 4. mean(mean(...,2,'omitnan'),3,'omitnan') ───────────
            #   • 첫 번째 nanmean: dim=1  (lines) → shape (k, 1, y)
            #   • 두 번째 nanmean: dim=2  (frames)→ shape (k,)
            int_img_avg = torch.nanmean(masked, dim=(1, 2))            # (k,)
            # dbg(f"int_img_avg shape = {int_img_avg.shape}")
            # dbg(f"int_img_avg stats: mean = {int_img_avg.mean():.6f}, std = {int_img_avg.std():.6f}")

            # ── 5. thVal = max(intImgAvg(15/16:end))*2 ───────────────
            tail_start = int(num_img_pix * 15 // 16)                   # 0-based 인덱스
            th_val = int_img_avg[tail_start:].max() * 2
            tail_max = int_img_avg[tail_start:].max()
            # dbg(f"th_val = {th_val:.6f} (from max of last 1/16: {tail_max:.6f})")


            # ── 6. sidx / eidx / thickness  --------------------------
            indices = torch.where(int_img_avg >= th_val)[0]            # 1-D tensor
            if indices.numel():
                sidx = int(indices[0].item())          # MATLAB 'first'
                eidx = int(indices[-1].item())         # MATLAB 'last'
                pidx = eidx
                thickness = eidx - sidx + 1
                # dbg(f"threshold crossed between sidx={sidx} and eidx={eidx}")
                # dbg(f"thickness = {thickness}")
            else:
                sidx = eidx = thickness = None         # or handle as needed
                # dbg("No index exceeds threshold — thickness undefined.")

            depth_per_pix = info["depthPerPixel"]          # [m / pixel]
            max_thick_pix = 180e-3 / depth_per_pix         # 180 µm → 픽셀
            
            while thickness > max_thick_pix and eidx > 0:
                pidx -= 1                                  # MATLAB: pidx = pidx - 1

                if int_img_avg[pidx] > th_val:             # 새로운 최대치 발견
                    th_val = int_img_avg[pidx]
                    sidx   = int(torch.where(int_img_avg >= th_val)[0][0].item())  # 'first'
                    eidx   = pidx

                thickness = pidx - sidx + 1                # 길이 재계산       

            pidx = eidx

            before = torch.where(int_img_avg[:eidx+1] < th_val)[0]   # 0…eidx 포함
            sidx   = int(before[-1].item()) + 1 if before.numel() else 0   # +1 보정

            thickness = pidx - sidx + 1

            # ── while thickness > 60e-3/Info.depthPerPixel && pidx <= numImgPixels ─
            max_thick_pix = 60e-3 / depth_per_pix           # 60 µm → 픽셀 단위

            while thickness > max_thick_pix and pidx <= num_img_pix - 1:
                pidx -= 1                                   # MATLAB: pidx = pidx-1;

                if int_img_avg[pidx] > th_val:              # 새로운 peak 발견
                    th_val = int_img_avg[pidx]

                    # sidx = find(intImgAvg(1:sidx)<thVal,'last')+1;
                    before = torch.where(int_img_avg[:sidx+1] < th_val)[0]
                    sidx   = int(before[-1].item()) + 1 if before.numel() else 0

                    eidx = pidx                             # 업데이트

                thickness = pidx - sidx + 1                 # 길이 재계산

            min_thick_pix = 60e-3 / depth_per_pix          # 60 µm → 픽셀

            while thickness < min_thick_pix and pidx <= num_img_pix - 1:
                pidx += 1                                  # MATLAB: pidx = pidx + 1

                # 새로운 최솟값(th_val) 갱신
                if int_img_avg[pidx] < th_val:
                    th_val = int_img_avg[pidx]

                    # sidx = find(intImgAvg(1:sidx)<thVal,'last') + 1;
                    before = torch.where(int_img_avg[:sidx + 1] < th_val)[0]
                    sidx   = int(before[-1].item()) if before.numel() else 0

                eidx = pidx
                thickness = pidx - sidx + 1

                print(f"\n FINAL thickness = {thickness:d} pixels  ({thickness * depth_per_pix * 1e6:.3f} µm)")

            slab_half_pix = math.ceil(0.1 / depth_per_pix)

            # eidx, sidx : 0-based inclusive indices (이미 계산된 상태라고 가정)
            if (eidx - sidx) > 0:
                ISOS_avg = sidx            # ≈ MATLAB ISOSAvg  (0-based)
                RPE_avg  = eidx            # ≈ MATLAB RPEAvg   (0-based)

                prl_slab_start = max(0,  ISOS_avg - slab_half_pix)
                prl_slab_end   = min(int_img.shape[0] - 1, RPE_avg + slab_half_pix)   # inclusive
            else:
                prl_slab_start = 0
                prl_slab_end   = num_img_pix - 1        # inclusive
            
            
            img_slab  = img[prl_slab_start : prl_slab_end + 1, ...]       # (k', x, y)
            int_img   = img_slab.real**2 + img_slab.imag**2               # |img|²

            k_len, n_x, n_frames = int_img.shape
            int_img_shift = torch.zeros(n_frames - 1, dtype=torch.float64, device=device)

            int_img_curr = torch.fft.fft(int_img[:, :, 0], dim=0)         # (k', x)

            upsample_fac = 128

            for frame in range(1, n_frames):
                int_img_prev = int_img_curr
                int_img_curr = torch.fft.fft(int_img[:, :, frame], dim=0)

                valid = ((int_img_prev != 0).any(dim=0) &
                        (int_img_curr != 0).any(dim=0))                 # 유효 column

                if valid.any():
                    prev_np = int_img_prev[:, valid].cpu().numpy()
                    curr_np = int_img_curr[:, valid].cpu().numpy()

                    # Guizar 알고리즘 – 행(row) 시프트만 필요
                    shift, error, _ = phase_cross_correlation(
                        prev_np, curr_np, 
                        space = "fourier",
                        upsample_factor=upsample_fac, 
                        normalization=None
                    )
                    int_img_shift[frame - 1] = torch.as_tensor(shift[0], dtype=int_img_shift.dtype, device=int_img_shift.device)                   # row shift
                else:
                    int_img_shift[frame - 1] = torch.nan

            # 누적 위상(rad) + Savitzky–Golay 평활
            int_phase = sign * int_img_shift * info["radPerPixel"]               # (N-1,)

            # 창 길이: 2*floor(n_frames/50)+1  → 반드시 홀수
            win_len = 2 * (n_frames // 50) + 1
            int_phase_filt = torch.tensor(
                savgol_filter(int_phase.cpu().numpy(), window_length=win_len, polyorder=3, mode="interp"),
                dtype=torch.float64, device=device
            )

            # Doppler phase aberration
            
            # FAC(k,x,y-1) = img(k,x,f+1) * conj(img(k,x,f))
            fac = img[:, :, 1:] * img[:, :, :-1].conj()

            # angle Σₖ Σₓ FAC → shape (y-1,)
            dopp_phase = torch.angle(fac.sum(dim=(0, 1)))          # torch.float64
            dopp_phase_wrap = dopp_phase.clone()                   # MATLAB doppPhaseWrap   
            
            wsize = 3

            # first unwrap
            delta       = wrap_to_pi(dopp_phase - int_phase_filt)
            dopp_phase  = unwrap2(delta, wsize) + int_phase_filt
            

            # int phase filt
            int_phase_filt -= torch.median(int_phase_filt - dopp_phase)
            delta       = wrap_to_pi(dopp_phase - int_phase_filt)
            dopp_phase  = unwrap2(delta, wsize) + int_phase_filt
            int_phase_filt *= torch.median(dopp_phase / int_phase_filt)
            
            # last unwrap
            delta       = wrap_to_pi(dopp_phase - int_phase_filt)
            dopp_phase  = unwrap2(delta, wsize) + int_phase_filt

            # Accumulated phase vector
            cum_phase = torch.zeros(n_frames, dtype=torch.float64, device=img.device)
            cum_phase[1:] = torch.cumsum(dopp_phase, dim=0) 

            if 'cum_phase_y' not in locals():            # 첫 호출이라면 새로 만들고
                cum_phase_y = torch.zeros_like(cum_phase, device=device)
            else:                                        # 이미 있다면 디바이스만 맞춰 주기
                cum_phase_y = cum_phase_y.to(device)

            cum_phase_y += cum_phase

            # Σₖ FAC  →  shape (x, y-1)
            fac_sum = fac.sum(dim=0)                        # (x, y-1)

            # permute to (y-1, x)  & multiply by exp(-j·doppPhase)
            fac_sum = fac_sum.permute(1, 0) * torch.exp(-1j * dopp_phase)[:, None]

            # Gaussian smoothing (σ = [5, n_frames/4+1])
            sigma_y = 5
            sigma_x = n_frames / 4 + 1
            fac_real = gaussian_filter(fac_sum.real.cpu().numpy(),
                                    sigma=[sigma_y, sigma_x], mode='nearest', truncate = 2.0)
            fac_imag = gaussian_filter(fac_sum.imag.cpu().numpy(),
                                    sigma=[sigma_y, sigma_x], mode='nearest', truncate = 2.0)
            fac_sum  = torch.tensor(fac_real, device=device) \
                    + 1j * torch.tensor(fac_imag, device=device)    # (y-1, x)

            #  phase, accumulated Phase 2-D
            dopp_phase_2d = torch.angle(fac_sum)                     # (y-1, x)

            cum_phase_2d = torch.zeros((n_frames, n_lines),
                                    dtype=torch.float64, device=device)
            cum_phase_2d[1:, :] = torch.cumsum(dopp_phase_2d, dim=0) # (y, x)

            #  cumPhaseArray (k, x, y) 
            cum_phase_array = (cum_phase[:, None] + cum_phase_2d).unsqueeze(0)            # (1, y, x)
            cum_phase_array = cum_phase_array.permute(0, 2, 1)   # (1, x, y)
            cum_phase_array = cum_phase_array.expand(info["numFTSamples"], -1, -1)

            #  apply y-phase correction 
            fringes *= torch.exp((-sign * 1j / kc) * k0 * cum_phase_array)

            #  remake intensity image 
            img = torch.fft.ifft(fringes, dim=0)
            img = img[prl_slab_start : prl_slab_end + 1, ...]        # (k', x, y)

            #  x-ramp correction 
            fac_x = img[:, 1:, :] * img[:, :-1, :].conj()            # (k', x-1, y)

            #DEBUG : 
            print("fac_x shape:", fac_x.shape)
            sum_x = fac_x.sum(dim=(0, 2))                    # → (x-1,)
            print("sum_x shape:", sum_x.shape)
            print("sum_x first 10:", sum_x[:10].cpu().numpy())        

        
            dopp_phase_x = torch.angle(fac_x.sum(dim=(0, 2)))        # (x-1,)
            dopp_phase_x = unwrap2(dopp_phase_x, wsize=3)

            cum_phase_x = torch.zeros(n_lines, dtype=torch.float64, device=device)
            cum_phase_x[1:] = torch.cumsum(dopp_phase_x, dim=0)

            cum_phase_array_x = cum_phase_x[None, :, None].expand(info["numFTSamples"],
                                                                n_lines, n_frames)
            
            if 'cum_phase_x_total' not in locals():
                cum_phase_x_total = torch.zeros_like(cum_phase_x, device=device)
            else:
                cum_phase_x_total = cum_phase_x_total.to(device)

            cum_phase_x_total += cum_phase_x

            fringes *= torch.exp((-sign * 1j / kc) * k0 * cum_phase_array_x)   # (k, x, y)
    
        center_x = np.asarray(info["centerX"]).ravel()   # → 1-D ndarray
        center_y = np.asarray(info["centerY"]).ravel()
        cum_px   = np.asarray(info["cumPhaseX"]).ravel()
        cum_py   = np.asarray(info["cumPhaseY"]).ravel()


        mid_line_idx   = int(round(info["numImgLines"] / 2) - 1)   # 0-based
        ref_x_idx      = int(center_x[m]) - 1                      # 역시 0-based
        cum_phase_x    = cum_phase_x.to(device)                    # (앞서 계산된 벡터)

        cum_phase_x_shift = (
            -cum_phase_x[mid_line_idx] +
            torch.tensor(cum_px[ref_x_idx],
                        dtype=cum_phase_x.dtype, device=device)
        )

        cum_phase_x += cum_phase_x_shift
        fringes     *= torch.exp((-sign * 1j / kc) * k0 * cum_phase_x_shift)



        mid_frame_idx  = int(round(info["numImgFrames"] / 2) - 1)
        ref_y_idx      = int(center_y[n]) - 1                      # 0-based
        cum_phase_y    = cum_phase_y.to(device)

        cum_phase_y_shift = (
            -cum_phase_y[mid_frame_idx] +
            torch.tensor(cum_py[ref_y_idx],
                        dtype=cum_phase_y.dtype, device=device)
        )

        cum_phase_y += cum_phase_y_shift
        fringes     *= torch.exp((-sign * 1j / kc) * k0 * cum_phase_y_shift)

        cache_dir   = Path("/home/work/OCT_DL/CDAC_OCT/CDAC_PYTHON/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        int_pt_path = cache_dir / f"{Path(d_file_name).stem}_Int_{vol:02d}_cumPhase.pt"
        torch.save({"cumPhaseX": cum_phase_x.cpu(),
                    "cumPhaseY": cum_phase_y.cpu()},
                int_pt_path)
        print(f"⇢ phase-shift info saved → {int_pt_path}")


        return (
            int_img_avg,        # MATLAB: intImgAvg
            sidx, eidx, th_val, # PRL slab 파라미터
            int_phase_filt,     # Y 방향 필터링 위상
            dopp_phase,         # 최종 Doppler Y phase
            dopp_phase_x,
            dopp_phase_wrap,    # wrap 되기 전 phase
            fringes,            # 보정 완료된 fringe
            cum_phase_x,        # 누적 위상 X
            cum_phase_y         # 누적 위상 Y
        )


def wrap_to_pi(phase: torch.Tensor) -> torch.Tensor:
    return (phase + math.pi) % (2 * math.pi) - math.pi

def unwrap2(phase: torch.Tensor, wsize: int) -> torch.Tensor:
    N        = phase.numel()
    phaseuw  = torch.zeros_like(phase)            # same dtype / device
    shift    = torch.median(phase[:wsize])

    for ii in range(N - wsize + 1):
        seg         = phase[ii : ii + wsize]
        unwrapped   = wrap_to_pi(seg - shift) + shift
        phaseuw[ii : ii + wsize] = unwrapped       # 중복 영역은 매번 덮어씀
        shift       = torch.median(unwrapped)      # 창마다 shift 갱신

    # 중앙값 기준 오프셋 정규화 (MATLAB 마지막 줄)
    mean_val  = torch.mean(phaseuw)
    phaseuw  -= 2 * math.pi * torch.round(mean_val / (2 * math.pi))
    return phaseuw           

def nanvar(t: torch.Tensor, unbiased=True):
    v = t[~torch.isnan(t)]
    return v.var(unbiased=unbiased)

#Helper 
def rms_adjacent_phase_y(fringes: torch.Tensor) -> float:
    """
    Parameters
    ----------
    fringes : complex64/complex128 tensor, shape (k, x, y)
    Returns
    -------
    float  # radian
    """
    img = torch.fft.ifft(fringes, dim=0)               # (k, x, y)
    fac = img[:, :, 1:] * img[:, :, :-1].conj()        # adjacent frames
    dphi = torch.angle(fac.sum(dim=(0, 1)))            # (y-1,)
    return dphi.std().item()

def rms_adjacent_phase_x(fringes: torch.Tensor) -> float:
    img = torch.fft.ifft(fringes, dim=0)               # (k, x, y)
    fac = img[:, 1:, :] * img[:, :-1, :].conj()        # adjacent lines
    dphi = torch.angle(fac.sum(dim=(0, 2)))            # (x-1,)
    return dphi.std().item()

#DEBUG FUNCTION 

def dbg_stats(name, arr):
    """arr: torch.Tensor or np.ndarray (real/complex)"""
    if isinstance(arr, np.ndarray):
        arr = torch.as_tensor(arr)

    mag   = torch.abs(arr)                  # magnitude
    valid = mag[~torch.isnan(mag)]

    mean = valid.mean().item()
    var  = valid.var(unbiased=True).item()  # N-1  ↔ MATLAB var()

    print(f"{name:<14} | size = {list(arr.shape)} | "
          f"mean = {mean:.6e} | var = {var:.6e}")

#--KAIST---#
import math
from pathlib import Path
import numpy as np
import torch

# optional deps
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.io import savemat
try:
    from skimage.registration import phase_cross_correlation
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# ────────────────────────────── helpers ──────────────────────────────
def _as_1d(a) -> np.ndarray:
    arr = np.asarray(a)
    return arr.reshape(-1)

def _unwrap1(x: np.ndarray) -> np.ndarray:
    """1D unwrap with 2π period (NumPy)."""
    return np.unwrap(x)

def _wrap_to_pi(x: np.ndarray | torch.Tensor):
    """Wrap to (-π, π]."""
    if isinstance(x, torch.Tensor):
        return (x + np.pi) % (2 * np.pi) - np.pi
    x = np.asarray(x)
    return (x + np.pi) % (2 * np.pi) - np.pi

def _unwrap2(x: np.ndarray, wsize: int = 3) -> np.ndarray:
    """
    MATLAB의 unwrap2 대용: 1D unwrap (기간 2π).
    (wsize는 유지용 인자; 실제 구현은 np.unwrap)
    """
    return np.unwrap(x)

def _ensure_center_vol_arrays(Info: dict) -> tuple[np.ndarray, np.ndarray]:
    """centerVolX/Y가 스칼라거나 길이가 안 맞으면 MATLAB 공식을 사용해 재계산."""
    sx, sy = map(int, _as_1d(Info["subdivFactors"])[:2])

    cx = _as_1d(Info.get("centerVolX", []))
    cy = _as_1d(Info.get("centerVolY", []))

    numVolLines  = int(Info.get("numVolLines",  Info["numImgLines"]))
    numVolFrames = int(Info.get("numVolFrames", Info["numImgFrames"]))
    numScanLines  = int(Info["numScanLines"])
    numScanFrames = int(Info["numScanFrames"])

    if cx.size != sx:
        cx = np.rint(np.linspace(numVolLines/2.0,  numScanLines  - numVolLines/2.0,  sx)).astype(np.int64)
    if cy.size != sy:
        cy = np.rint(np.linspace(numVolFrames/2.0, numScanFrames - numVolFrames/2.0, sy)).astype(np.int64)
    return cx, cy

def _phase_corr_row_shift(prev_fft: np.ndarray, curr_fft: np.ndarray, up=128) -> float:
    """
    prev_fft, curr_fft: shape (K, X_valid), complex (Fourier domain)
    반환: 행(축=0) 방향 shift (float)
    """
    if _HAS_SKIMAGE:
        # skimage는 공간/푸리에 모두 지원. 여기서는 fourier domain로 맞춤.
        shift, _, _ = phase_cross_correlation(prev_fft, curr_fft, space="fourier", upsample_factor=up, normalization=None)
        return float(shift[0])
    # 폴백(간단 평균 1D FFT 상호상관, 정수 해상도)
    prev_mean = np.mean(prev_fft, axis=1)
    curr_mean = np.mean(curr_fft, axis=1)
    cps = np.fft.ifft(np.fft.fft(curr_mean) * np.conj(np.fft.fft(prev_mean)))
    idx = int(np.argmax(np.abs(cps)))
    # 중심 기준으로 음/양 방향 해석
    if idx > prev_mean.shape[0] // 2:
        idx = idx - prev_mean.shape[0]
    return float(idx)


# ───────────────────────── stabilizePhase (1:1 재현) ─────────────────────────
# drop-in replacement: 메모리 최적화 버전
@torch.no_grad()
def stabilize_phase_kaist(
    fringes: torch.Tensor,              # (K, L, F) complex64
    Info: dict,
    d_file_name: str | None = None,     # MATLAB 4-input 모드일 때 지정
    vol_1b: int | None = None,          # 1-based volume index
):
    """
    MATLAB stabilizePhase.m 100% 재현 + 메모리 안전 곱셈(결과는 동일).
    항상 아래 11개를 반환합니다:
      (int_img_avg, sidx, eidx, th_val,
       int_phase_filt, dopp_phase, dopp_phase_x, dopp_phase_wrap,
       fringes, cumPhaseX, cumPhaseY)

    ※ 원본 MATLAB에서 ONH mask 적용 라인은 주석 처리되어 있으므로 여기서도 적용하지 않습니다.
    """
    assert fringes.is_complex(), "fringes must be complex tensor of shape (K,L,F)"
    dev = fringes.device
    K, L, F = map(int, fringes.shape)
    dtype_r = fringes.real.dtype  # 보통 float32
    f32 = torch.float32

    # ───────── helpers ─────────
    def _wrap_to_pi(x: torch.Tensor):
        return (x + np.pi) % (2*np.pi) - np.pi

    def _ind2sub_fortran(vol_1b: int, sz: tuple[int,int]) -> tuple[int,int]:
        # MATLAB ind2sub( [sx sy], vol ) with column-major
        sx, sy = sz
        m = (vol_1b - 1) % sx           # 0-based
        n = (vol_1b - 1) // sx          # 0-based
        return m, n


    # in-place 위상 곱 (메모리 절약; 결과는 MATLAB과 동일)
    def _mul_phase_kLF_inplace(frg: torch.Tensor, kvec: torch.Tensor, phi_LF: torch.Tensor,
                               alpha: torch.Tensor, block_k: int = 32):
        # frg: (K,L,F) complex64, kvec:(K,), phi_LF:(L,F) float32, alpha: float32
        K_, L_, F_ = frg.shape
        for s in range(0, K_, block_k):
            e = min(K_, s + block_k)
            theta = (kvec[s:e].view(-1, 1, 1) * alpha) * phi_LF
            rot = torch.polar(torch.ones_like(theta, dtype=f32), theta)      # complex64
            frg[s:e].mul_(rot.to(torch.complex64))
            del theta, rot

    def _mul_phase_kL_inplace_broadcast_F(frg: torch.Tensor, kvec: torch.Tensor, phi_L: torch.Tensor,
                                          alpha: torch.Tensor, block_k: int = 64):
        # phi_L:(L,)를 (blk,L,1)로 만들어 F로 브로드캐스트
        K_, L_, _ = frg.shape
        for s in range(0, K_, block_k):
            e = min(K_, s + block_k)
            theta = (kvec[s:e].view(-1, 1, 1) * alpha) * phi_L.view(1, L_, 1)
            rot = torch.polar(torch.ones_like(theta, dtype=f32), theta)
            frg[s:e].mul_(rot.to(torch.complex64))
            del theta, rot

    def _mul_phase_k_scalar_inplace(frg: torch.Tensor, kvec: torch.Tensor, phi_scalar: torch.Tensor,
                                    alpha: torch.Tensor, block_k: int = 256):
        K_, _, _ = frg.shape
        for s in range(0, K_, block_k):
            e = min(K_, s + block_k)
            theta = (kvec[s:e] * alpha * phi_scalar).view(-1, 1, 1)
            rot = torch.polar(torch.ones_like(theta, dtype=f32), theta)
            frg[s:e].mul_(rot.to(torch.complex64))
            del theta, rot
    # ───────────────────────────

    # ── subdiv, (m,n) ──
    has_4args = (d_file_name is not None) and (vol_1b is not None)
    if has_4args:
        sx, sy = map(int, np.asarray(Info["subdivFactors"]).reshape(-1)[:2])
        m, n = _ind2sub_fortran(vol_1b, (sx, sy))
        centerVolX, centerVolY = Info['centerVolX'] , Info['centerVolY']
        # (아래 lineShift/frameShift는 원본 MATLAB에서 사용 안 하는 구간에 해당하지만 그대로 계산)
        lineShift  = int(round(centerVolX - round(Info["numVolLines"]/2)))
        frameShift = int(round(centerVolY[n] - round(Info["numVolFrames"]/2)))
    else:
        centerVolX = centerVolY = None

    # 초기 누적 위상 벡터 (두 모드 모두 반환을 위해 준비)
    cumPhaseX = torch.zeros(L, dtype=torch.float64, device=dev)
    cumPhaseY = torch.zeros(F, dtype=torch.float64, device=dev)

    # ── k-벡터/상수 ──
    n_refr  = float(Info["n"])
    kl = n_refr * 2.0 * np.pi / float(Info["wlhigh"])
    kh = n_refr * 2.0 * np.pi / float(Info["wllow"])
    numUsed = int(Info["numUsedSamples"]); numFT = int(Info["numFTSamples"])
    ke = (kh - kl) / (numUsed - 1) * (numFT - numUsed)
    kc = (kh + kl) / 2.0

    if bool(Info.get("FDFlip", False)):
        kvec = torch.linspace(-kh, -kl + ke, steps=K, device=dev, dtype=dtype_r); sign = -1.0
    else:
        kvec = torch.linspace(kl - ke, kh, steps=K, device=dev, dtype=dtype_r);  sign =  1.0
    alpha = torch.tensor(-sign / kc, dtype=f32, device=dev)

    # ── 반복 파라미터 ──
    repeats       = int(Info["stabilizePhaseRepeats"])
    numImgPixels  = int(Info["numImgPixels"])
    depthPerPixel = float(Info["depthPerPixel"])
    radPerPixel   = float(Info["radPerPixel"])

    # 리턴용 변수들 (한 번 갱신된 최신 값을 그대로 반환)
    int_img_avg     = None
    sidx = eidx      = None
    th_val          = None
    int_phase_filt  = None
    dopp_phase      = None
    dopp_phase_x    = None
    dopp_phase_wrap = None

    for rep in range(1, repeats + 1):
        # make intensity image
        img = torch.fft.ifft(fringes, dim=0)                      # (K, L, F)
        img = img[:numImgPixels, :, :]
        intImg = (img.real**2 + img.imag**2) / (10.0**(float(Info["noiseFloor"])/10.0))

        # average axial profile (omit zeros)
        valid = (intImg != 0).any(dim=0).to(torch.float64)        # (L, F)
        valid[valid == 0] = float('nan')
        int_img_avg = torch.nanmean(intImg * valid.unsqueeze(0), dim=(1, 2))  # (K',)

        # PRL slab 찾기 (원문 그대로)
        tail_start = int(numImgPixels * 15 // 16)
        th_val = torch.max(int_img_avg[tail_start:])
        idx = torch.where(int_img_avg >= th_val)[0]
        eidx = int(idx[-1].item()); sidx = int(idx[0].item()); pidx = eidx
        thickness = pidx - sidx + 1

        # 180 µm 제한
        while thickness > 180e-3 / depthPerPixel and pidx > 0:
            pidx -= 1
            if int_img_avg[pidx] > th_val:
                th_val = int_img_avg[pidx]
                sidx  = int(torch.where(int_img_avg >= th_val)[0][0].item())
                eidx  = pidx
            thickness = pidx - sidx + 1

        # refine to 60 µm
        pidx = eidx
        before = torch.where(int_img_avg[:eidx+1] < th_val)[0]
        sidx = int(before[-1].item()) + 1 if before.numel() else 0
        thickness = pidx - sidx + 1

        while thickness > 60e-3 / depthPerPixel and pidx <= numImgPixels - 1:
            pidx -= 1
            if int_img_avg[pidx] > th_val:
                th_val = int_img_avg[pidx]
                before = torch.where(int_img_avg[:sidx+1] < th_val)[0]
                sidx   = int(before[-1].item()) + 1 if before.numel() else 0
                eidx   = pidx
            thickness = pidx - sidx + 1

        while thickness < 60e-3 / depthPerPixel and pidx <= numImgPixels - 1:
            pidx += 1
            if int_img_avg[pidx] < th_val:
                th_val = int_img_avg[pidx]
                before = torch.where(int_img_avg[:sidx+1] < th_val)[0]
                sidx = int(before[-1].item()) + 1 if before.numel() else 0
            eidx = pidx
            thickness = pidx - sidx + 1

        # slab 범위
        if (eidx - sidx) > 0:
            slab_half = int(math.ceil(0.1 / depthPerPixel))
            z0 = max(0, sidx - slab_half)
            z1 = min(intImg.shape[0] - 1, eidx + slab_half)
        else:
            z0, z1 = 0, numImgPixels - 1

        # subpixel registration (Y)
        img_slab    = img[z0:z1+1, :, :]                             # (K_s, L, F)
        intImg_slab = img_slab.real**2 + img_slab.imag**2

        shifts = torch.zeros(F - 1, dtype=torch.float64, device=dev)
        IntCurr = torch.fft.fft(intImg_slab[:, :, 0], dim=0)         # (K_s, L)
        from skimage.registration import phase_cross_correlation
        for f in range(1, F):
            IntPrev = IntCurr
            IntCurr = torch.fft.fft(intImg_slab[:, :, f], dim=0)
            valid_cols = ((IntPrev != 0).any(dim=0) & (IntCurr != 0).any(dim=0)).cpu().numpy()
            if np.any(valid_cols):
                prev_np = IntPrev[:, valid_cols].detach().cpu().numpy()
                curr_np = IntCurr[:, valid_cols].detach().cpu().numpy()
                shift, _, _ = phase_cross_correlation(prev_np, curr_np, space="fourier",
                                                      upsample_factor=128, normalization=None)
                shifts[f - 1] = float(shift[0])  # row shift
            else:
                shifts[f - 1] = torch.nan

        int_phase = torch.as_tensor(sign * shifts.cpu().numpy() * radPerPixel, dtype=torch.float64, device=dev)
        win_len = 2 * (img_slab.shape[2] // 50) + 1
        win_len = max(5, win_len | 1)  # 홀수, 최소 5
        int_phase_filt = torch.as_tensor(
            savgol_filter(int_phase.detach().cpu().numpy(), window_length=win_len, polyorder=3, mode="interp"),
            dtype=torch.float64, device=dev
        )

        # y correction
        FAC = img_slab[:, :, 1:] * img_slab[:, :, :-1].conj()        # (K_s, L, F-1)
        dopp_phase = torch.angle(FAC.sum(dim=(0, 1)))                # (F-1,)
        dopp_phase_wrap = dopp_phase.clone()

        delta = _wrap_to_pi(dopp_phase - int_phase_filt)
        dopp_phase = torch.as_tensor(np.unwrap(delta.detach().cpu().numpy()), dtype=torch.float64, device=dev) + int_phase_filt

        int_phase_filt = int_phase_filt - torch.median(int_phase_filt - dopp_phase)
        delta = _wrap_to_pi(dopp_phase - int_phase_filt)
        dopp_phase = torch.as_tensor(np.unwrap(delta.detach().cpu().numpy()), dtype=torch.float64, device=dev) + int_phase_filt

        scale = torch.median(dopp_phase / (int_phase_filt + 1e-12))
        int_phase_filt = int_phase_filt * scale

        delta = _wrap_to_pi(dopp_phase - int_phase_filt)
        dopp_phase = torch.as_tensor(np.unwrap(delta.detach().cpu().numpy()), dtype=torch.float64, device=dev) + int_phase_filt

        cumY_add = torch.zeros(F, dtype=torch.float64, device=dev)
        cumY_add[1:] = torch.cumsum(dopp_phase, dim=0)
        cumPhaseY = cumPhaseY + cumY_add

        # y ramp 2D
        FACsum = FAC.sum(dim=0).permute(1, 0)                        # (F-1, L)
        rotF = torch.polar(torch.ones_like(dopp_phase, dtype=f32), -dopp_phase.float())  # (F-1,)
        FACsum = FACsum.to(torch.complex64) * rotF[:, None].to(torch.complex64)
        fac_r = gaussian_filter(FACsum.real.detach().cpu().numpy(),
                                sigma=[5.0, F/4.0 + 1.0], mode="nearest", truncate=2.0)
        fac_i = gaussian_filter(FACsum.imag.detach().cpu().numpy(),
                                sigma=[5.0, F/4.0 + 1.0], mode="nearest", truncate=2.0)
        FACsum_s = torch.tensor(fac_r, device=dev, dtype=f32) + 1j * torch.tensor(fac_i, device=dev, dtype=f32)
        doppPhase2D = torch.angle(FACsum_s)                            # (F-1, L)
        cumPhase2D  = torch.zeros((F, L), dtype=torch.float32, device=dev)
        cumPhase2D[1:, :] = torch.cumsum(doppPhase2D, dim=0)           # (F, L)

        # apply y correction (블록 곱)
        cum_LF = (cumY_add.float()[:, None] + cumPhase2D).T            # (L, F)
        _mul_phase_kLF_inplace(fringes, kvec.float(), cum_LF, alpha, block_k=32)

        # x correction
        img = torch.fft.ifft(fringes, dim=0)
        img_slab = img[z0:z1+1, :, :]
        FACx = img_slab[:, 1:, :] * img_slab[:, :-1, :].conj()         # (K_s, L-1, F)
        dopp_phase_x = torch.angle(FACx.sum(dim=(0, 2)))               # (L-1,)
        dopp_phase_x = torch.as_tensor(np.unwrap(dopp_phase_x.detach().cpu().numpy()),
                                       dtype=torch.float64, device=dev)

        addX = torch.zeros(L, dtype=torch.float64, device=dev)
        addX[1:] = torch.cumsum(dopp_phase_x, dim=0)
        cumPhaseX = cumPhaseX + addX
        _mul_phase_kL_inplace_broadcast_F(fringes, kvec.float(), addX.float(), alpha, block_k=64)

    # 최종 shift & 저장 (4-input 모드)
    if has_4args:
        numVolLines  = int(Info.get("numVolLines",  Info["numImgLines"]))
        numVolFrames = int(Info.get("numVolFrames", Info["numImgFrames"]))

        # X shift:  -cumPhaseX(round(numVolLines/2)) + Info.cumPhaseX( Info.centerVolX(m) )
        mid_line_idx = int(round(numVolLines / 2.0)) - 1
        ref_x_idx    = int(centerVolX) - 1
        cumX_full = torch.as_tensor(Info["cumPhaseX"], dtype=torch.float64, device=dev)
        mid_line_idx = max(0, min(mid_line_idx, L-1))
        ref_x_idx    = max(0, min(ref_x_idx,    cumX_full.numel()-1))
        shiftX = -cumPhaseX[mid_line_idx] + cumX_full[ref_x_idx]
        cumPhaseX = cumPhaseX + shiftX
        _mul_phase_k_scalar_inplace(fringes, kvec.float(), shiftX.float(), alpha, block_k=256)

        # Y shift:  -cumPhaseY(round(numImgFrames/3/2)) + Info.cumPhaseY( Info.centerVolY(n) )
        # (원본 MATLAB: round(Info.numImgFrames/3/2) → ≈ numImgFrames/6)
        mid_frame_idx = int(round(float(Info["numImgFrames"]) / 3.0 / 2.0)) - 1
        ref_y_idx     = int(centerVolY[n]) - 1
        cumY_full = torch.as_tensor(Info["cumPhaseY"], dtype=torch.float64, device=dev)
        mid_frame_idx = max(0, min(mid_frame_idx, F-1))
        ref_y_idx     = max(0, min(ref_y_idx,    cumY_full.numel()-1))
        shiftY = -cumPhaseY[mid_frame_idx] + cumY_full[ref_y_idx]
        cumPhaseY = cumPhaseY + shiftY
        _mul_phase_k_scalar_inplace(fringes, kvec.float(), shiftY.float(), alpha, block_k=256)

        # 저장 (.mat)
        out_path = f"{d_file_name}_Int_{vol_1b:02d}_cumPhase.pt"
        torch.save({
            "cumPhaseX": cumPhaseX.detach().cpu(), # .numpy() 제거
            "cumPhaseY": cumPhaseY.detach().cpu(), # .numpy() 제거
        }, out_path)

    # 요청한 11개 반환
    return (
        int_img_avg,        # (numImgPixels,)
        sidx, eidx, th_val, # scalars
        int_phase_filt,     # (F-1,)
        dopp_phase,         # (F-1,)
        dopp_phase_x,       # (L-1,)
        dopp_phase_wrap,    # (F-1,)
        fringes,            # (K, L, F) complex64  (보정 완료)
        cumPhaseX,          # (L,)
        cumPhaseY,          # (F,)
    )

# def stabilize_phase_kaist(
#     fringes: torch.Tensor,              # (K, L, F) complex
#     Info: dict,
#     d_file_name: str | None = None,     # 지정 시 MATLAB 4입력 형태로 동작(+저장/반환 변경)
#     vol_1b: int | None = None,          # 1-based
# ):
#     """
#     MATLAB:
#       [fringes, cumPhaseX, cumPhaseY] = stabilizePhase(fringes, dFileName, vol, Info);
#       [fringes, cumPhaseX, cumPhaseY] = stabilizePhase(fringes, Info);  (여기선 cumPhaseX/Y 미반환)

#     Python:
#       d_file_name, vol_1b가 주어지면 cumPhaseX/Y 누적 및 저장/반환 수행.
#       아니면 fringes만 반환.
#     """
#     assert fringes.is_complex(), "fringes must be complex tensor of shape (K, L, F)"

#     dev = fringes.device
#     dtype_r = fringes.real.dtype

#     K, L, F = map(int, fringes.shape)

#     # ── subdiv, m/n (MATLAB ind2sub, Fortran order) ──────────────────
#     has_4args = (d_file_name is not None) and (vol_1b is not None)
#     if has_4args:
#         sx, sy = map(int, _as_1d(Info["subdivFactors"])[:2])
#         m = (vol_1b - 1) % sx        # 0-based
#         n = (vol_1b - 1) // sx
#         centerVolX, centerVolY = _ensure_center_vol_arrays(Info)

#         # line/frame shifts (resampled coordinates)
#         lineShift  = int(round(centerVolX[m] - round(Info["numVolLines"]/2)))
#         frameShift = int(round(centerVolY[n] - round(Info["numVolFrames"]/2)))
#         # 초기 누적 위상
#         cumPhaseX = torch.zeros(L, dtype=torch.float64, device=dev)
#         cumPhaseY = torch.zeros(F, dtype=torch.float64, device=dev)
#     else:
#         # 2-입력 모드 (MATLAB: [fringes, cumPhaseX, cumPhaseY] = stabilizePhase(fringes, Info))
#         cumPhaseX = None
#         cumPhaseY = None

#     # ── k-벡터/상수 ───────────────────────────────────────────────────
#     n_refr  = float(Info["n"])
#     wlhigh  = float(Info["wlhigh"])
#     wllow   = float(Info["wllow"])
#     numUsed = int(Info["numUsedSamples"])
#     numFT   = int(Info["numFTSamples"])

#     kl = n_refr * 2.0 * np.pi / wlhigh
#     kh = n_refr * 2.0 * np.pi / wllow
#     ke = (kh - kl) / (numUsed - 1) * (numFT - numUsed)
#     kc = (kh + kl) / 2.0

#     if bool(Info.get("FDFlip", False)):
#         kvec = torch.linspace(-kh, -kl + ke, steps=K, device=dev, dtype=dtype_r)
#         sign = -1.0
#     else:
#         kvec = torch.linspace(kl - ke, kh, steps=K, device=dev, dtype=dtype_r)
#         sign =  1.0
#     k0 = kvec[:, None, None].expand(K, L, F)  # (K, L, F)

#     # ── 반복 보정 ─────────────────────────────────────────────────────
#     repeats = int(Info["stabilizePhaseRepeats"])
#     numImgPixels = int(Info["numImgPixels"])
#     depthPerPixel = float(Info["depthPerPixel"])
#     radPerPixel   = float(Info["radPerPixel"])

#     for rep in range(1, repeats + 1):
#         # make intensity image
#         img = torch.fft.ifft(fringes, dim=0)                 # (K, L, F) complex
#         img = img[:numImgPixels, :, :]                       # (K', L, F)
#         intImg = (img.real**2 + img.imag**2) / (10.0**(float(Info["noiseFloor"])/10.0))

#         # valid mask & average axial profile (omitnan emulation)
#         valid = (intImg != 0).any(dim=0).to(torch.float64)   # (L, F)
#         valid[valid == 0] = float('nan')
#         masked = intImg * valid.unsqueeze(0)                 # (K', L, F)
#         intImgAvg = torch.nanmean(masked, dim=(1, 2))        # (K',)

#         tail_start = int(numImgPixels * 15 // 16)
#         thVal = torch.max(intImgAvg[tail_start:])            # scalar
#         # s/e/p idx (0-based)
#         indices = torch.where(intImgAvg >= thVal)[0]
#         eidx = int(indices[-1].item())
#         pidx = eidx
#         sidx = int(indices[0].item())
#         thickness = pidx - sidx + 1

#         # while thickness > 180e-3/Info.depthPerPixel && pidx > 0
#         max_thick_pix = 180e-3 / depthPerPixel
#         while thickness > max_thick_pix and pidx > 0:
#             pidx -= 1
#             if intImgAvg[pidx] > thVal:
#                 thVal = intImgAvg[pidx]
#                 sidx  = int(torch.where(intImgAvg >= thVal)[0][0].item())
#                 eidx  = pidx
#             thickness = pidx - sidx + 1

#         # pidx = eidx; sidx = find(intImgAvg(1:eidx)<thVal,'last')+1;
#         pidx = eidx
#         before = torch.where(intImgAvg[:eidx+1] < thVal)[0]
#         sidx = int(before[-1].item()) + 1 if before.numel() else 0
#         thickness = pidx - sidx + 1

#         # while thickness > 60e-3/Info.depthPerPixel && pidx <= Info.numImgPixels
#         max_thick_pix2 = 60e-3 / depthPerPixel
#         while thickness > max_thick_pix2 and pidx <= numImgPixels - 1:
#             pidx -= 1
#             if intImgAvg[pidx] > thVal:
#                 thVal = intImgAvg[pidx]
#                 before = torch.where(intImgAvg[:sidx+1] < thVal)[0]
#                 sidx   = int(before[-1].item()) + 1 if before.numel() else 0
#                 eidx   = pidx
#             thickness = pidx - sidx + 1

#         # while thickness < 60e-3/Info.depthPerPixel && pidx <= Info.numImgPixels
#         while thickness < max_thick_pix2 and pidx <= numImgPixels - 1:
#             pidx += 1
#             if intImgAvg[pidx] < thVal:
#                 thVal = intImgAvg[pidx]
#                 before = torch.where(intImgAvg[:sidx+1] < thVal)[0]
#                 sidx = int(before[-1].item()) + 1 if before.numel() else 0
#             eidx = pidx
#             thickness = pidx - sidx + 1

#         # PRL slab
#         if (eidx - sidx) > 0:
#             ISOSAvg = sidx
#             RPEAvg  = eidx
#             slab_half = int(math.ceil(0.1 / depthPerPixel))
#             PRLSlabStart = max(0, ISOSAvg - slab_half)
#             PRLSlabEnd   = min(intImg.shape[0] - 1, RPEAvg + slab_half)
#         else:
#             PRLSlabStart = 0
#             PRLSlabEnd   = numImgPixels - 1

#         # make intensity image on slab
#         img_slab = img[PRLSlabStart:PRLSlabEnd+1, :, :]        # (K_s, L, F)
#         intImg_slab = img_slab.real**2 + img_slab.imag**2

#         # subpixel registration (row shift between frames)
#         shifts = torch.zeros(F - 1, dtype=torch.float64, device=dev)
#         IntImgCurr = torch.fft.fft(intImg_slab[:, :, 0], dim=0)  # (K_s, L)
#         up = 128
#         for f in range(1, F):
#             IntImgPrev = IntImgCurr
#             IntImgCurr = torch.fft.fft(intImg_slab[:, :, f], dim=0)

#             valid_cols = ((IntImgPrev != 0).any(dim=0) & (IntImgCurr != 0).any(dim=0)).cpu().numpy()
#             if np.any(valid_cols):
#                 prev_np = IntImgPrev[:, valid_cols].detach().cpu().numpy()
#                 curr_np = IntImgCurr[:, valid_cols].detach().cpu().numpy()
#                 shift_row = _phase_corr_row_shift(prev_np, curr_np, up=up)
#                 shifts[f - 1] = shift_row
#             else:
#                 shifts[f - 1] = torch.nan

#         # intPhase / sgolay
#         intPhase = torch.as_tensor(sign * shifts.cpu().numpy() * radPerPixel, dtype=torch.float64, device=dev)
#         win_len = 2 * (intImg_slab.shape[2] // 50) + 1
#         win_len = max(5, win_len | 1)  # odd, >=5
#         intPhaseFilt = torch.as_tensor(
#             savgol_filter(intPhase.detach().cpu().numpy(), window_length=win_len, polyorder=3, mode="interp"),
#             dtype=torch.float64, device=dev
#         )

#         # ── y correction ──────────────────────────────────────────────
#         FAC = img_slab[:, :, 1:] * img_slab[:, :, :-1].conj()        # (K_s, L, F-1)
#         # doppPhase (F-1,)
#         doppPhase = torch.angle(FAC.sum(dim=(0, 1)))                  # (F-1,)
#         doppPhaseWrap = doppPhase.clone()

#         delta = _wrap_to_pi(doppPhase - intPhaseFilt)                 # (F-1,)
#         doppPhase = torch.as_tensor(_unwrap2(delta.detach().cpu().numpy(), 3), dtype=torch.float64, device=dev) + intPhaseFilt

#         intPhaseFilt = intPhaseFilt - torch.median(intPhaseFilt - doppPhase)
#         delta = _wrap_to_pi(doppPhase - intPhaseFilt)
#         doppPhase = torch.as_tensor(_unwrap2(delta.detach().cpu().numpy(), 3), dtype=torch.float64, device=dev) + intPhaseFilt

#         # median scale
#         scale = torch.median(doppPhase / (intPhaseFilt + 1e-12))
#         intPhaseFilt = intPhaseFilt * scale

#         delta = _wrap_to_pi(doppPhase - intPhaseFilt)
#         doppPhase = torch.as_tensor(_unwrap2(delta.detach().cpu().numpy(), 3), dtype=torch.float64, device=dev) + intPhaseFilt

#         # cumPhase (F,)
#         cumPhase = torch.zeros(F, dtype=torch.float64, device=dev)
#         cumPhase[1:] = torch.cumsum(doppPhase, dim=0)

#         if has_4args:
#             cumPhaseY = cumPhaseY + cumPhase

#         # y-ramp correction (2D)
#         FACsum = FAC.sum(dim=0).permute(1, 0)                        # (F-1, L)
#         FACsum = FACsum * torch.exp(-1j * doppPhase)[:, None]        # (F-1, L)
#         # Gaussian smoothing
#         sigma_y = 5.0
#         sigma_x = F / 4.0 + 1.0
#         fac_real = gaussian_filter(FACsum.real.detach().cpu().numpy(), sigma=[sigma_y, sigma_x], mode="nearest", truncate=2.0)
#         fac_imag = gaussian_filter(FACsum.imag.detach().cpu().numpy(), sigma=[sigma_y, sigma_x], mode="nearest", truncate=2.0)
#         FACsum_s = torch.tensor(fac_real, device=dev, dtype=torch.float64) + 1j * torch.tensor(fac_imag, device=dev, dtype=torch.float64)  # (F-1, L)

#         doppPhase2D = torch.angle(FACsum_s)                           # (F-1, L)
#         cumPhase2D = torch.zeros((F, L), dtype=torch.float64, device=dev)
#         cumPhase2D[1:, :] = torch.cumsum(doppPhase2D, dim=0)          # (F, L)

#         # cumPhaseArray: (K, L, F)
#         cumPhaseArray = (cumPhase[:, None] + cumPhase2D)              # (F, L)
#         cumPhaseArray = cumPhaseArray.T.unsqueeze(0).expand(K, L, F)  # (K, L, F)

#         # apply y correction
#         fringes = fringes * torch.exp(torch.as_tensor((-sign * 1j / kc), device=dev) * k0 * cumPhaseArray)

#         # remake intensity image (slab)
#         img = torch.fft.ifft(fringes, dim=0)
#         img_slab = img[PRLSlabStart:PRLSlabEnd+1, :, :]               # (K_s, L, F)

#         # ── x correction ──────────────────────────────────────────────
#         FACx = img_slab[:, 1:, :] * img_slab[:, :-1, :].conj()        # (K_s, L-1, F)
#         doppPhaseX = torch.angle(FACx.sum(dim=(0, 2)))                # (L-1,)
#         doppPhaseX = torch.as_tensor(_unwrap1(doppPhaseX.detach().cpu().numpy()), dtype=torch.float64, device=dev)

#         cumPhaseX_add = torch.zeros(L, dtype=torch.float64, device=dev)
#         cumPhaseX_add[1:] = torch.cumsum(doppPhaseX, dim=0)           # (L,)
#         cumPhaseArrayX = cumPhaseX_add[None, :, None].expand(K, L, F) # (K, L, F)

#         if has_4args:
#             cumPhaseX = cumPhaseX + cumPhaseX_add

#         fringes = fringes * torch.exp(torch.as_tensor((-sign * 1j / kc), device=dev) * k0 * cumPhaseArrayX)

#     # ── 최종 shift 적용 및 저장 (4-입력 모드) ───────────────────────────
#     if has_4args:
#         numVolLines  = int(Info.get("numVolLines",  Info["numImgLines"]))
#         numVolFrames = int(Info.get("numVolFrames", Info["numImgFrames"]))
#         centerVolX, centerVolY = _ensure_center_vol_arrays(Info)

#         # X shift
#         mid_line_idx = int(round(numVolLines / 2.0)) - 1                # 0-based
#         ref_x_idx    = int(centerVolX[m]) - 1                           # 0-based (MATLAB index)
#         cumPhaseX_full = torch.as_tensor(Info["cumPhaseX"], dtype=torch.float64, device=dev)
#         cumPhaseXshift = -cumPhaseX[mid_line_idx] + cumPhaseX_full[ref_x_idx]
#         cumPhaseX = cumPhaseX + cumPhaseXshift
#         fringes = fringes * torch.exp(torch.as_tensor((-sign * 1j / kc), device=dev) * k0 * cumPhaseXshift)

#         # Y shift  (MATLAB: round(Info.numImgFrames/3/2) ⇒ ≈ numImgFrames/6)
#         mid_frame_idx = int(round(float(Info["numImgFrames"]) / 3.0 / 2.0)) - 1
#         ref_y_idx     = int(centerVolY[n]) - 1
#         cumPhaseY_full = torch.as_tensor(Info["cumPhaseY"], dtype=torch.float64, device=dev)
#         # 경계 방어
#         mid_frame_idx = max(0, min(mid_frame_idx, F - 1))
#         ref_y_idx     = max(0, min(ref_y_idx,     cumPhaseY_full.numel()-1))
#         cumPhaseYshift = -cumPhaseY[mid_frame_idx] + cumPhaseY_full[ref_y_idx]
#         cumPhaseY = cumPhaseY + cumPhaseYshift
#         fringes = fringes * torch.exp(torch.as_tensor((-sign * 1j / kc), device=dev) * k0 * cumPhaseYshift)

#         # 저장 (.mat, MATLAB 호환)
#         out_path = f"{d_file_name}_Int_{vol_1b:02d}_cumPhase.mat"
#         savemat(out_path, {
#             "cumPhaseX": cumPhaseX.detach().cpu().numpy(),
#             "cumPhaseY": cumPhaseY.detach().cpu().numpy(),
#         })

#         return fringes, cumPhaseX, cumPhaseY

#     # 2-입력 모드: fringes만 반환
#     return fringes