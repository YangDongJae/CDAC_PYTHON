import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- 1. 경로 및 설정 ───────────────────────────────────────────────────
ROOT = Path("/home/work/OCT_DL/CDAC_OCT/CDAC_PYTHON")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.utils import load_mat_info
from core.bgcal.bg import BackgroundCorrector # BG/Noise 계산 클래스

# ===================================================================
#   COMPONENT 1: FrameProcessor 클래스 (단일 프레임 처리 전문가)
# ===================================================================
def _to_int(x): 
    try: return int(x)
    except: return int(float(x))

class FrameProcessor:
    """MATLAB의 processFrame.m 로직을 100% 재현하는 객체지향 프레임 처리 클래스."""
    def __init__(self, info: dict, device: str = "cuda"):
        self.info = info
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._load_params()
        self._ensure_centers(self.info)


    def _ensure_centers(self, info: dict) -> None:
        sx, sy = map(_to_int, info.get('subdivFactors', [1, 1]))
        num_img_lines   = _to_int(info.get('numImgLines', 512))
        num_img_frames  = _to_int(info.get('numImgFrames', 512))
        num_scan_lines  = _to_int(info.get('numScanLines', num_img_lines))
        num_scan_frames = _to_int(info.get('numScanFrames', num_img_frames))

        # 우선 centerVolX/Y 있으면 그걸 우선 사용
        cx = info.get('centerX',  info.get('centerVolX', None))
        cy = info.get('centerY',  info.get('centerVolY', None))

        # 정수면 리스트로 감싸기
        if isinstance(cx, (int, float)): cx = [int(cx)]
        if isinstance(cy, (int, float)): cy = [int(cy)]

        # 길이가 subdiv과 맞지 않으면 MATLAB 로직으로 새로 생성
        if not isinstance(cx, (list, tuple)) or len(cx) != sx:
            # round(linspace(NumLines/2, NumScanLines-NumLines/2, sx))
            import numpy as np
            start = num_img_lines/2
            end   = num_scan_lines - num_img_lines/2
            if sx == 1:
                cx = [int(round((start+end)/2))]
            else:
                cx = [int(round(v)) for v in np.linspace(start, end, sx)]

        if not isinstance(cy, (list, tuple)) or len(cy) != sy:
            import numpy as np
            start = num_img_frames/2
            end   = num_scan_frames - num_img_frames/2
            if sy == 1:
                cy = [int(round((start+end)/2))]
            else:
                cy = [int(round(v)) for v in np.linspace(start, end, sy)]

        info['centerX']    = list(map(int, cx))
        info['centerY']    = list(map(int, cy))
        info['centerVolX'] = info['centerX']  # 호환 키도 채워두면 안전
        info['centerVolY'] = info['centerY']

    def _load_params(self):
        """info 딕셔너리에서 자주 사용하는 파라미터를 인스턴스 속성으로 로드합니다."""
        self.used_idx_np = np.asarray(self.info['usedSamples']).ravel() - 1
        self.K = len(self.used_idx_np)
        self.num_samples = int(self.info['numSamples'])
        self.num_img_lines = int(self.info['numImgLines'])
        self.num_scan_lines = int(self.info['numScanLines'])
        self.num_flyback_lines = int(self.info['numFlybackLines'])
        self.num_scan_frames = int(self.info['numScanFrames'])
        self.num_flyback_frames = int(self.info.get('numFlybackFrames', 0))
        self.num_ft_samples = int(self.info['numFTSamples'])
        self.bg_mean = np.asarray(self.info['bgMean'], dtype=np.float64)
        self.bg_osc = np.asarray(self.info.get('bgOsc', 0), dtype=np.float64)
        self.noise_profile = np.asarray(self.info['noiseProfile'], dtype=np.float64)
        self.disp_comp = np.asarray(self.info['dispComp'])
        self.spectral_window = np.asarray(self.info['spectralWindow'])

    def process(self, fid, vol_1b: int, frame_1b: int) -> torch.Tensor:
        """단일 프레임을 처리하는 메인 메서드."""
        frame_shift, line_shift = self._calculate_shifts(vol_1b, frame_1b)
        first_raw, last_raw, num_raw = self._determine_raw_line_window(line_shift)
        if num_raw <= 0: return torch.zeros((self.num_ft_samples, self.num_img_lines), dtype=torch.complex64, device=self.device)
        fringes = self._read_raw_frame(fid, vol_1b, frame_1b, frame_shift, first_raw, num_raw)
        fringes_bs = self._subtract_background(fringes)
        fringes_re_a = self._resample_a_scans(fringes_bs)
        img_calib = self._apply_calibrations(fringes_re_a)
        img_calib = self._apply_band_cutting(img_calib)
        if self.info.get('doResample', False) and num_raw != self.num_img_lines:
            img_calib = self._resample_b_scans(img_calib, first_raw, last_raw, line_shift)
        fringes_calib = np.fft.fft(img_calib, axis=0)
        fringes_result = self._apply_zero_padding(fringes_calib)
        return torch.from_numpy(fringes_result.astype(np.complex64)).to(self.device)

    def _calculate_shifts(self, vol_1b, frame_1b):
        m, n = np.unravel_index(vol_1b - 1, self.info['subdivFactors'], order='F')
        frame_shift = self.info['centerY'][n] - round(self.info['numImgFrames'] / 2)
        motion_idx = frame_shift + frame_1b - 1
        motion_shift = self.info['motionLineShift'][motion_idx] if 0 <= motion_idx < len(self.info['motionLineShift']) else 0
        line_shift = self.info['centerX'][m] - round(self.info['numImgLines'] / 2) - motion_shift
        return int(frame_shift), int(line_shift)

    def _determine_raw_line_window(self, line_shift):
        if not self.info.get('doResample', False): return int(line_shift + 1), 0, int(self.num_img_lines)
        resamp_trace_b = np.asarray(self.info['resampTraceB']); target_start, target_end = line_shift + 1, line_shift + self.num_img_lines
        indices_after_start = np.where(resamp_trace_b > target_start)[0]
        first_raw = indices_after_start[0] if len(indices_after_start) > 0 else 1
        indices_before_end = np.where(resamp_trace_b < target_end)[0]
        last_raw = indices_before_end[-1] + 1 if len(indices_before_end) > 0 else self.num_scan_lines
        num_raw = last_raw - first_raw + 1
        return first_raw, last_raw, num_raw
        
    def _read_raw_frame(self, fid, vol_1b, frame_1b, frame_shift, first_raw, num_raw):
        line_idx = (self.info['initLineShift'] + (frame_shift + frame_1b + (self.num_scan_frames + self.num_flyback_frames) * (vol_1b - 1) - 1) * (self.num_scan_lines + self.num_flyback_lines) + first_raw - 1)
        fid.seek(2 * (self.info['trigDelay'] + self.num_samples * (line_idx - 1)), 0)
        raw = np.fromfile(fid, count=self.num_samples * num_raw, dtype=np.uint16)
        fringes = raw.reshape((num_raw, self.num_samples)).T.astype(np.float64)
        return fringes[self.used_idx_np, :]

    def _subtract_background(self, fringes):
        if self.info.get('adaptiveBG', False):
            fringes_bs = np.zeros_like(fringes); bg_mean_c = self.bg_mean - 2**15; den = np.sum(bg_mean_c**2)
            for i in range(fringes.shape[1]):
                sig_c = fringes[:, i] - 2**15; a = np.sum(sig_c * bg_mean_c) / den; fringes_bs[:, i] = sig_c / a - self.bg_mean + 2**15
        else: fringes_bs = fringes - self.bg_mean[:, np.newaxis]
        if self.info.get('adaptiveBGOsc', False) and np.any(self.bg_osc != 0):
            den_osc = np.sum(self.bg_osc**2)
            for i in range(fringes.shape[1]):
                a_osc = np.sum(fringes_bs[:, i] * self.bg_osc) / den_osc; fringes_bs[:, i] -= a_osc * self.bg_osc
        return fringes_bs

    def _resample_a_scans(self, fringes_bs):
        x_src = np.asarray(self.info["resampTraceA"]).reshape(-1); x_tgt = np.linspace(0, 1, self.K)
        interp_func = interp1d(x_src, fringes_bs, axis=0, kind='cubic', bounds_error=False, fill_value=0.0)
        return interp_func(x_tgt)

    def _apply_calibrations(self, fringes_re_a):
        correction = self.disp_comp * self.spectral_window; fringes_calib = fringes_re_a * correction.ravel()[:, np.newaxis]
        img_calib = np.fft.ifft(fringes_calib, axis=0)
        return img_calib / self.noise_profile[:, np.newaxis]

    def _apply_band_cutting(self, img_calib):
        cut_lo = round(self.info.get('bgBW', 0) * self.K / self.num_ft_samples)
        if cut_lo > 0: img_calib[:cut_lo, :] = 0
        img_calib[int(np.ceil(self.K / 2)) :, :] = 0
        return img_calib
        
    def _resample_b_scans(self, img_calib, first_raw, last_raw, line_shift):
        mag_img = np.abs(img_calib); ph_img = img_calib / (mag_img + 1e-9)
        x_src = np.asarray(self.info['resampTraceB'][first_raw-1 : last_raw])
        x_tgt = np.arange(line_shift + 1, line_shift + self.num_img_lines + 1)
        mag_interp_func = interp1d(x_src, mag_img.T, kind='linear', bounds_error=False, fill_value=0.0)
        mag_resampled = mag_interp_func(x_tgt).T
        ph_interp_func = interp1d(x_src, ph_img.T, kind='linear', bounds_error=False, fill_value=0.0)
        ph_resampled = ph_interp_func(x_tgt).T
        ph_norm = np.abs(ph_resampled); ph_resampled[ph_norm > 0] /= ph_norm[ph_norm > 0]
        return mag_resampled * ph_resampled

    def _apply_zero_padding(self, fringes_calib):
        pad_shape = (self.num_ft_samples - self.K, self.num_img_lines); pad = np.zeros(pad_shape, dtype=np.complex128)
        if self.info.get("FDFlip", False): return np.vstack([fringes_calib, pad])
        else: return np.vstack([pad, fringes_calib])
