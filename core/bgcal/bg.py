import numpy as np
import torch
from pathlib import Path
from scipy.signal import lombscargle, savgol_filter
from scipy.interpolate import interp1d

class BackgroundCorrector:
    """
    배경(BG) 및 노이즈 프로필을 계산하는 객체지향 클래스.

    MATLAB의 calcBG 및 calcNoise 로직을 기반으로 하며, 데이터 로딩부터
    모든 계산 단계를 포함합니다.
    """
    def __init__(self, info: dict, bin_path: str | Path, device="cuda"):
        """
        Args:
            info (dict): 계산에 필요한 메타데이터.
            bin_path (str or Path): .data 파일의 경로.
            device (str): 연산에 사용할 PyTorch 디바이스.
        """
        self.info = info
        self.bin_path = Path(bin_path)
        self.device = torch.device(device)
        self._load_params()
        print("BackgroundCorrector initialized.")

    def _load_params(self):
        """info 딕셔너리에서 필요한 파라미터를 인스턴스 속성으로 로드합니다."""
        self.ns = int(self.info["numSamples"])
        self.Lbg = int(self.info["bgLines"])
        self.trig_del = int(self.info["trigDelay"])
        self.bg_shift = int(self.info["bgLineShift"])
        
        # usedSamples를 0-based numpy 인덱스로 변환
        used_samples_list = self.info["usedSamples"]
        self.used_idx_np = np.asarray(used_samples_list, dtype=np.int64).ravel() - 1

    def _load_bg_frame(self, trig_delay: int) -> np.ndarray:
        """MATLAB의 Column-major 방식과 동일하게 바이너리 데이터를 로드합니다."""
        with open(self.bin_path, "rb") as fp:
            fp.seek(2 * (trig_delay + self.ns * self.bg_shift))
            raw = np.fromfile(fp, count=self.ns * self.Lbg, dtype=np.uint16)
        
        # MATLAB의 fread([ns, Lbg], ...)는 Column-major이므로,
        # numpy에서 이를 재현하려면 reshape 후 Transpose를 해야 합니다.
        bg_frame = raw.reshape((self.Lbg, self.ns)).T.astype(np.float64)
        return bg_frame[self.used_idx_np, :]

    def _calc_bg(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """배경 평균(bgMean)과 진동 성분(bgOsc)을 계산합니다. (MATLAB calcBG 대응)"""
        print("Calculating background profile (bgMean, bgOsc)...")
        # ✨ 1. 데이터 로드 (dftregistration 로직은 원본에서 비활성화되어 있어 생략)
        trig_delay = self.trig_del
        bg_frame = self._load_bg_frame(trig_delay)
        bg_frame_t = torch.from_numpy(bg_frame).to(self.device, torch.float32)

        # 2. BG 평균 계산
        bg_mean = bg_frame_t.mean(1)

        # 3. 501-tap 평균 필터 (Replicate Padding)
        pad_size = 250
        resid = bg_frame_t - bg_mean[:, None]
        resid_padded = torch.nn.functional.pad(resid.unsqueeze(1), (pad_size, pad_size), mode="replicate")
        ker = torch.ones(1, 1, 2 * pad_size + 1, device=self.device) / (2 * pad_size + 1)
        bg_res = torch.nn.functional.conv1d(resid_padded, ker)[:, 0, :]

        # 4. 주기성 분석 (Lomb-Scargle Periodogram)
        bg_res_sum = bg_res.abs().sum(0).cpu().numpy()
        if np.dot(bg_res_sum, bg_res_sum) < 1e-12:
            return bg_mean, torch.zeros_like(bg_mean), trig_delay

        t = np.arange(self.Lbg, dtype=np.float64)
        freqs = np.fft.rfftfreq(self.Lbg, d=1.0)[1:]
        pxx = lombscargle(t, bg_res_sum, freqs, normalize=False)
        
        if pxx.size == 0 or pxx.max() <= 2 * pxx[0]:
            return bg_mean, torch.zeros_like(bg_mean), trig_delay
        
        # 5. 주 진동 성분(bgOsc) 추출 및 정규화
        T = int(round(1.0 / freqs[np.argmax(pxx)]))
        if T <= 1: return bg_mean, torch.zeros_like(bg_mean), trig_delay
        
        Lcrop = (self.Lbg // T) * T
        if Lcrop == 0: return bg_mean, torch.zeros_like(bg_mean), trig_delay

        osc_fft = torch.fft.fft(bg_res[:, :Lcrop], dim=1)
        bg_osc_complex = osc_fft[:, Lcrop // T]

        val = bg_osc_complex.abs().max().clamp(min=1e-12)
        sum_complex = bg_osc_complex.sum()
        theta = sum_complex / sum_complex.abs().clamp(min=1e-12)
        bg_osc = (bg_osc_complex / val / theta).real

        return bg_mean, bg_osc.to(bg_mean.dtype), trig_delay

    def _calc_noise(self, bg_mean_np: np.ndarray, bg_osc_np: np.ndarray, trig_delay: int) -> torch.Tensor:
        """노이즈 프로필을 계산합니다. (MATLAB calcNoise 대응)"""
        print("Calculating noise profile...")
        # 1. 데이터 로드 및 BG Mean/Osc 제거 (벡터화 연산)
        bg_frame = self._load_bg_frame(trig_delay)
        bg_mean_c = bg_mean_np - 2**15
        a = np.sum((bg_frame - 2**15) * bg_mean_c[:, np.newaxis], axis=0) / np.sum(bg_mean_c**2)
        bg_frame_bs = ((bg_frame - 2**15) / a) - bg_mean_c[:, np.newaxis]
        
        if np.any(bg_osc_np != 0):
            a_osc = np.sum(bg_frame_bs * bg_osc_np[:, np.newaxis], axis=0) / np.sum(bg_osc_np**2)
            bg_frame_bs -= a_osc * bg_osc_np[:, np.newaxis]
        
        # ✨ 2. A-scan 리샘플링 (interp1)
        x_src = np.asarray(self.info["resampTraceA"]).reshape(-1)
        x_tgt = np.linspace(0.0, 1.0, len(self.used_idx_np))
        interp_func = interp1d(x_src, bg_frame_bs, axis=0, kind='cubic', bounds_error=False, fill_value=0.0)
        bg_frame_re = interp_func(x_tgt)

        # 3. 분산 보정 및 스펙트럼 윈도우 적용
        correction = (np.asarray(self.info["dispComp"]) * np.asarray(self.info["spectralWindow"])).astype(np.complex128)
        bg_frame_calib = bg_frame_re * correction.ravel()[:, np.newaxis]

        # 4. IFFT 및 ✨ 표준편차(std) 계산
        img = np.fft.ifft(bg_frame_calib, axis=0)
        std_mag_profile = np.std(np.abs(img), axis=1)

        # ✨ 5. 스무딩 필터 (sgolayfilt)
        smoothed_profile = savgol_filter(std_mag_profile, window_length=51, polyorder=3)
        
        # 6. 최종 정규화
        noise_profile_np = smoothed_profile / np.mean(smoothed_profile) * 0.7
        return torch.from_numpy(noise_profile_np.astype(np.float32)).to(self.device)

    def run(self) -> dict:
        """BG 및 Noise 계산을 순차적으로 실행하고 업데이트된 info 객체를 반환합니다."""
        # 1. BG 계산
        bg_mean, bg_osc, trig_delay = self._calc_bg()
        
        # 2. Noise 계산을 위해 결과를 numpy로 변환하여 전달
        bg_mean_np = bg_mean.cpu().numpy()
        bg_osc_np = bg_osc.cpu().numpy()
        noise_profile = self._calc_noise(bg_mean_np, bg_osc_np, trig_delay)
        
        # 3. info 딕셔너리 업데이트
        self.info['bgMean'] = bg_mean_np
        self.info['bgOsc'] = bg_osc_np
        self.info['trigDelay'] = trig_delay
        self.info['noiseProfile'] = noise_profile.cpu().numpy()
        
        print("Background and noise calculations complete.")
        return self.info