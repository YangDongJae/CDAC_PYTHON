import os
import math
from pathlib import Path
import numpy as np
import torch

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation


class PhaseStabilizer:
    """
    OCT 위상 안정화 (KAIST stabilizePhase.m 포팅).
    - CPU/CUDA/float32/float64 어디서든 일관 동작하도록 장치/자료형 관리 강화
    """

    def __init__(self, info: dict):
        self.info = info
        self._initialize_params()
        print(f"PhaseStabilizer initialized: W={self.W_info}, F={self.F_info}, DeviceHint={self.device_hint}")

    # ------------------------------- utils (device/dtype-safe) -------------------------------

    @staticmethod
    def _ensure_odd(n: int) -> int:
        return n if (n % 2 == 1) else (n + 1)

    @staticmethod
    def _nan_interp1d(x: np.ndarray) -> np.ndarray:
        """1D NaN 선형보간 (양 끝은 최근값으로 채움)."""
        x = np.asarray(x, dtype=np.float64)
        mask = ~np.isnan(x)
        if mask.sum() < 2:
            # 데이터가 거의 없으면 0으로 대체
            return np.nan_to_num(x, nan=0.0)
        idx = np.arange(x.size)
        x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
        return x

    @staticmethod
    def _savgol_safe(x_np: np.ndarray, polyorder: int = 3, approx_seg: int = 50) -> np.ndarray:
        """
        Savitzky–Golay를 안전하게 호출:
        - NaN은 먼저 선형보간으로 제거
        - window_length는 홀수, polyorder보다 크게, len(x) 이하
        """
        x_np = PhaseStabilizer._nan_interp1d(x_np)
        n = int(x_np.shape[0])
        if n <= polyorder + 2:
            return x_np.copy()

        win = 2 * max(1, n // approx_seg) + 1  # 홀수
        min_ok = polyorder + 2                 # polyorder < win
        if min_ok % 2 == 0:
            min_ok += 1
        win = max(win, min_ok)

        max_ok = n if (n % 2 == 1) else (n - 1)
        win = min(win, max_ok)

        if win <= polyorder or win < 3:
            return x_np.copy()

        return savgol_filter(x_np, window_length=win, polyorder=polyorder, mode="interp")

    @staticmethod
    def _wrap_to_pi(x):
        """torch/np 모두 지원, x의 device/dtype을 따름."""
        if isinstance(x, torch.Tensor):
            pi = x.new_tensor(math.pi)
            two_pi = pi * 2
            return torch.remainder(x + pi, two_pi) - pi
        x_np = np.asarray(x, dtype=np.float64)
        return (x_np + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _unwrap2(vec: torch.Tensor, wsize: int = 3) -> torch.Tensor:
        """
        MATLAB unwrap2 재현(토치 네이티브).
        - vec: 1D float tensor (device/dtype 유지)
        """
        assert vec.ndim == 1, "unwrap2 expects 1D tensor"
        device, dtype = vec.device, vec.dtype
        N = vec.numel()
        out = torch.zeros_like(vec)
        if N == 0:
            return out

        pi = vec.new_tensor(math.pi)
        two_pi = pi * 2

        def wrap_to_pi_t(y):
            return torch.remainder(y + pi, two_pi) - pi

        w = max(1, int(wsize))
        w = min(w, N)
        # 초기 shift
        shift = torch.median(vec[:w])

        # 슬라이딩 윈도우
        i = 0
        while i <= N - w:
            seg = vec[i:i + w]
            seg_uw = wrap_to_pi_t(seg - shift) + shift
            out[i:i + w] = seg_uw
            shift = torch.median(seg_uw)
            i += 1

        # 남는 꼬리 구간 처리 (윈도우보다 짧은 tail)
        if i < N:
            tail = vec[i:]
            seg_uw = wrap_to_pi_t(tail - shift) + shift
            out[i:] = seg_uw

        out = out - two_pi * torch.round(torch.mean(out) / two_pi)
        return out.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------------------------------


    def _apply_phase_rotation_inplace(
        self,
        fringes: torch.Tensor,     # (K,W,F) complex
        k_line: torch.Tensor,      # (K,) real
        cumLF: torch.Tensor,       # (W,F) real  (Y 보정: cumPhaseY_add+cumPhase2D, X 보정: cumPhaseX_add broadcast)
        sign: float,               # ±1
        kc: float | torch.Tensor,  # scalar
        chunk_k: int = 256         # 청크 크기 (상황 맞춰 128~1024 조정)
    ) -> None:
        """
        fringes ← fringes * exp( j * coef * k * cumLF ) 를 K축 청크로 나눠서 적용.
        메모리 폭발 방지용.
        """
        device = fringes.device
        rdtype = fringes.real.dtype

        # 스칼라/상수 dtype·device 정리
        k_line = k_line.to(device=device, dtype=rdtype)
        coef   = (-float(sign) / float(kc))
        cumLF  = cumLF.to(device=device, dtype=rdtype)  # (W,F)

        K = k_line.numel()
        W, F = cumLF.shape

        # 각 청크에 대해: theta = coef * k_chunk[:,None,None] * cumLF[None,:,:]
        for i0 in range(0, K, chunk_k):
            i1 = min(i0 + chunk_k, K)
            k_chunk = k_line[i0:i1].view(-1, 1, 1)             # (k',1,1)
            theta   = coef * k_chunk * cumLF.view(1, W, F)     # (k',W,F)
            rot     = torch.polar(torch.ones_like(theta, dtype=rdtype), theta)  # complex
            fringes[i0:i1] = fringes[i0:i1] * rot.to(fringes.dtype)
            
    def _initialize_params(self):
        self.W_info = int(self.info["numImgLines"])
        self.F_info = int(self.info["numImgFrames"])
        self.num_pixels = int(self.info["numImgPixels"])
        self.rad_per_pixel = float(self.info["radPerPixel"])
        self.depth_per_pixel = float(self.info["depthPerPixel"])
        self.device_hint = "cuda" if torch.cuda.is_available() else "cpu"

        n = float(self.info["n"])
        kl = n * 2.0 * np.pi / float(self.info["wlhigh"])
        kh = n * 2.0 * np.pi / float(self.info["wllow"])
        num_used = int(self.info["numUsedSamples"])
        num_ft = int(self.info["numFTSamples"])

        ke = (kh - kl) / (num_used - 1) * (num_ft - num_used)
        kc = (kh + kl) / 2.0
        self.kc = torch.tensor(kc, dtype=torch.float32)

        if bool(self.info.get("FDFlip", False)):
            k_line = torch.linspace(-kh, -kl + ke, steps=num_ft, dtype=torch.float32)
            self.sign = -1.0
        else:
            k_line = torch.linspace(kl - ke, kh, steps=num_ft, dtype=torch.float32)
            self.sign = 1.0

        self.k_line = k_line
        self._k0_cache_shape = None
        self._k0_cached = None

    def _fast_ilmisos_like(self, intImg_n: torch.Tensor, sigmaZ: float, sigmaX: float, sigmaY: float) -> torch.Tensor:
        """간이 fastILMISOS 대체 (CPU SciPy로 3D 가우시안 후 argmax)."""
        dev = intImg_n.device
        vol = intImg_n.detach().cpu().numpy().astype(np.float64)
        vol_s = gaussian_filter(vol, sigma=(sigmaZ, sigmaX, sigmaY), mode="nearest", truncate=2.0)
        idx = np.argmax(vol_s, axis=0).astype(np.int64)  # (W,F)
        return torch.from_numpy(idx).to(dev)

    def _k0(self, W: int, F: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._k0_cached is None
            or self._k0_cache_shape != (W, F)
            or self._k0_cached.device != device
            or self._k0_cached.dtype != dtype
        ):
            k = self.k_line.to(device=device, dtype=dtype)           # (K,)
            k0 = k.view(-1, 1, 1).expand(-1, W, F)                   # (K,W,F)
            self._k0_cached = k0
            self._k0_cache_shape = (W, F)
        return self._k0_cached

    # ------------------------------------------------------------------------------------------

    def stabilize_volume(
        self,
        fringes: torch.Tensor,
        d_file_name: str | None = None,
        vol_1b: int | None = None,
        subvol_1b: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_device = fringes.device
        self._last_dfile_name = d_file_name
        try:
            return self._stabilize_logic(fringes, d_file_name, vol_1b, subvol_1b)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fr_cpu = fringes.cpu()
            torch.set_num_threads(os.cpu_count() or 1)
            return self._stabilize_logic(fr_cpu, d_file_name, vol_1b, subvol_1b)

    def _stabilize_logic(self, fringes: torch.Tensor, d_file_name: str | None, vol_1b: int | None, subvol_1b: int | None):
        assert fringes.is_complex()
        dev = fringes.device
        K, W, F = fringes.shape
        if (W != self.W_info) or (F != self.F_info):
            self.W_info, self.F_info = W, F

        # 상수/파라미터를 입력 텐서의 device/dtype으로
        kc = self.kc.to(dev, dtype=fringes.real.dtype)
        k_line = self.k_line.to(dev, dtype=fringes.real.dtype)
        sign = torch.tensor(self.sign, dtype=fringes.real.dtype, device=dev)

        # 누적 위상
        five_arg_mode = (d_file_name is not None and vol_1b is not None and subvol_1b is not None)
        if five_arg_mode:
            cumPhaseX = torch.zeros(W, dtype=torch.float64, device=dev)
            cumPhaseY = torch.zeros(F, dtype=torch.float64, device=dev)
            sx, sy = map(int, np.asarray(self.info.get("subdivFactors", [1, 1])).reshape(-1)[:2])
            sub = int(subvol_1b) - 1
            m = sub % sx
            n = sub // sx
        else:
            cumPhaseX = torch.zeros(W, dtype=torch.float64, device=dev)
            cumPhaseY = torch.zeros(F, dtype=torch.float64, device=dev)

        repeats = int(self.info.get("stabilizePhaseRepeats", 1))

        for rep in range(1, repeats + 1):
            print(f"Stabilizing Phase {rep}/{repeats} on {dev.type.upper()}")

            # intensity
            img = torch.fft.ifft(fringes, dim=0)           # (K,W,F) -> complex
            img = img[: self.num_pixels, :, :]             # (Z,W,F)
            intImg = (img.real**2 + img.imag**2)           # (Z,W,F), float

            # segmentation sigma 스케일
            if rep == 1:
                sigmaZ = float(self.info["sigmaZ"])
                sigmaX = float(self.info["sigmaX"]) * 2.0
                sigmaY = float(self.info["sigmaY"]) / 2.0
            else:
                sigmaZ = float(self.info["sigmaZ"])
                sigmaX = float(self.info["sigmaX"])
                sigmaY = float(self.info["sigmaY"])

            # fastILMISOS 대체 입력
            nf = float(self.info["noiseFloor"])
            intImg_n = torch.clamp(intImg / (10.0 ** (nf / 10.0)) - 1.0, min=0.0)

            ISOS = self._fast_ilmisos_like(intImg_n, sigmaZ, sigmaX, sigmaY)  # (W,F), long

            # PRL mask
            Z = torch.arange(intImg.shape[0], device=dev, dtype=intImg.dtype).view(-1, 1, 1).expand_as(intImg)
            ISOSArray = Z - ISOS.to(intImg.dtype).unsqueeze(0)  # (Z,W,F)
            PRLMask = ((ISOSArray > (-25e-3 / self.depth_per_pixel)) & (ISOSArray < (75e-3 / self.depth_per_pixel)))
            valid_any = PRLMask.any(dim=(1, 2))

            if valid_any.any():
                nz = torch.nonzero(valid_any, as_tuple=False).squeeze(1)
                depthStart = int(nz[0].item())
                depthEnd = int(nz[-1].item())
            else:
                depthStart = 0
                depthEnd = int(min(self.num_pixels - 1, intImg.shape[0] - 1))

            intImg = (intImg * PRLMask)[depthStart:depthEnd + 1]
            img = (img * PRLMask)[depthStart:depthEnd + 1]

            # ---- Y 방향 subpixel registration ----
            IntImgCurr = torch.fft.fft(intImg[:, :, 0], dim=0)
            intImgShift = torch.zeros(F - 1, dtype=torch.float64, device=dev)

            for frame in range(1, F):
                IntPrev = IntImgCurr
                IntImgCurr = torch.fft.fft(intImg[:, :, frame], dim=0)

                valid = ((IntPrev != 0).any(dim=0) & (IntImgCurr != 0).any(dim=0)).detach().cpu().numpy()
                if np.any(valid):
                    # skimage는 numpy/CPU만 지원 → 실수형으로 캐스팅
                    A = IntPrev[:, valid].detach().cpu().numpy().astype(np.complex64)
                    B = IntImgCurr[:, valid].detach().cpu().numpy().astype(np.complex64)
                    shift, _, _ = phase_cross_correlation(A, B, space="fourier", upsample_factor=128, normalization=None)
                    intImgShift[frame - 1] = float(shift[0])
                else:
                    intImgShift[frame - 1] = float("nan")

            # rad/pixel → phase
            intPhase = sign.to(torch.float64) * intImgShift * float(self.info["radPerPixel"])
            int_np = intPhase.detach().cpu().numpy().astype(np.float64)
            int_np_f = self._savgol_safe(int_np, polyorder=3, approx_seg=50)
            intPhaseFilt = torch.from_numpy(int_np_f).to(device=dev, dtype=torch.float64)

            # ---- Y doppler/unwrap2 ----
            FAC = img[:, :, 1:] * img[:, :, :-1].conj()        # (Z,W,F-1)
            doppPhase = torch.angle(FAC.sum(dim=(0, 1)))       # (F-1,), float
            doppPhaseWrap = doppPhase.clone()

            doppPhase = self._unwrap2(self._wrap_to_pi(doppPhase - intPhaseFilt), 3) + intPhaseFilt
            intPhaseFilt = intPhaseFilt - torch.median(intPhaseFilt - doppPhase)
            doppPhase = self._unwrap2(self._wrap_to_pi(doppPhase - intPhaseFilt), 3) + intPhaseFilt
            intPhaseFilt = intPhaseFilt * torch.median(doppPhase / (intPhaseFilt + 1e-12))
            doppPhase = self._unwrap2(self._wrap_to_pi(doppPhase - intPhaseFilt), 3) + intPhaseFilt

            cumPhaseY_add = torch.zeros(F, dtype=torch.float64, device=dev)
            cumPhaseY_add[1:] = torch.cumsum(doppPhase, dim=0)
            if five_arg_mode:
                cumPhaseY += cumPhaseY_add

            # ---- y-ramp 2D correction ----
            FACsum = FAC.sum(dim=0).permute(1, 0) * torch.exp(-1j * doppPhase.to(FAC.dtype))[:, None]   # (F-1,W)
            # CPU gaussian_filter
            fac_r = gaussian_filter(FACsum.real.detach().cpu().numpy(), sigma=[5.0, F/4.0 + 1.0], mode="nearest", truncate=2.0)
            fac_i = gaussian_filter(FACsum.imag.detach().cpu().numpy(), sigma=[5.0, F/4.0 + 1.0], mode="nearest", truncate=2.0)
            fac_r_t = torch.from_numpy(fac_r).to(device=dev, dtype=fringes.real.dtype)
            fac_i_t = torch.from_numpy(fac_i).to(device=dev, dtype=fringes.real.dtype)
            doppPhase2D = torch.atan2(fac_i_t, fac_r_t)  # angle(fac_r + j*fac_i) = atan2(im, re)  → (F-1,W)

            cumPhase2D = torch.zeros((F, W), dtype=fringes.real.dtype, device=dev)
            cumPhase2D[1:, :] = torch.cumsum(doppPhase2D, dim=0)

            cumLF = (cumPhaseY_add.float().unsqueeze(1) + cumPhase2D).T  # (W,F)
            self._apply_phase_rotation_inplace(
                fringes=fringes,
                k_line=self.k_line.to(dev),    # (K,)
                cumLF=cumLF,                   # (W,F)
                sign=sign,
                kc=self.kc.item(),             # float 스칼라로
                chunk_k=256
            )

            # ---- X correction ----
            img2 = torch.fft.ifft(fringes, dim=0)
            img2 = img2[depthStart:depthEnd + 1]
            FACx = img2[:, 1:, :] * img2[:, :-1, :].conj()
            doppPhaseX = torch.angle(FACx.sum(dim=(0, 2)))                   # (W-1,)
            doppPhaseX = torch.from_numpy(np.unwrap(doppPhaseX.detach().cpu().numpy())).to(dev, dtype=torch.float64)

            cumPhaseX_add = torch.zeros(W, dtype=torch.float64, device=dev)
            cumPhaseX_add[1:] = torch.cumsum(doppPhaseX, dim=0)
            if five_arg_mode:
                cumPhaseX += cumPhaseX_add

            cumLFx = cumPhaseX_add.to(torch.float32).view(W, 1).expand(W, F)  # (W,F)로 브로드캐스트
            self._apply_phase_rotation_inplace(
                fringes=fringes,
                k_line=self.k_line.to(dev),    # (K,)
                cumLF=cumLFx,                  # (W,F)
                sign=sign,
                kc=self.kc.item(),
                chunk_k=256
            )

        # ----------------- 저장/참조 정렬 (nargin==5) -----------------
        if five_arg_mode:
            mid_line = round(W / 2) - 1
            mid_frame = round(F / 2) - 1

            centerX = np.asarray(self.info["centerX"]).reshape(-1)
            centerY = np.asarray(self.info["centerY"]).reshape(-1)
            refCumX = torch.as_tensor(self.info["cumPhaseX"], dtype=torch.float64, device=dev)
            refCumY = torch.as_tensor(self.info["cumPhaseY"], dtype=torch.float64, device=dev)

            ref_x_idx = int(centerX[m]) - 1
            ref_y_idx = int(centerY[n]) - 1

            cumPhaseXshift = -cumPhaseX[mid_line] + refCumX[ref_x_idx]
            cumPhaseX += cumPhaseXshift
            theta_sx = (-sign / kc) * k_line.view(-1, 1, 1) * cumPhaseXshift.to(fringes.real.dtype)
            fringes = fringes * torch.polar(torch.ones_like(fringes.real), theta_sx).to(fringes.dtype)

            cumPhaseYshift = -cumPhaseY[mid_frame] + refCumY[ref_y_idx]
            cumPhaseY += cumPhaseYshift
            theta_sy = (-sign / kc) * k_line.view(-1, 1, 1) * cumPhaseYshift.to(fringes.real.dtype)
            fringes = fringes * torch.polar(torch.ones_like(fringes.real), theta_sy).to(fringes.dtype)

            p = Path(d_file_name)
            save_path = p.parent / f"{p.stem}_Int_{int(vol_1b):02d}_{int(subvol_1b):02d}_cumPhase.pt"

            old_mat = p.parent / f"{p.stem}_Int_{int(vol_1b):02d}_{int(subvol_1b):02d}_cumPhase.mat"
            try:
                if old_mat.exists():
                    old_mat.unlink()
            except Exception:
                pass

            torch.save({
                "cumPhaseX": cumPhaseX.detach().cpu(),
                "cumPhaseY": cumPhaseY.detach().cpu(),
            }, save_path)

            print(f"⇢ Phase-shift info saved → {save_path}")

        return fringes, cumPhaseX, cumPhaseY