import os
import math
from pathlib import Path
import numpy as np
import torch

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

# ..segment 폴더는 core 폴더와 같은 레벨에 있다고 가정
from ..segment.layer_segmenter import RetinalLayerSegmenter
from .utils import revert_fringes  


class PhaseStabilizer:
    """
    OCT 위상 안정화 및 관련 계산을 수행하는 종합 클래스.

    KAIST의 stabilizePhase.m 로직을 기반으로 하며, OOM 자동 복구,
    위상 데이터 처리, 픽셀 시프트 계산 등 다양한 기능을 포함합니다.
    """
    
    def __init__(self, info: dict):
        """
        Args:
            info (dict): OCT 메타데이터 (numImgLines, numImgFrames, radPerPixel 등 포함)
        """
        self.info = info
        self._initialize_params()
        print(f"PhaseStabilizer initialized: W={self.W_info}, F={self.F_info}, DeviceHint={self.device_hint}")

    def _initialize_params(self):
        """Info 딕셔너리로부터 주요 파라미터를 추출하고 계산하여 인스턴스 속성으로 저장합니다."""
        # 기본 정보
        self.W_info = int(self.info["numImgLines"])
        self.F_info = int(self.info["numImgFrames"])
        self.num_pixels = int(self.info["numImgPixels"])
        self.rad_per_pixel = float(self.info["radPerPixel"])
        self.depth_per_pixel = float(self.info["depthPerPixel"])
        self.device_hint = "cuda" if torch.cuda.is_available() else "cpu"

        # k-space 벡터 계산
        n_refr = float(self.info["n"])
        kl = n_refr * 2.0 * np.pi / float(self.info["wlhigh"])
        kh = n_refr * 2.0 * np.pi / float(self.info["wllow"])
        num_used = int(self.info["numUsedSamples"])
        num_ft = int(self.info["numFTSamples"])

        ke = (kh - kl) / (num_used - 1) * (num_ft - num_used)
        kc = (kh + kl) / 2.0
        
        self.sign = -1.0 if self.info.get("FDFlip", False) else 1.0
        if self.sign == -1.0:
            k_start, k_end = -kh, -kl + ke
        else:
            k_start, k_end = kl - ke, kh
        
        # device-agnostic 텐서로 우선 생성
        self.kvec = torch.linspace(k_start, k_end, steps=num_ft)
        self.alpha = torch.tensor(-self.sign / kc, dtype=torch.float32)

    # ===================================================================
    #   주요 기능: KAIST 위상 안정화 로직
    # ===================================================================

    def stabilize_volume(
        self,
        fringes: torch.Tensor,
        d_file_name: str | None = None,
        vol_1b: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        입력된 fringes 볼륨의 위상을 안정화합니다. (MATLAB stabilizePhase.m 포팅)
        OOM 발생 시 자동으로 CPU로 전환하여 재시도합니다.
        
        Args:
            fringes (torch.Tensor): (K, L, F) 복소수 프린지 텐서.
            d_file_name (str, optional): 결과 저장을 위한 파일명.
            vol_1b (int, optional): 볼륨 인덱스 (1-based).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - 안정화된 fringes (K, L, F)
            - cumPhaseX (L,)
            - cumPhaseY (F,)
        """
        target_device = fringes.device
        try:
            # 지정된 디바이스에서 실행 시도
            return self._stabilize_logic(fringes, d_file_name, vol_1b)
        
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print("\n" + "="*60)
            print(f"⚠️  Warning: Caught a CUDA error: {e}")
            print("Automatically retrying on CPU with multi-threading.")
            print("="*60 + "\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # CPU로 데이터 이동 및 스레드 설정
            fringes_cpu = fringes.cpu()
            num_cores = os.cpu_count() or 1
            torch.set_num_threads(num_cores)
            print(f"Switched to CPU. Using {torch.get_num_threads()} threads.")
            
            # CPU에서 로직 재실행 후 결과를 원래 디바이스로 복원
            fringes_s, cpx, cpy = self._stabilize_logic(fringes_cpu, d_file_name, vol_1b)
            return fringes_s, cpx, cpy

    def _stabilize_logic(
        self,
        fringes: torch.Tensor,
        d_file_name: str | None,
        vol_1b: int | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """위상 안정화의 핵심 알고리즘. (K,W,F)는 항상 입력 볼륨에서 읽는다."""
        assert fringes.is_complex(), "fringes must be a complex tensor"

        dev = fringes.device
        K, W, F = fringes.shape  # ← 항상 실제 데이터에서 사용

        # ✅ Info와 다르면 경고만 출력하고 실제 값으로 내부 상태를 동기화
        if (W != getattr(self, "W_info", W)) or (F != getattr(self, "F_info", F)):
            print(
                f"⚠️ Info/Volume mismatch detected: "
                f"using volume dims W={W}, F={F} (Info had W={getattr(self, 'W_info', 'NA')}, F={getattr(self, 'F_info', 'NA')})"
            )
            self.W_info, self.F_info = W, F  # 내부 상태 업데이트

        # 파라미터를 현재 디바이스로 이동
        kvec  = self.kvec.to(device=dev, dtype=fringes.real.dtype)
        alpha = self.alpha.to(device=dev)

        # 누적 위상 크기도 실제 W/F로 생성
        cumPhaseX = torch.zeros(W, dtype=torch.float64, device=dev)
        cumPhaseY = torch.zeros(F, dtype=torch.float64, device=dev)

        repeats = int(self.info.get("stabilizePhaseRepeats", 1))

        for rep in range(1, repeats + 1):
            print(f"\nStabilizing Phase {rep}/{repeats} on {dev.type.upper()}", flush=True)

            # 1) PRL 슬랩
            img, intImg, int_img_avg, sidx, eidx, th_val, z0, z1 = self._get_prl_slab(fringes, rep)
            img_slab = img[z0:z1+1, :, :]  # (Zslab, W, F)

            # 2) 프레임 간(Y) 위상
            cumY_add, dopp_phase_y, int_phase_filt, dopp_phase_wrap, FAC_y = self._correct_inter_frame_phase(img_slab)
            cumPhaseY += cumY_add
            self._apply_2d_phase_correction(fringes, cumY_add, dopp_phase_y, FAC_y, kvec, alpha)

            # 3) 프레임 내(X) 위상
            addX, dopp_phase_x = self._correct_intra_frame_phase(fringes, z0, z1)
            cumPhaseX += addX
            self._mul_phase_kL_inplace_broadcast_F(fringes, kvec, addX, alpha)

        # 4) (옵션) 기준 정렬 & 저장
        if d_file_name and vol_1b:
            self._align_to_reference(fringes, cumPhaseX, cumPhaseY, vol_1b, kvec, alpha)
            self._save_phase_info(cumPhaseX, cumPhaseY, d_file_name, vol_1b)

        # ✅ main()이 기대하는 3-튜플만 반환
        return fringes, cumPhaseX, cumPhaseY

    # ===================================================================
    #   ✨ 내부 헬퍼 메서드: 새로운 알고리즘으로 교체 및 추가
    # ===================================================================
    
    def _get_prl_slab(self, fringes: torch.Tensor, repetition: int) -> tuple:
        """✨ 평균 A-scan 기반의 새로운 Slab 검출 알고리즘"""
        dev = fringes.device
        img = torch.fft.ifft(fringes, dim=0)[:self.num_pixels]
        intImg = (img.real**2 + img.imag**2) / (10.0**(float(self.info["noiseFloor"])/10.0))

        # 평균 A-scan 프로필 계산
        valid = (intImg != 0).any(dim=0).to(torch.float64)
        valid[valid == 0] = torch.nan
        int_img_avg = torch.nanmean(intImg * valid.unsqueeze(0), dim=(1, 2))

        # 가장 밝은 점 기반으로 임계값 및 초기 Slab 위치(sidx, eidx) 찾기
        tail_start = int(self.num_pixels * 15 / 16)
        th_val = torch.max(int_img_avg[tail_start:])
        idx = torch.where(int_img_avg >= th_val)[0]
        eidx = int(idx[-1].item()); sidx = int(idx[0].item()); pidx = eidx
        thickness = pidx - sidx + 1

        # 180um / 60um 조건에 맞게 Slab 두께 정제
        while thickness > 180e-3 / self.depth_per_pixel and pidx > 0:
            pidx -= 1
            if int_img_avg[pidx] > th_val:
                th_val = int_img_avg[pidx]
                sidx = int(torch.where(int_img_avg >= th_val)[0][0].item())
                eidx = pidx; thickness = pidx - sidx + 1
        
        before = torch.where(int_img_avg[:eidx+1] < th_val)[0]
        sidx = int(before[-1].item()) + 1 if before.numel() else 0; thickness = eidx - sidx + 1

        while thickness > 60e-3 / self.depth_per_pixel and eidx > 0: # pidx -> eidx
            eidx -= 1
            if int_img_avg[eidx] > th_val:
                th_val = int_img_avg[eidx]
                before = torch.where(int_img_avg[:sidx+1] < th_val)[0]
                sidx = int(before[-1].item()) + 1 if before.numel() else 0; thickness = eidx - sidx + 1
        
        # 최종 Slab 범위(z0, z1) 계산
        if (eidx - sidx) > 0:
            slab_half = int(math.ceil(0.1 / self.depth_per_pixel))
            z0 = max(0, sidx - slab_half)
            z1 = min(intImg.shape[0] - 1, eidx + slab_half)
        else:
            z0, z1 = 0, self.num_pixels - 1
        
        return img, intImg, int_img_avg, sidx, eidx, th_val, z0, z1

    def _correct_inter_frame_phase(self, img_slab: torch.Tensor) -> tuple:
        """Inter-frame (Y-축) 위상 변이를 계산합니다."""
        dev = img_slab.device; _, _, F = img_slab.shape
        intImg_slab = img_slab.real**2 + img_slab.imag**2
        shifts = torch.zeros(F - 1, dtype=torch.float64, device=dev)
        IntCurr = torch.fft.fft(intImg_slab[:, :, 0], dim=0)
        for f in range(1, F):
            IntPrev, IntCurr = IntCurr, torch.fft.fft(intImg_slab[:, :, f], dim=0); valid_cols = ((IntPrev != 0).any(dim=0) & (IntCurr != 0).any(dim=0)).cpu().numpy()
            if np.any(valid_cols):
                shift, _, _ = phase_cross_correlation(IntPrev[:, valid_cols].cpu().numpy(), IntCurr[:, valid_cols].cpu().numpy(), space="fourier", upsample_factor=128, normalization=None); shifts[f-1] = float(shift[0])
            else: shifts[f-1] = torch.nan
        int_phase = self.sign * shifts * self.rad_per_pixel; win_len = max(5, 2 * (F // 50) + 1); int_phase_filt = torch.from_numpy(savgol_filter(int_phase.cpu().numpy(), win_len, 3, mode="interp")).to(device=dev, dtype=torch.float64)
        FAC = img_slab[:, :, 1:] * img_slab[:, :, :-1].conj(); dopp_phase = torch.angle(FAC.sum(dim=(0, 1))); dopp_phase_wrap = dopp_phase.clone()
        for i in range(3):
            delta = (dopp_phase - int_phase_filt + np.pi) % (2*np.pi) - np.pi; dopp_phase = torch.from_numpy(np.unwrap(delta.cpu().numpy())).to(dev) + int_phase_filt
            if i == 0: int_phase_filt -= torch.median(int_phase_filt - dopp_phase)
            if i == 1: int_phase_filt *= torch.median(dopp_phase / (int_phase_filt + 1e-12))
        cumY_add = torch.zeros(F, device=dev, dtype=torch.float64); cumY_add[1:] = torch.cumsum(dopp_phase, dim=0)
        return cumY_add, dopp_phase, int_phase_filt, dopp_phase_wrap, FAC

    def _apply_2d_phase_correction(self, fringes, cumY_add, dopp_phase_y, FAC_y, kvec, alpha):
        dev = fringes.device; K, W, F = fringes.shape;
        FACsum = FAC_y.sum(dim=0).permute(1, 0)
        rotF = torch.polar(torch.ones_like(dopp_phase_y), -dopp_phase_y)
        FACsum = FACsum.to(torch.complex64) * rotF.to(torch.complex64).unsqueeze(1)
        fac_r = gaussian_filter(FACsum.real.cpu().numpy(), sigma=[5.0, F/4.0+1.0], mode="nearest", truncate=2.0); fac_i = gaussian_filter(FACsum.imag.cpu().numpy(), sigma=[5.0, F/4.0+1.0], mode="nearest", truncate=2.0)
        doppPhase2D = torch.angle(torch.from_numpy(fac_r).to(dev) + 1j * torch.from_numpy(fac_i).to(dev))
        cumPhase2D = torch.zeros((F, W), device=dev, dtype=torch.float32); cumPhase2D[1:, :] = torch.cumsum(doppPhase2D, dim=0)
        cum_LF = (cumY_add.float().unsqueeze(1) + cumPhase2D).T
        self._mul_phase_kLF_inplace(fringes, kvec, cum_LF, alpha)

    def _correct_intra_frame_phase(self, fringes: torch.Tensor, z0: int, z1: int) -> tuple:
        dev = fringes.device; K, W, F = fringes.shape
        img = torch.fft.ifft(fringes, dim=0); img_slab_x = img[z0:z1+1, :, :]
        FACx = img_slab_x[:, 1:, :] * img_slab_x[:, :-1, :].conj()
        dopp_phase_x = torch.angle(FACx.sum(dim=(0, 2)))
        dopp_phase_x = torch.from_numpy(np.unwrap(dopp_phase_x.cpu().numpy())).to(dev, torch.float64)
        addX = torch.zeros(W, device=dev, dtype=torch.float64); addX[1:] = torch.cumsum(dopp_phase_x, dim=0)
        return addX, dopp_phase_x

    def _align_to_reference(self, fringes, cumPhaseX, cumPhaseY, vol_1b, kvec, alpha):
        """계산된 위상을 기준 프레임/라인에 정렬. 레퍼런스 길이가 맞을 때만 적용."""
        K, W, F = fringes.shape

        # subdiv 인덱스 계산
        sx, sy = map(int, np.asarray(self.info.get("subdivFactors", [1, 1])).reshape(-1)[:2])
        m, n = (vol_1b - 1) % sx, (vol_1b - 1) // sx

        # ---- X 기준 정렬 ----
        centerVolX = self.info.get("centerVolX", None)
        ref_cumX   = self.info.get("cumPhaseX", None)

        if centerVolX is not None and ref_cumX is not None and len(ref_cumX) >= 1:
            mid_line_idx = round(W / 2.0) - 1
            ref_x_idx = int(centerVolX[m]) - 1 if isinstance(centerVolX, (list, np.ndarray)) else int(centerVolX) - 1

            ref_cumX_t = torch.as_tensor(ref_cumX, dtype=torch.float64, device=fringes.device)
            if 0 <= ref_x_idx < ref_cumX_t.numel():
                shiftX = -cumPhaseX[mid_line_idx] + ref_cumX_t[ref_x_idx]
                cumPhaseX += shiftX
                self._mul_phase_k_scalar_inplace(fringes, kvec, shiftX, alpha)
            else:
                print("⚠️ Skip X-alignment: centerVolX index out of range.")
        else:
            print("⚠️ Skip X-alignment: missing or incompatible reference cumPhaseX/centerVolX.")

        # ---- Y 기준 정렬 ----
        centerVolY = self.info.get("centerVolY", None)
        ref_cumY   = self.info.get("cumPhaseY", None)

        if centerVolY is not None and ref_cumY is not None and len(ref_cumY) >= 1:
            mid_frame_idx = round(F / 2.0) - 1
            ref_y_idx = int(centerVolY[n]) - 1 if isinstance(centerVolY, (list, np.ndarray)) else int(centerVolY) - 1

            ref_cumY_t = torch.as_tensor(ref_cumY, dtype=torch.float64, device=fringes.device)
            if 0 <= ref_y_idx < ref_cumY_t.numel():
                shiftY = -cumPhaseY[mid_frame_idx] + ref_cumY_t[ref_y_idx]
                cumPhaseY += shiftY
                self._mul_phase_k_scalar_inplace(fringes, kvec, shiftY, alpha)
            else:
                print("⚠️ Skip Y-alignment: centerVolY index out of range.")
        else:
            print("⚠️ Skip Y-alignment: missing or incompatible reference cumPhaseY/centerVolY.")

    def _save_phase_info(self, cumPhaseX, cumPhaseY, d_file_name, vol_1b):
        p = Path(d_file_name); save_path = p.parent / f"{p.stem}_Int_{vol_1b:02d}_cumPhase.pt"; torch.save({"cumPhaseX": cumPhaseX.cpu(), "cumPhaseY": cumPhaseY.cpu()}, save_path); print(f"⇢ Phase-shift info saved → {save_path}")

    # ✨ 메모리 최적화를 위한 블록 단위 곱셈 헬퍼 메서드들
    def _mul_phase_kLF_inplace(self, frg, kvec, phi_LF, alpha, block_k=32):
        K, _, _ = frg.shape
        for s in range(0, K, block_k):
            e = min(K, s + block_k); theta = (kvec[s:e].view(-1, 1, 1) * alpha) * phi_LF.to(kvec.device)
            rot = torch.polar(torch.ones_like(theta), theta); frg[s:e].mul_(rot.to(frg.dtype)); del theta, rot

    def _mul_phase_kL_inplace_broadcast_F(self, frg, kvec, phi_L, alpha, block_k=64):
        K, L, _ = frg.shape
        for s in range(0, K, block_k):
            e = min(K, s + block_k); theta = (kvec[s:e].view(-1, 1, 1) * alpha) * phi_L.to(kvec.device).view(1, L, 1)
            rot = torch.polar(torch.ones_like(theta), theta); frg[s:e].mul_(rot.to(frg.dtype)); del theta, rot

    def _mul_phase_k_scalar_inplace(self, frg, kvec, phi_scalar, alpha, block_k=256):
        K, _, _ = frg.shape
        for s in range(0, K, block_k):
            e = min(K, s + block_k); theta = (kvec[s:e] * alpha * phi_scalar.to(kvec.device)).view(-1, 1, 1)
            rot = torch.polar(torch.ones_like(theta), theta); frg[s:e].mul_(rot.to(frg.dtype)); del theta, rot