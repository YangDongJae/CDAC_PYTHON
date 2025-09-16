# /core/segment/prl_rpe_segmenter.py
import torch
import numpy as np

# 정확 포팅한 fastILMISOS 사용 (위에 만든 모듈)
from .fastILMISOS import (
    fastILMISOS as fastILMISOS_torch,  # torch 래퍼
    _imgaussfilt3_exact,               # imgaussfilt3 복제
    mround,
)

class PrlRpeSegmenter:
    """
    MATLAB segmentPRLNFL.m의 1:1 포팅 (객체지향 래퍼)
    Steps:
      1) fastILMISOS → (ILM, NFL, ISOS_pre)
      2) ISOS 기준 평탄화 (FFT 위상 램프)
      3) 평탄화 슬랩에 이방성 가우시안 (σ=[σz/4, 6σx, 6σy])
      4) 평균 기울기 기반 ISOS/RPE 재탐색 + 서브픽셀
      5) 원 좌표로 복귀 및 클램프
    """

    def __init__(self, info: dict, device: str | torch.device = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.info = info
        self.depth_per_pixel = float(info["depthPerPixel"])
        self.sigma = (
            float(info["sigmaZ"]),
            float(info["sigmaX"]),
            float(info["sigmaY"]),
        )

        # MATLAB 기본값 동일
        self.thr_isos_edge = float(info.get("thrISOSedge", 3.0))   # +edge
        self.thr_rpe_edge  = float(info.get("thrRPEedge", -3.0))   # -edge
        self.dBRange       = float(info.get("dBRange", 40.0))

        # 윈도우 half-size (기울기 인덱스계) : max(1, round(2*sigmaZ))
        self.win_half = int(max(1, mround(2.0 * self.sigma[0])))

        print(f"PrlRpeSegmenter initialized on device: {self.device}")

    # ----------------------- PUBLIC API -----------------------
    def segment(self, int_img: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            int_img: (L,Z-depth, W=X-lines, H=frames) = (Z, X, Y) float 텐서
        Returns:
            (ILM, NFL, ISOS, RPE)  — 각 (W, H), torch.float64, 1-based 좌표
        """
        # dtype/device 정렬
        X = int_img.to(self.device, dtype=torch.float64)
        L, W, H = X.shape

        # 1) 초기 레이어 (fastILMISOS 정확포팅 버전)
        ILM, NFL, ISOS_pre = fastILMISOS_torch(X, self.sigma)
        # fastILMISOS_torch는 이미 1-based float64 텐서를 반환

        # 2) ISOS 기준 슬랩 평탄화 ([-25µm, +75µm] → slabLen)
        flat_slab, dz_low, slab_len = self._flatten_slab(X, ISOS_pre)  # (slab_len, W, H)

        # 3) 평탄화 슬랩 필터링 (imgaussfilt3 정확복제)
        #    σ = [σz/4, 6σx, 6σy]  (단, σz/4가 0이면 eps로 처리)
        sigma_z, sigma_x, sigma_y = self.sigma
        sigZ2 = max(sigma_z / 4.0, np.finfo(float).eps)
        anis_sig = (sigZ2, 6.0 * sigma_x, 6.0 * sigma_y)
        flat_slab_np = flat_slab.detach().cpu().numpy().astype(np.float64)
        flat_slab_f = _imgaussfilt3_exact(flat_slab_np, anis_sig)
        flat_slab_f = torch.from_numpy(flat_slab_f).to(self.device, dtype=torch.float64)

        # 4) 프레임별 평균기울기 윈도우 + 서브픽셀: ISOS/RPE 오프셋 탐색
        ISOS_offsets = torch.zeros((W, H), device=self.device, dtype=torch.float64)
        RPE_offsets  = torch.zeros((W, H), device=self.device, dtype=torch.float64)

        for f in range(H):
            V = flat_slab_f[:, :, f]  # (slab_len, W)
            iso_off = self._detect_offsets_avg_grad(
                V, polarity=+1, order="first", edge_thr=self.thr_isos_edge,
                win_half=self.win_half, dz_low=dz_low
            )
            rpe_off = self._detect_offsets_avg_grad(
                V, polarity=-1, order="last",  edge_thr=self.thr_rpe_edge,
                win_half=self.win_half, dz_low=dz_low
            )
            ISOS_offsets[:, f] = iso_off
            RPE_offsets[:, f]  = rpe_off

        # 5) 원좌표로 복귀 + 클램프
        ISOS = ISOS_pre + ISOS_offsets
        RPE  = ISOS_pre + RPE_offsets

        ISOS = torch.clamp(ISOS, min=1.0, max=float(L))
        RPE  = torch.clamp(RPE,  min=1.0, max=float(L))

        return ILM, NFL, ISOS, RPE

    # ----------------------- INTERNALS -----------------------
    def _flatten_slab(self, X: torch.Tensor, ISOS_pre: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        MATLAB:
          dzLow = round(-0.025 / depthPerPixel)
          dzHigh = round(+0.075 / depthPerPixel)
          slabLen = dzHigh - dzLow + 1
          z0 = 1 - dzLow
          s = z0 - ISOS_pre     (1-based, fractional 가능)
          F = fft(X, [], 1); phase = exp(-1i * 2π/L * k .* s3)
          Xshift = ifft(F .* phase, [], 1, 'symmetric'); flatSlab = Xshift(1:slabLen,:,:)
        """
        L, W, H = X.shape
        dz_low  = int(mround(-0.025 / self.depth_per_pixel))  # MATLAB round
        dz_high = int(mround( 0.075 / self.depth_per_pixel))
        slab_len = dz_high - dz_low + 1

        z0 = 1.0 - dz_low                                # scalar (1-based target)
        s = z0 - ISOS_pre                                # (W,H) fractional
        s3 = s.view(1, W, H)                             # (1,W,H)

        # FFT along z
        F = torch.fft.fft(X, dim=0)
        k = torch.arange(L, device=self.device, dtype=torch.float64).view(L, 1, 1)
        phase = torch.exp(-1j * 2.0 * np.pi / L * k * s3)  # (L,W,H)
        Xshift = torch.fft.ifft(F * phase, dim=0).real     # 'symmetric' ≈ real

        flat_slab = Xshift[:slab_len, :, :]  # (slab_len, W, H)
        return flat_slab, dz_low, slab_len

    def _detect_offsets_avg_grad(
        self,
        V: torch.Tensor,              # (slab_len, W)
        polarity: int,                # +1: ISOS(+edge), -1: RPE(-edge)
        order: str,                   # 'first' or 'last'
        edge_thr: float,              # threshold on avg gradient (signed)
        win_half: int,                # half-window (in gradient index)
        dz_low: int,                  # base offset in original coords
    ) -> torch.Tensor:
        """
        MATLAB detectLayerOffsetsFromAvgGrad() 1:1 구현.
        반환: offsets (W,)  — ISOS_fit(=ISOS_pre) 대비 델타 (원 좌표계, 1-based)
        """
        slab_len, W = V.shape
        # G = diff(V,1,1) → (slab_len-1, W)
        G = torch.diff(V, n=1, dim=0).to(torch.float64)
        # gAvg = mean(G,2) → (slab_len-1, 1)
        g_avg = torch.mean(G, dim=1)  # (slab_len-1,)

        # edgeIdx 선택 (1-based, MATLAB 규칙)
        if polarity > 0:
            # +edge
            idxs = torch.nonzero(g_avg > edge_thr, as_tuple=False).view(-1)
            fallback = int(mround(0.25 * (slab_len - 1)))
            rectG = torch.clamp(G, min=0.0)
        else:
            # -edge
            idxs = torch.nonzero(g_avg < edge_thr, as_tuple=False).view(-1)
            fallback = int(mround(0.75 * (slab_len - 1)))
            rectG = torch.clamp(-G, min=0.0)

        if idxs.numel() == 0:
            edge_idx_1b = fallback if fallback >= 1 else 1
        else:
            if order.lower() == "first":
                edge_idx_1b = int(idxs[0].item() + 1)  # to 1-based
            else:
                edge_idx_1b = int(idxs[-1].item() + 1)

        # Window around edgeIdx  (모두 1-based 인덱스계)
        z1_1b = max(1, edge_idx_1b - win_half)
        z2_1b = min(slab_len - 1, edge_idx_1b + win_half)
        # Python 슬라이스 (0-based, end-exclusive): start=z1-1, stop=z2
        if z2_1b < z1_1b:
            # 윈도우가 비는 경우: 0-offset 반환 (MATLAB에는 없음; 안전장치)
            return torch.zeros(W, device=self.device, dtype=torch.float64)

        win = rectG[(z1_1b - 1):z2_1b, :]  # (win_len, W), 0-based slice
        win_len = win.shape[0]
        if win_len == 0:
            return torch.zeros(W, device=self.device, dtype=torch.float64)

        # Argmax per column (1-based within window)
        # torch.argmax → 0-based → MATLAB 1-based로 +1 변환
        k_local_0b = torch.argmax(win, dim=0)                 # (W,)
        k0_1b = k_local_0b.to(torch.float64) + 1.0            # (W,) 1-based in window

        # Sub-pixel parabolic refinement (MATLAB 그대로)
        if win_len >= 3:
            # km = max(k0-1,1); kp = min(k0+1, win_len)
            km_1b = torch.clamp(k0_1b - 1.0, min=1.0)
            kp_1b = torch.clamp(k0_1b + 1.0, max=float(win_len))

            # 0-based 인덱스로 환산
            km_0b = (km_1b - 1.0).round().long()
            k0_0b = (k0_1b - 1.0).round().long()
            kp_0b = (kp_1b - 1.0).round().long()

            col_idx = torch.arange(W, device=self.device, dtype=torch.long)

            y_m = win[km_0b, col_idx]
            y_0 = win[k0_0b, col_idx]
            y_p = win[kp_0b, col_idx]

            denom = (y_m - 2.0 * y_0 + y_p)
            denom[torch.abs(denom) < 1e-12] = 1e-12
            delta = 0.5 * (y_m - y_p) / denom
            delta = torch.clamp(delta, min=-1.0, max=1.0)

            k_sub_1b = k0_1b + delta  # (W,) 1-based within window
        else:
            k_sub_1b = k0_1b  # 윈도우 길이 < 3이면 서브픽셀 생략

        # slabIndex = (z1 - 1) + kSub + 1  (모두 1-based)
        slab_index_1b = float(z1_1b) + k_sub_1b  # (W,)

        # offsets = dzLow + slabIndex
        # 주의: dzLow는 원 좌표계에서의 오프셋(정수), slab_index_1b는 슬랩 내 1-based 좌표
        offsets = (dz_low + slab_index_1b).to(torch.float64)  # (W,)
        return offsets