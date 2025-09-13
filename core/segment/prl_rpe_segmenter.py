# /core/segment/prl_rpe_segmenter.py

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from .layer_segmenter import RetinalLayerSegmenter # 기존 fastILMISOS 포팅 클래스

class PrlRpeSegmenter:
    """
    MATLAB의 segmentPRLNFL.m 로직을 객체지향적으로 포팅한 클래스입니다.

    1. `RetinalLayerSegmenter`를 사용하여 ILM, NFL, ISOS의 초기 위치를 찾습니다.
    2. ISOS를 기준으로 OCT 볼륨을 평탄화(Flattening)합니다.
    3. 평탄화된 이미지 슬랩(slab)에 이방성 3D 가우시안 필터를 적용합니다.
    4. 필터링된 슬랩에서 ISOS와 RPE 경계면을 더 정밀하게 재탐색합니다.
    """

    def __init__(self, info: dict, device: str | torch.device = None):
        """
        초기화 메서드

        Args:
            info (dict): OCT 메타데이터. depthPerPixel, sigmaZ/X/Y 등이 필요합니다.
            device (str | torch.device, optional): 연산을 수행할 장치 ('cuda' or 'cpu').
                                                   None이면 자동으로 감지합니다.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # 1. 파라미터 설정
        self.info = info
        self.depth_per_pixel = float(info['depthPerPixel'])
        self.sigma = (float(info['sigmaZ']), float(info['sigmaX']), float(info['sigmaY']))
        
        # 기본값 설정 (MATLAB 코드와 동일)
        self.thr_isos_edge = float(info.get('thrISOSedge', 3.0))
        self.thr_rpe_edge = float(info.get('thrRPEedge', -3.0))

        # 2. 초기 레이어 분할기 인스턴스 생성
        self.layer_segmenter = RetinalLayerSegmenter(sigma=self.sigma)
        print(f"PrlRpeSegmenter initialized on device: {self.device}")

    def segment(self, int_img: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        주어진 강도 이미지에서 ILM, NFL, ISOS, RPE 레이어를 분할합니다.

        Args:
            int_img (torch.Tensor): (Z, X, Y) 형상의 3D 강도 이미지 텐서.

        Returns:
            Tuple[torch.Tensor, ...]: ILM, NFL, ISOS, RPE 2D 경계면 맵 (X, Y) 튜플.
        """
        int_img = int_img.to(self.device, dtype=torch.float64)
        L, W, H = int_img.shape

        # 1. 초기 레이어 분할 (fastILMISOS 호출)
        ilm, nfl, isos_pre = self.layer_segmenter.segment(int_img)

        # 2. ISOS 주위 슬랩 평탄화 (FFT 기반 시프트)
        flat_slab = self._flatten_slab(int_img, isos_pre)
        
        # 3. 평탄화된 슬랩에 이방성 3D 가우시안 필터 적용
        filtered_slab = self._filter_slab(flat_slab)

        # 4. 필터링된 슬랩에서 ISOS 및 RPE 재탐색 (오프셋 계산)
        win_half = max(1, round(2 * self.sigma[0]))
        
        isos_offsets = torch.zeros((W, H), device=self.device, dtype=torch.float64)
        rpe_offsets = torch.zeros((W, H), device=self.device, dtype=torch.float64)

        for f in range(H):
            V = filtered_slab[:, :, f] # 단일 프레임
            
            # ISOS: 첫 번째 강한 양의 경계
            isos_offsets[:, f] = self._detect_layer_offsets_from_avg_grad(
                V, polarity=+1, order='first', edge_thr=self.thr_isos_edge, win_half=win_half
            )
            # RPE: 마지막 강한 음의 경계
            rpe_offsets[:, f] = self._detect_layer_offsets_from_avg_grad(
                V, polarity=-1, order='last', edge_thr=self.thr_rpe_edge, win_half=win_half
            )
            
        # 5. 원래 좌표계로 오프셋 매핑 및 경계값 처리
        isos = isos_pre + isos_offsets
        rpe = isos_pre + rpe_offsets
        
        isos = torch.clamp(isos, min=1, max=L)
        rpe = torch.clamp(rpe, min=1, max=L)

        return ilm, nfl, isos, rpe

    def _flatten_slab(self, X: torch.Tensor, isos_pre: torch.Tensor) -> torch.Tensor:
        """FFT 기반 시프트를 사용하여 ISOS 레이어 기준으로 이미지를 평탄화합니다."""
        L, W, H = X.shape
        
        dz_low_mm = -0.025  # -25 µm
        dz_high_mm = 0.075   # +75 µm
        dz_low = round(dz_low_mm / self.depth_per_pixel)
        dz_high = round(dz_high_mm / self.depth_per_pixel)
        slab_len = dz_high - dz_low + 1
        
        z0 = 1 - dz_low  # ISOS가 평탄화 후 위치할 목표 z 인덱스 (1-based)
        
        s = z0 - isos_pre # 각 A-scan이 시프트되어야 할 양 (픽셀 단위)
        s3 = s.view(1, W, H)

        # FFT -> 위상 적용 -> IFFT
        F = torch.fft.fft(X, axis=0)
        k = torch.arange(L, device=self.device, dtype=torch.float64).view(L, 1, 1)
        phase = torch.exp(-1j * 2 * np.pi / L * k * s3)
        
        # 'symmetric' 옵션은 ifft 결과의 허수부를 버리는 것과 유사
        X_shift = torch.fft.ifft(F * phase, axis=0).real
        
        # 0-based 인덱싱으로 슬랩 추출
        flat_slab = X_shift[0:slab_len, :, :]
        return flat_slab

    def _filter_slab(self, slab: torch.Tensor) -> torch.Tensor:
        """평탄화된 슬랩에 이방성 3D 가우시안 필터를 적용합니다."""
        # SciPy 함수는 CPU NumPy 배열에서만 동작
        slab_np = slab.cpu().numpy()
        
        anis_sig = [max(self.sigma[0]/4, np.finfo(float).eps), 6*self.sigma[1], 6*self.sigma[2]]
        
        filtered_slab_np = gaussian_filter(slab_np, sigma=anis_sig)
        
        return torch.from_numpy(filtered_slab_np).to(self.device)

    def _detect_layer_offsets_from_avg_grad(
        self, V: torch.Tensor, polarity: int, order: str, edge_thr: float, win_half: int
    ) -> torch.Tensor:
        """평균 기울기에서 경계를 찾고, 각 A-scan에서 위치를 정밀화하여 오프셋을 계산합니다."""
        slab_len, W = V.shape
        G = torch.diff(V, n=1, dim=0)
        g_avg = torch.mean(G, dim=1)
        
        # 1. 평균 기울기에서 기준 경계 인덱스 찾기
        if polarity > 0:
            indices = torch.where(g_avg > edge_thr)[0]
            rect_G = torch.clamp(G, min=0) # 양의 경계만 유지
            fallback_idx = round(0.25 * (slab_len - 1))
        else:
            indices = torch.where(g_avg < edge_thr)[0]
            rect_G = torch.clamp(-G, min=0) # 음의 경계만 유지 (양수화)
            fallback_idx = round(0.75 * (slab_len - 1))

        if len(indices) == 0:
            edge_idx = fallback_idx
        elif order == 'first':
            edge_idx = indices[0].item()
        else: # 'last'
            edge_idx = indices[-1].item()
            
        # 2. 기준 인덱스 주변 윈도우 설정 (0-based)
        z1 = max(0, edge_idx - win_half)
        z2 = min(slab_len - 2, edge_idx + win_half) # diff로 길이가 1 줄었으므로 -2
        win = rect_G[z1:z2+1, :]
        
        if win.shape[0] == 0: # 윈도우가 비었을 경우
             return torch.zeros(W, device=self.device, dtype=torch.float64)

        # 3. 윈도우 내에서 각 A-scan별 최대값 위치 찾기
        _, k_local = torch.max(win, dim=0) # 윈도우 내 로컬 인덱스 (0-based)

        # 4. 서브픽셀 정밀도 향상 (포물선 보간)
        if win.shape[0] >= 3:
            k_sub = self._parabolic_refinement(win, k_local)
        else:
            k_sub = k_local.to(torch.float64)
            
        # 5. 최종 오프셋 계산
        # 기울기 인덱스 -> 슬랩 인덱스 -> 원본 좌표계 오프셋
        slab_index = z1 + k_sub
        dz_low = round(-0.025 / self.depth_per_pixel)
        offsets = dz_low + slab_index
        return offsets

    def _parabolic_refinement(self, win: torch.Tensor, k0: torch.Tensor) -> torch.Tensor:
        """포물선 보간법으로 서브픽셀 단위의 최대값 위치를 찾습니다."""
        W = win.shape[1]
        win_len = win.shape[0]
        
        km = torch.clamp(k0 - 1, min=0)
        kp = torch.clamp(k0 + 1, max=win_len - 1)
        
        # arange를 사용하여 각 열(column)에서 올바른 인덱스의 값을 추출
        w_indices = torch.arange(W, device=self.device)
        y_m = win[km, w_indices]
        y_0 = win[k0, w_indices]
        y_p = win[kp, w_indices]
        
        denom = y_m - 2 * y_0 + y_p
        # 0으로 나누는 것을 방지
        denom[denom.abs() < 1e-9] = 1e-9
        
        delta = 0.5 * (y_m - y_p) / denom
        delta = torch.clamp(delta, min=-1, max=1)
        
        return k0.to(torch.float64) + delta