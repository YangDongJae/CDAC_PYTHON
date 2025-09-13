# /core/segment/layer_segmenter.py

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class RetinalLayerSegmenter:
    """
    MATLAB의 fastILMISOS.m 로직을 객체지향적으로 포팅한 클래스.
    3D OCT 강도 이미지에서 ILM, NFL, ISOS 망막 경계면을 분할한다.
    """

    def __init__(
        self,
        sigma: tuple[float, float, float],
        thr_ilm: float = 5.0,
        thr_nfl: float = 15.0,
        thr_isos: float = 0.015,
    ):
        self.sigma = sigma
        self.thr_ilm = thr_ilm
        self.thr_nfl = thr_nfl
        self.thr_isos = thr_isos

    def segment(
        self, int_img_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        주어진 강도 이미지 텐서에서 망막 레이어를 분할.

        Args:
            int_img_tensor: (Z, X, Y) 형상의 3D 강도 이미지 (PyTorch 텐서)

        Returns:
            (ILM, NFL, ISOS) 각 (X, Y) torch.FloatTensor
        """
        device = int_img_tensor.device
        int_img = int_img_tensor.detach().cpu().numpy().astype(np.float64)

        # 1) 3D 가우시안 필터(denoise)
        int_img_f = gaussian_filter(int_img, sigma=self.sigma)

        # 2) ILM (bias 없이 원시 인덱스)
        ilm = self._find_ilm(int_img_f)

        # 3) NFL (bias 없이 원시 인덱스)
        nfl = self._find_nfl(int_img_f, ilm)

        # 4) ISOS (bias 없이 원시 인덱스)
        isos = self._find_isos(int_img_f, nfl)

        # 5) 여기서만 +round(sigmaZ) 바이어스 1회 일괄 적용
        bias = int(round(self.sigma[0]))
        ilm = ilm + bias
        nfl = nfl + bias
        isos = isos + bias

        # 6) 텐서 변환
        ilm_t = torch.from_numpy(ilm).to(device=device, dtype=torch.float32)
        nfl_t = torch.from_numpy(nfl).to(device=device, dtype=torch.float32)
        isos_t = torch.from_numpy(isos).to(device=device, dtype=torch.float32)
        return ilm_t, nfl_t, isos_t

    # ----------------------- ILM -----------------------
    def _find_ilm(self, img: np.ndarray) -> np.ndarray:
        L, W, H = img.shape
        z_coords = np.arange(1, L + 1).reshape(-1, 1, 1)  # 1-based

        mask_ilm = img > self.thr_ilm
        z1 = np.where(mask_ilm, z_coords, np.inf)

        ilm = np.squeeze(np.min(z1, axis=0))  # (W,H), Inf if none
        ilm = self._subpixel_edge_crossing(ilm, img, self.thr_ilm, sense="up")
        return ilm  # bias는 segment()에서 일괄 적용

    # ----------------------- NFL -----------------------
    def _find_nfl(self, img: np.ndarray, ilm: np.ndarray) -> np.ndarray:
        L, W, H = img.shape
        z_coords = np.arange(1, L + 1).reshape(-1, 1, 1)  # 1-based

        dz_nfl = max(1, int(round(3 * self.sigma[0])))
        ilm_i = np.floor(ilm).astype(int)
        ilm_i[~np.isfinite(ilm_i)] = 1

        z_start = np.maximum(2, ilm_i)        # z-1 필요하므로 최소 2
        z_end = np.minimum(L, ilm_i + dz_nfl)

        search_nfl = (z_coords >= z_start.reshape(1, W, H)) & (
            z_coords <= z_end.reshape(1, W, H)
        )

        above_thr = (img >= self.thr_nfl) & search_nfl
        z_mask_a = np.where(above_thr, z_coords, np.inf)
        z_first_above = np.squeeze(np.min(z_mask_a, axis=0))  # (W,H)

        after_above = z_coords >= (z_first_above + 1).reshape(1, W, H)
        below_thr = (img < self.thr_nfl) & after_above
        z_mask_b = np.where(below_thr, z_coords, np.inf)
        z_first_below = np.squeeze(np.min(z_mask_b, axis=0))  # (W,H)

        nfl = z_first_below
        nfl_exists = np.isfinite(z_first_above) & np.isfinite(z_first_below)

        # 폴백: (바이어스 없는) ILM + sigmaZ
        nfl[~nfl_exists] = ilm[~nfl_exists] + self.sigma[0]

        # 서브픽셀 (down 엣지)
        nfl = self._subpixel_edge_crossing(
            nfl, img, self.thr_nfl, sense="down", xy_mask=nfl_exists
        )
        return nfl  # bias는 segment()에서 일괄 적용

    # ----------------------- ISOS -----------------------
    def _find_isos(self, img: np.ndarray, nfl: np.ndarray) -> np.ndarray:
        L, W, H = img.shape

        # NFL 아래(z >= NFL + 6*sigmaZ)에서 A-scan별 최대값으로 정규화
        z_full = np.arange(L).reshape(L, 1, 1)  # 0..L-1
        nfl_rep = nfl.reshape(1, W, H)  # (1,W,H)
        search_mask_base = z_full >= (nfl_rep + 6 * self.sigma[0])
        # 마스크된 영역의 최대 (없으면 -inf -> 이후 eps로 대체)
        masked_vals = np.where(search_mask_base, img, -np.inf)
        max_a = np.max(masked_vals, axis=0, keepdims=True)  # (1,W,H)
        max_a[~np.isfinite(max_a)] = 1e-9  # eps
        img_norm = img / max_a  # (L,W,H)

        # z-미분 (L-1, W, H)
        z_grad = np.diff(img_norm, axis=0)

        # ISOS 탐색 마스크: grad 차원에 맞춰서 사용
        search_mask = search_mask_base[:-1, :, :]  # (L-1,W,H)
        mask_isos = (z_grad > self.thr_isos) & search_mask

        z_coords_grad = np.arange(1, L).reshape(-1, 1, 1)  # 1..L-1 (grad 인덱스 1-based)
        z3 = np.where(mask_isos, z_coords_grad, np.inf)
        isos = np.squeeze(np.min(z3, axis=0))  # (W,H)

        # 서브픽셀 (up 엣지) - z_grad에서 보정
        isos = self._subpixel_edge_crossing(isos, z_grad, self.thr_isos, sense="up")

        # 결측 보간
        bad = ~np.isfinite(isos)
        if np.any(bad):
            isos = self._fill_missing_smooth(isos, bad)

        return isos  # bias는 segment()에서 일괄 적용

    # -------------------- Subpixel helper --------------------
    def _subpixel_edge_crossing(
        self,
        Z: np.ndarray,            # (W,H) 실수 인덱스(1-based) or Inf
        V: np.ndarray,            # (L,W,H) 또는 (L-1,W,H) 데이터 (원본 또는 그라디언트)
        thr: float,
        sense: str = "up",        # 'up' or 'down'
        xy_mask: np.ndarray | None = None,  # (W,H) True 위치만 보정
    ) -> np.ndarray:
        """
        MATLAB subpixelEdgeCrossing을 Python으로 포팅.
        - z-1, z에서 쓰는 선형 보간.
        - sense='up'  : V(z-1) < thr, V(z)   >= thr 일 때만 보정
        - sense='down': V(z-1) >= thr, V(z)  <  thr 일 때만 보정
        """
        W, H = Z.shape
        Z_out = Z.copy()

        valid = np.isfinite(Z_out)
        if xy_mask is not None:
            valid = valid & xy_mask
        if not np.any(valid):
            return Z_out

        # 0-based 정수 인덱스로 변환 (z_int-1이 접근되므로 최소 1)
        z_int = np.floor(Z_out[valid]).astype(int) - 1
        # V 길이에 맞춰 clamp (z_int, z_int+1 사용)
        z_int = np.clip(z_int, 0, V.shape[0] - 2)

        # 유효 좌표 (W,H) 중 valid인 좌표만 추출
        w_idx, h_idx = np.where(valid)

        v1 = V[z_int, w_idx, h_idx]       # V(z_int)
        v2 = V[z_int + 1, w_idx, h_idx]   # V(z_int+1)

        # 엣지 방향 조건
        if sense.lower() == "up":
            orient_ok = (v1 < thr) & (v2 >= thr)
        elif sense.lower() == "down":
            orient_ok = (v1 >= thr) & (v2 < thr)
        else:
            orient_ok = np.ones_like(v1, dtype=bool)

        # 선형 보간
        denom = (v2 - v1)
        denom[np.abs(denom) < 1e-12] = 1e-12
        corr = (thr - v1) / denom
        in_unit = (corr >= 0.0) & (corr < 1.0)

        ok = orient_ok & in_unit
        if np.any(ok):
            subpix = (z_int + 1).astype(np.float64) + corr  # (z_int+1) + corr  (1-based)
            # 결과 기록
            Z_sel = Z_out[valid]
            Z_sel[ok] = subpix[ok]
            Z_out[valid] = Z_sel

        return Z_out

    # -------------------- Fill-missing helper --------------------
    def _fill_missing_smooth(self, Z: np.ndarray, miss: np.ndarray) -> np.ndarray:
        """
        결측치를 주변 값으로 부드럽게 보간.
        MATLAB scatteredInterpolant(x,y,z,'natural','linear') 유사:
        - 1차: LinearNDInterpolator(=piecewise linear)
        - 2차: NearestNDInterpolator로 남은 부분 보정
        """
        W, H = Z.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")

        known = ~miss & np.isfinite(Z)
        if not np.any(known):
            return np.zeros_like(Z)

        pts = np.column_stack((x[known].ravel(), y[known].ravel()))
        vals = Z[known].ravel()

        lin = LinearNDInterpolator(pts, vals, fill_value=np.nan)
        Z_lin = lin(x, y)

        # 남은 NaN 채우기 (nearest)
        nan_mask = np.isnan(Z_lin)
        if np.any(nan_mask):
            nei = NearestNDInterpolator(pts, vals)
            Z_lin[nan_mask] = nei(x, y)[nan_mask]

        Z_out = Z.copy()
        Z_out[miss] = Z_lin[miss]
        return Z_out