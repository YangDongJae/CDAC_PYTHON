# core/volume_process/processor.py
from __future__ import annotations
from pathlib import Path
from typing import List, Literal, Tuple

import torch
from config.context import OCTContext
from core.frame_process.utils import (
    read_fringe_frame_bin_gpu,             # single-frame loader
    load_volume_variable, stack_frames     # 이미 존재
)
from core.phase_stabilize import stabilize_phase
from core.isam.resample import resample_isam        # §3.2
from core.segment.prl_segment import segment_prl_nfl  # §3.3
from core.cao.cao_optimize import optimize_zernike_surrogate  # §3.4
from core.io.saver import Saver                               # §3.5


class VolumeProcessor:
    """
    GPU-friendly replica of MATLAB *processAndSaveVolume*.
    한 sub-volume(vol) 전체를 처리하여 TIF / MAT 결과를 저장합니다.
    """

    def __init__(self, ctx: OCTContext, data_path: Path, device="cuda"):
        self.ctx        = ctx
        self.info       = ctx.config
        self.data_path  = Path(data_path)
        self.dev        = torch.device(device)
        self.saver      = Saver(self.info, self.data_path.parent)
        if self.dev.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this system.")
            torch.cuda.set_device(self.dev.index or 0)

    # ──────────────────────────────────────────────────────────────── #
    def load_volume(self, vol: int) -> List[torch.Tensor]:
        return load_volume_variable(
            self.data_path, vol, self.info.model_dump(), device=self.dev
        )

    # ──────────────────────────────────────────────────────────────── #
    def _phase_stab(self, stack: torch.Tensor, vol: int) -> torch.Tensor:
        stack_corr, cumX, cumY = stabilize_phase(
            stack, self.info.model_dump(), dfile_name=str(self.data_path.with_suffix("")),
            vol_idx=vol, use_cuda=(self.dev.type == "cuda")
        )
        # cumPhase 저장 back-prop
        self.info.Motion.cumPhaseX = cumX.cpu().numpy()
        self.info.Motion.cumPhaseY = cumY.cpu().numpy()
        return stack_corr

    # ──────────────────────────────────────────────────────────────── #
    def _cao_isam(self, fringes: torch.Tensor, vol: int) -> Tuple[torch.Tensor, dict]:
        """
        CAO & ISAM 모듈을 순차 적용.
        반환: (프린지_보정본, 추가_메타)
        """
        meta: dict = {}
        if self.info.CAO.enable:
            z_cube, _ = optimize_zernike_surrogate(fringes, self.info, self.dev)
            meta["CAO_zernike_phase"] = z_cube

        if self.info.ISAM.enable:
            zf = self.info.zfMap[tuple(divmod(vol - 1, self.info.subdivFactors[0]))]
            fringes = resample_isam(fringes, zf, self.info, axis="x", device=self.dev)
            fringes = resample_isam(fringes, zf, self.info, axis="y", device=self.dev)

        return fringes, meta

    # ──────────────────────────────────────────────────────────────── #
    def process_and_save(self, vol: int):
        # ① Load
        frames = self.load_volume(vol)
        stack  = stack_frames(frames, mode="pad")   # (K, L, F)
        # ② Phase stabilise
        stack  = self._phase_stab(stack, vol)
        # ③ CAO / ISAM
        stack, meta = self._cao_isam(stack, vol)
        # ④ IFFT → img cube
        img = torch.fft.ifft(stack, dim=0)[: self.info.numImgPixels]
        # ⑤ Segmentation (ILM,NFL,ISOS,RPE)
        layers = segment_prl_nfl(img.abs().square(), self.info, self.dev)
        # ⑥ 저장
        self.saver.save_volume(vol, img, layers, meta)
        # ⑦ NaN / Inf 리포트
        n_nan = torch.isnan(stack).sum().item()
        n_inf = torch.isinf(stack).sum().item()
        print(f"   ✅ NaN={n_nan:,}  |  Inf={n_inf:,}")

    def run_all_volumes(self):
        for v in range(1, self.info.numVolumes + 1):
            self.process_and_save(vol=v)        